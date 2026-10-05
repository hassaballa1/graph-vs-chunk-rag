"""The one LLM call used everywhere, with a disk cache, and the shared answer prompt."""
import json
import os
import time

import requests

from src.common import CACHE, paragraph_text, sha

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
ANTHROPIC = "https://api.anthropic.com/v1/messages"
LLM_CACHE = CACHE / "llm"

ANSWER_PROMPT = """Answer the question using the context below.
Reply with the short answer only: a name, date, number, yes or no, or a short phrase of at most five words.
No explanation and no full sentence. If the context seems incomplete, still give your single best short answer.

Context:
{context}

Question: {question}
Answer:"""

NO_CONTEXT_PROMPT = """Answer the question from your own knowledge.
Reply with the short answer only: a name, date, number, yes or no, or a short phrase of at most five words.
No explanation and no full sentence. If unsure, still give your single best short answer.

Question: {question}
Answer:"""


def _ollama_request(prompt, cfg, json_mode):
    request = {
        "model": cfg["llm_model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": cfg["llm_temperature"],
            "seed": cfg["seed"],
            "num_ctx": cfg["llm_num_ctx"],
        },
    }
    if json_mode:
        request["format"] = "json"
    return f"{OLLAMA}/api/chat", {}, request


def _anthropic_request(prompt, cfg, json_mode):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is not set. Put it in .env (see .env.example).")
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    request = {
        "model": cfg["llm_model"],
        "max_tokens": 1500 if json_mode else 100,
        "temperature": cfg["llm_temperature"],
        "messages": [{"role": "user", "content": prompt}],
    }
    return ANTHROPIC, headers, request


def _parse(provider, body):
    if provider == "anthropic":
        text = "".join(b.get("text", "") for b in body["content"])
        usage = body["usage"]
        return text, usage["input_tokens"], usage["output_tokens"]
    return body["message"]["content"], body.get("prompt_eval_count", 0), body.get("eval_count", 0)


def call_llm(prompt, cfg, json_mode=False):
    """Returns {"text", "prompt_tokens", "output_tokens", "seconds", "cached"}.

    The cache key covers everything that changes the output (never the API key),
    so a cached reply is exactly what the model would return for this request.
    """
    provider = cfg["llm_provider"]
    build = _anthropic_request if provider == "anthropic" else _ollama_request
    url, headers, request = build(prompt, cfg, json_mode)
    key = sha(json.dumps([provider, request], sort_keys=True))
    path = LLM_CACHE / key[:2] / f"{key}.json"
    if path.exists():
        os.utime(path)  # mark as in use, so entries no run reads can be pruned
        with open(path) as f:
            return {**json.load(f), "cached": True}

    start = time.time()
    r = None
    for attempt in range(6):
        try:
            r = requests.post(url, json=request, headers=headers, timeout=600)
            if r.status_code in (429, 500, 502, 503, 529):
                raise requests.HTTPError(f"retryable {r.status_code}", response=r)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 5 or (r is not None and r.status_code in (400, 401, 403)):
                raise
            time.sleep(2 ** attempt)
    text, prompt_tokens, output_tokens = _parse(provider, r.json())
    result = {
        "text": text.strip(),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "seconds": round(time.time() - start, 3),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f)
    return {**result, "cached": False}


def answer(question, paragraphs, cfg):
    if paragraphs is None:
        prompt = NO_CONTEXT_PROMPT.format(question=question)
    else:
        context = "\n\n".join(paragraph_text(p) for p in paragraphs)
        prompt = ANSWER_PROMPT.format(context=context, question=question)
    text = call_llm(prompt, cfg)["text"]
    # Keep the first line and drop a trailing full stop; scoring ignores punctuation anyway.
    return text.splitlines()[0].strip().rstrip(".") if text else ""


def ensure_model(cfg):
    """Pull the configured model into the ollama container if it is missing."""
    if cfg["llm_provider"] != "ollama":
        return
    tags = requests.get(f"{OLLAMA}/api/tags", timeout=30).json().get("models", [])
    if any(m["name"] == cfg["llm_model"] for m in tags):
        return
    print(f"pulling {cfg['llm_model']} into ollama (one time)...", flush=True)
    with requests.post(
        f"{OLLAMA}/api/pull", json={"model": cfg["llm_model"]}, stream=True, timeout=None
    ) as r:
        r.raise_for_status()
        last = None
        for line in r.iter_lines():
            status = json.loads(line).get("status")
            if status != last and not status.startswith("pulling "):
                print(f"  {status}", flush=True)
            last = status
