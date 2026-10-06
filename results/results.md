# Results

150 HotpotQA questions (100 bridge, 50 comparison), generator `claude-haiku-4-5-20251001`, embeddings `sentence-transformers/all-MiniLM-L6-v2`, k = 5, context cap 1200 tokens.

| Pipeline | Recall@5 | Full support@5 | F1 all | F1 bridge | F1 comparison | EM | Index tokens | Index time | Latency |
|---|---|---|---|---|---|---|---|---|---|
| Chunk | 78.3 | 60.0 | 59.2 | 57.9 | 61.7 | 44.7 | 0 | 21 s | 117 ms |
| Graph | 75.7 | 57.3 | 58.8 | 52.7 | 71.0 | 44.7 | 1,193,425 | 680 s | 125 ms |
| Graph v2 | 82.7 | 70.0 | 64.3 | 59.5 | 74.0 | 48.7 | 1,193,425 | 652 s | 119 ms |
| Hybrid (fusion) | 76.0 | 57.3 | 58.8 | 51.7 | 73.0 | 45.3 | 1,193,425 | 701 s | 127 ms |
| Hybrid (routed) | 80.7 | 64.0 | 62.2 | 57.8 | 71.0 | 46.7 | 1,193,425 | 701 s | 119 ms |
| Graph (hub-pruned) | 75.3 | 56.7 | 58.9 | 52.8 | 71.0 | 45.3 | 1,193,425 | 680 s | 125 ms |

No-context floor (model answers from memory): F1 45.2, EM 35.3.

Oracle router (best of chunk and graph per question, a ceiling, not a pipeline): Recall@5 84.0, F1 64.1 (bridge 60.2, comparison 72.0).

## Difference from chunk, with paired bootstrap 95% intervals

| Pipeline | Metric | Subset | Difference | 95% interval | Crosses zero |
|---|---|---|---|---|---|
| Graph | recall | all | -2.7 | [-7.3, 1.7] | yes |
| Graph | recall | bridge | -7.0 | [-13.0, -1.5] | no |
| Graph | recall | comparison | 6.0 | [0.0, 12.0] | yes |
| Graph | full_support | all | -2.7 | [-10.0, 4.7] | yes |
| Graph | full_support | bridge | -9.0 | [-18.0, 0.0] | yes |
| Graph | full_support | comparison | 10.0 | [-2.0, 22.0] | yes |
| Graph | f1 | all | -0.4 | [-5.1, 4.4] | yes |
| Graph | f1 | bridge | -5.3 | [-10.8, -0.1] | no |
| Graph | f1 | comparison | 9.3 | [1.7, 17.6] | no |
| Graph v2 | recall | all | 4.3 | [-1.0, 9.3] | yes |
| Graph v2 | recall | bridge | 1.0 | [-5.5, 7.0] | yes |
| Graph v2 | recall | comparison | 11.0 | [4.0, 18.0] | no |
| Graph v2 | full_support | all | 10.0 | [2.0, 18.0] | no |
| Graph v2 | full_support | bridge | 5.0 | [-5.0, 15.0] | yes |
| Graph v2 | full_support | comparison | 20.0 | [8.0, 32.0] | no |
| Graph v2 | f1 | all | 5.2 | [-0.8, 11.3] | yes |
| Graph v2 | f1 | bridge | 1.6 | [-6.1, 9.0] | yes |
| Graph v2 | f1 | comparison | 12.3 | [1.7, 23.2] | no |
| Hybrid (fusion) | recall | all | -2.3 | [-6.7, 1.7] | yes |
| Hybrid (fusion) | recall | bridge | -6.0 | [-11.5, -1.5] | no |
| Hybrid (fusion) | recall | comparison | 5.0 | [-1.0, 11.0] | yes |
| Hybrid (fusion) | full_support | all | -2.7 | [-9.3, 4.0] | yes |
| Hybrid (fusion) | full_support | bridge | -8.0 | [-16.0, -1.0] | no |
| Hybrid (fusion) | full_support | comparison | 8.0 | [-4.0, 20.0] | yes |
| Hybrid (fusion) | f1 | all | -0.4 | [-4.9, 4.2] | yes |
| Hybrid (fusion) | f1 | bridge | -6.2 | [-11.4, -1.6] | no |
| Hybrid (fusion) | f1 | comparison | 11.3 | [2.9, 20.3] | no |
| Hybrid (routed) | recall | all | 2.3 | [0.0, 4.7] | yes |
| Hybrid (routed) | recall | bridge | 0.5 | [0.0, 1.5] | yes |
| Hybrid (routed) | recall | comparison | 6.0 | [0.0, 12.0] | yes |
| Hybrid (routed) | full_support | all | 4.0 | [0.0, 8.7] | yes |
| Hybrid (routed) | full_support | bridge | 1.0 | [0.0, 3.0] | yes |
| Hybrid (routed) | full_support | comparison | 10.0 | [-2.0, 22.0] | yes |
| Hybrid (routed) | f1 | all | 3.0 | [0.5, 6.0] | no |
| Hybrid (routed) | f1 | bridge | -0.1 | [-0.3, 0.0] | yes |
| Hybrid (routed) | f1 | comparison | 9.3 | [1.7, 17.6] | no |
| Graph (hub-pruned) | recall | all | -3.0 | [-8.0, 1.3] | yes |
| Graph (hub-pruned) | recall | bridge | -7.5 | [-14.0, -2.0] | no |
| Graph (hub-pruned) | recall | comparison | 6.0 | [0.0, 12.0] | yes |
| Graph (hub-pruned) | full_support | all | -3.3 | [-10.7, 4.0] | yes |
| Graph (hub-pruned) | full_support | bridge | -10.0 | [-19.0, -1.0] | no |
| Graph (hub-pruned) | full_support | comparison | 10.0 | [-2.0, 22.0] | yes |
| Graph (hub-pruned) | f1 | all | -0.3 | [-5.3, 4.7] | yes |
| Graph (hub-pruned) | f1 | bridge | -5.1 | [-11.3, 0.5] | yes |
| Graph (hub-pruned) | f1 | comparison | 9.3 | [1.7, 17.6] | no |

## Difference from graph, with paired bootstrap 95% intervals

| Pipeline | Metric | Subset | Difference | 95% interval | Crosses zero |
|---|---|---|---|---|---|
| Graph v2 | recall | all | 7.0 | [2.3, 11.7] | no |
| Graph v2 | recall | bridge | 8.0 | [1.5, 14.5] | no |
| Graph v2 | recall | comparison | 5.0 | [0.0, 10.0] | yes |
| Graph v2 | full_support | all | 12.7 | [5.3, 20.0] | no |
| Graph v2 | full_support | bridge | 14.0 | [4.0, 24.0] | no |
| Graph v2 | full_support | comparison | 10.0 | [0.0, 20.0] | yes |
| Graph v2 | f1 | all | 5.6 | [-0.3, 11.5] | yes |
| Graph v2 | f1 | bridge | 6.9 | [-1.1, 15.2] | yes |
| Graph v2 | f1 | comparison | 3.0 | [-4.9, 12.0] | yes |
| Hybrid (fusion) | recall | all | 0.3 | [-1.7, 2.3] | yes |
| Hybrid (fusion) | recall | bridge | 1.0 | [-1.5, 4.0] | yes |
| Hybrid (fusion) | recall | comparison | -1.0 | [-3.0, 0.0] | yes |
| Hybrid (fusion) | full_support | all | 0.0 | [-3.3, 4.0] | yes |
| Hybrid (fusion) | full_support | bridge | 1.0 | [-4.0, 6.0] | yes |
| Hybrid (fusion) | full_support | comparison | -2.0 | [-6.0, 0.0] | yes |
| Hybrid (fusion) | f1 | all | 0.0 | [-2.6, 2.7] | yes |
| Hybrid (fusion) | f1 | bridge | -1.0 | [-4.6, 2.6] | yes |
| Hybrid (fusion) | f1 | comparison | 2.0 | [0.0, 6.0] | yes |
| Hybrid (routed) | recall | all | 5.0 | [1.3, 9.0] | no |
| Hybrid (routed) | recall | bridge | 7.5 | [2.5, 13.5] | no |
| Hybrid (routed) | recall | comparison | 0.0 | [0.0, 0.0] | yes |
| Hybrid (routed) | full_support | all | 6.7 | [0.7, 13.3] | no |
| Hybrid (routed) | full_support | bridge | 10.0 | [1.0, 19.0] | no |
| Hybrid (routed) | full_support | comparison | 0.0 | [0.0, 0.0] | yes |
| Hybrid (routed) | f1 | all | 3.4 | [-0.0, 7.2] | yes |
| Hybrid (routed) | f1 | bridge | 5.2 | [0.0, 10.7] | no |
| Hybrid (routed) | f1 | comparison | 0.0 | [0.0, 0.0] | yes |
