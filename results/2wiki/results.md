# Results

160 2WikiMultiHopQA questions (40 comparison, 40 bridge-comparison, 40 compositional, 40 inference), generator `claude-haiku-4-5-20251001`, embeddings `sentence-transformers/all-MiniLM-L6-v2`, k = 5, context cap 1200 tokens.

| Pipeline | Recall@5 | Full support@5 | F1 all | F1 comparison | F1 bridge-comparison | F1 compositional | F1 inference | EM | Index tokens | Index time | Latency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Chunk | 69.1 | 39.4 | 46.8 | 62.5 | 67.4 | 17.9 | 39.3 | 40.6 | 0 | 13 s | 118 ms |
| Graph | 68.9 | 41.9 | 42.0 | 63.0 | 49.4 | 12.5 | 43.1 | 34.4 | 876,405 | 523 s | 123 ms |
| Graph v2 | 75.9 | 50.6 | 52.6 | 82.5 | 58.2 | 17.5 | 52.4 | 46.9 | 876,405 | 500 s | 116 ms |
| Hybrid (fusion) | 69.4 | 43.1 | 42.3 | 63.0 | 50.8 | 11.2 | 44.1 | 34.4 | 876,405 | 536 s | 125 ms |
| Hybrid (routed) | 68.0 | 38.8 | 42.4 | 63.0 | 49.4 | 17.9 | 39.3 | 35.0 | 876,405 | 536 s | 119 ms |

No-context floor (model answers from memory): F1 32.0, EM 25.0.

Oracle router (best of chunk and graph per question, a ceiling, not a pipeline): Recall@5 76.6, F1 56.3 (comparison 72.5, bridge-comparison 75.6, compositional 20.4, inference 56.9).

## Difference from chunk, with paired bootstrap 95% intervals

| Pipeline | Metric | Subset | Difference | 95% interval | Crosses zero |
|---|---|---|---|---|---|
| Graph | recall | all | -0.2 | [-4.4, 4.2] | yes |
| Graph | recall | comparison | 0.0 | [-8.8, 7.5] | yes |
| Graph | recall | bridge-comparison | -4.4 | [-10.6, 2.5] | yes |
| Graph | recall | compositional | -5.0 | [-12.5, 2.5] | yes |
| Graph | recall | inference | 8.8 | [-1.2, 18.8] | yes |
| Graph | full_support | all | 2.5 | [-5.0, 9.4] | yes |
| Graph | full_support | comparison | -2.5 | [-17.5, 12.5] | yes |
| Graph | full_support | bridge-comparison | 0.0 | [0.0, 0.0] | yes |
| Graph | full_support | compositional | -2.5 | [-15.0, 10.0] | yes |
| Graph | full_support | inference | 15.0 | [-2.5, 32.5] | yes |
| Graph | f1 | all | -4.8 | [-11.7, 2.4] | yes |
| Graph | f1 | comparison | 0.5 | [-12.5, 13.0] | yes |
| Graph | f1 | bridge-comparison | -17.9 | [-33.1, -2.0] | no |
| Graph | f1 | compositional | -5.4 | [-15.0, 3.8] | yes |
| Graph | f1 | inference | 3.8 | [-12.4, 19.7] | yes |
| Graph v2 | recall | all | 6.9 | [2.5, 11.2] | no |
| Graph v2 | recall | comparison | 12.5 | [5.0, 21.2] | no |
| Graph v2 | recall | bridge-comparison | 6.2 | [1.9, 10.6] | no |
| Graph v2 | recall | compositional | 2.5 | [-6.2, 11.2] | yes |
| Graph v2 | recall | inference | 6.2 | [-5.0, 17.5] | yes |
| Graph v2 | full_support | all | 11.2 | [3.8, 18.8] | no |
| Graph v2 | full_support | comparison | 22.5 | [10.0, 37.5] | no |
| Graph v2 | full_support | bridge-comparison | 2.5 | [0.0, 7.5] | yes |
| Graph v2 | full_support | compositional | 7.5 | [-7.5, 22.5] | yes |
| Graph v2 | full_support | inference | 12.5 | [-7.5, 32.5] | yes |
| Graph v2 | f1 | all | 5.9 | [-1.7, 13.7] | yes |
| Graph v2 | f1 | comparison | 20.0 | [5.0, 35.0] | no |
| Graph v2 | f1 | bridge-comparison | -9.2 | [-26.1, 6.9] | yes |
| Graph v2 | f1 | compositional | -0.4 | [-8.3, 7.5] | yes |
| Graph v2 | f1 | inference | 13.1 | [-2.7, 29.1] | yes |
| Hybrid (fusion) | recall | all | 0.3 | [-3.4, 4.4] | yes |
| Hybrid (fusion) | recall | comparison | 0.0 | [-8.8, 7.5] | yes |
| Hybrid (fusion) | recall | bridge-comparison | -5.0 | [-11.2, 1.2] | yes |
| Hybrid (fusion) | recall | compositional | -3.8 | [-10.0, 2.5] | yes |
| Hybrid (fusion) | recall | inference | 10.0 | [1.2, 18.8] | no |
| Hybrid (fusion) | full_support | all | 3.8 | [-2.5, 10.0] | yes |
| Hybrid (fusion) | full_support | comparison | -2.5 | [-17.5, 12.5] | yes |
| Hybrid (fusion) | full_support | bridge-comparison | 0.0 | [0.0, 0.0] | yes |
| Hybrid (fusion) | full_support | compositional | 0.0 | [-10.0, 10.0] | yes |
| Hybrid (fusion) | full_support | inference | 17.5 | [2.5, 32.5] | no |
| Hybrid (fusion) | f1 | all | -4.5 | [-10.3, 1.9] | yes |
| Hybrid (fusion) | f1 | comparison | 0.5 | [-12.5, 13.0] | yes |
| Hybrid (fusion) | f1 | bridge-comparison | -16.5 | [-31.1, -1.2] | no |
| Hybrid (fusion) | f1 | compositional | -6.7 | [-15.0, 0.0] | yes |
| Hybrid (fusion) | f1 | inference | 4.8 | [-6.5, 15.8] | yes |
| Hybrid (routed) | recall | all | -1.1 | [-3.8, 1.6] | yes |
| Hybrid (routed) | recall | comparison | 0.0 | [-8.8, 7.5] | yes |
| Hybrid (routed) | recall | bridge-comparison | -4.4 | [-10.6, 2.5] | yes |
| Hybrid (routed) | recall | compositional | 0.0 | [0.0, 0.0] | yes |
| Hybrid (routed) | recall | inference | 0.0 | [0.0, 0.0] | yes |
| Hybrid (routed) | full_support | all | -0.6 | [-5.0, 3.1] | yes |
| Hybrid (routed) | full_support | comparison | -2.5 | [-17.5, 12.5] | yes |
| Hybrid (routed) | full_support | bridge-comparison | 0.0 | [0.0, 0.0] | yes |
| Hybrid (routed) | full_support | compositional | 0.0 | [0.0, 0.0] | yes |
| Hybrid (routed) | full_support | inference | 0.0 | [0.0, 0.0] | yes |
| Hybrid (routed) | f1 | all | -4.4 | [-9.8, 0.8] | yes |
| Hybrid (routed) | f1 | comparison | 0.5 | [-12.5, 13.0] | yes |
| Hybrid (routed) | f1 | bridge-comparison | -17.9 | [-33.1, -2.0] | no |
| Hybrid (routed) | f1 | compositional | 0.0 | [0.0, 0.0] | yes |
| Hybrid (routed) | f1 | inference | 0.0 | [0.0, 0.0] | yes |

## Difference from graph, with paired bootstrap 95% intervals

| Pipeline | Metric | Subset | Difference | 95% interval | Crosses zero |
|---|---|---|---|---|---|
| Graph v2 | recall | all | 7.0 | [3.0, 11.1] | no |
| Graph v2 | recall | comparison | 12.5 | [5.0, 20.0] | no |
| Graph v2 | recall | bridge-comparison | 10.6 | [5.0, 16.9] | no |
| Graph v2 | recall | compositional | 7.5 | [-1.2, 15.0] | yes |
| Graph v2 | recall | inference | -2.5 | [-11.3, 7.5] | yes |
| Graph v2 | full_support | all | 8.8 | [1.2, 16.2] | no |
| Graph v2 | full_support | comparison | 25.0 | [10.0, 40.0] | no |
| Graph v2 | full_support | bridge-comparison | 2.5 | [0.0, 7.5] | yes |
| Graph v2 | full_support | compositional | 10.0 | [-2.5, 22.6] | yes |
| Graph v2 | full_support | inference | -2.5 | [-20.1, 17.5] | yes |
| Graph v2 | f1 | all | 10.6 | [3.5, 17.6] | no |
| Graph v2 | f1 | comparison | 19.5 | [4.5, 36.0] | no |
| Graph v2 | f1 | bridge-comparison | 8.7 | [-8.3, 25.6] | yes |
| Graph v2 | f1 | compositional | 5.0 | [0.0, 11.2] | yes |
| Graph v2 | f1 | inference | 9.3 | [-3.8, 23.1] | yes |
| Hybrid (fusion) | recall | all | 0.5 | [-1.6, 2.5] | yes |
| Hybrid (fusion) | recall | comparison | 0.0 | [0.0, 0.0] | yes |
| Hybrid (fusion) | recall | bridge-comparison | -0.6 | [-1.9, 0.0] | yes |
| Hybrid (fusion) | recall | compositional | 1.2 | [-2.5, 5.0] | yes |
| Hybrid (fusion) | recall | inference | 1.2 | [-5.0, 7.5] | yes |
| Hybrid (fusion) | full_support | all | 1.2 | [-2.5, 5.0] | yes |
| Hybrid (fusion) | full_support | comparison | 0.0 | [0.0, 0.0] | yes |
| Hybrid (fusion) | full_support | bridge-comparison | 0.0 | [0.0, 0.0] | yes |
| Hybrid (fusion) | full_support | compositional | 2.5 | [-5.0, 10.0] | yes |
| Hybrid (fusion) | full_support | inference | 2.5 | [-10.0, 15.0] | yes |
| Hybrid (fusion) | f1 | all | 0.3 | [-3.3, 3.7] | yes |
| Hybrid (fusion) | f1 | comparison | 0.0 | [0.0, 0.0] | yes |
| Hybrid (fusion) | f1 | bridge-comparison | 1.4 | [0.0, 4.2] | yes |
| Hybrid (fusion) | f1 | compositional | -1.2 | [-7.5, 3.8] | yes |
| Hybrid (fusion) | f1 | inference | 1.0 | [-10.8, 12.5] | yes |
| Hybrid (routed) | recall | all | -0.9 | [-4.4, 2.5] | yes |
| Hybrid (routed) | recall | comparison | 0.0 | [0.0, 0.0] | yes |
| Hybrid (routed) | recall | bridge-comparison | 0.0 | [0.0, 0.0] | yes |
| Hybrid (routed) | recall | compositional | 5.0 | [-2.5, 12.5] | yes |
| Hybrid (routed) | recall | inference | -8.8 | [-18.8, 1.2] | yes |
| Hybrid (routed) | full_support | all | -3.1 | [-8.8, 2.5] | yes |
| Hybrid (routed) | full_support | comparison | 0.0 | [0.0, 0.0] | yes |
| Hybrid (routed) | full_support | bridge-comparison | 0.0 | [0.0, 0.0] | yes |
| Hybrid (routed) | full_support | compositional | 2.5 | [-10.0, 15.0] | yes |
| Hybrid (routed) | full_support | inference | -15.0 | [-32.5, 2.5] | yes |
| Hybrid (routed) | f1 | all | 0.4 | [-4.5, 5.0] | yes |
| Hybrid (routed) | f1 | comparison | 0.0 | [0.0, 0.0] | yes |
| Hybrid (routed) | f1 | bridge-comparison | 0.0 | [0.0, 0.0] | yes |
| Hybrid (routed) | f1 | compositional | 5.4 | [-3.8, 15.0] | yes |
| Hybrid (routed) | f1 | inference | -3.8 | [-19.7, 12.4] | yes |
