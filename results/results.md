# Results

150 HotpotQA questions (100 bridge, 50 comparison), generator `claude-haiku-4-5-20251001`, embeddings `sentence-transformers/all-MiniLM-L6-v2`, k = 5, context cap 1200 tokens.

| Pipeline | Recall@5 | Full support@5 | F1 all | F1 bridge | F1 comparison | EM | Index tokens | Index time | Latency |
|---|---|---|---|---|---|---|---|---|---|
| Chunk | 78.3 | 60.0 | 59.2 | 57.9 | 61.7 | 44.7 | 0 | 20 s | 113 ms |
| Graph | 75.7 | 57.3 | 58.8 | 52.7 | 71.0 | 44.7 | 1,193,425 | 664 s | 120 ms |
| Graph (hub-pruned) | 75.3 | 56.7 | 58.9 | 52.8 | 71.0 | 45.3 | 1,193,425 | 664 s | 118 ms |

No-context floor (model answers from memory): F1 45.2, EM 35.3.

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
| Graph (hub-pruned) | recall | all | -3.0 | [-8.0, 1.3] | yes |
| Graph (hub-pruned) | recall | bridge | -7.5 | [-14.0, -2.0] | no |
| Graph (hub-pruned) | recall | comparison | 6.0 | [0.0, 12.0] | yes |
| Graph (hub-pruned) | full_support | all | -3.3 | [-10.7, 4.0] | yes |
| Graph (hub-pruned) | full_support | bridge | -10.0 | [-19.0, -1.0] | no |
| Graph (hub-pruned) | full_support | comparison | 10.0 | [-2.0, 22.0] | yes |
| Graph (hub-pruned) | f1 | all | -0.3 | [-5.3, 4.7] | yes |
| Graph (hub-pruned) | f1 | bridge | -5.1 | [-11.3, 0.5] | yes |
| Graph (hub-pruned) | f1 | comparison | 9.3 | [1.7, 17.6] | no |
