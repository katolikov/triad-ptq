# LLM results (WikiText-2 perplexity)

| model                    | method   |   bits |   ppl |   tok_per_sec |   calib_sec |   n_eval_tokens |
|:-------------------------|:---------|-------:|------:|--------------:|------------:|----------------:|
| SmolLM-135M              | AWQ-like |      4 | 23.85 |          41.2 |          26 |           32736 |
| SmolLM-135M              | FP32     |     32 | 18.87 |          38.6 |           0 |           32736 |
| SmolLM-135M              | RTN      |      4 | 26.6  |          42   |           1 |           32736 |
| SmolLM-135M              | TRIAD    |      4 | 21.56 |          38   |         213 |           32736 |
| SmolLM-360M              | AWQ-like |      4 | 16.6  |          32   |          54 |           32736 |
| SmolLM-360M              | FP32     |     32 | 14.07 |          31.7 |           0 |           32736 |
| SmolLM-360M              | RTN      |      4 | 17.29 |          32   |           4 |           32736 |
| SmolLM-360M              | TRIAD    |      4 | 15.79 |          29.3 |         843 |           32736 |
| TinyLlama-1.1B-Chat-v1.0 | FP32     |     32 |  8.45 |          20.3 |           0 |           16368 |
| TinyLlama-1.1B-Chat-v1.0 | RTN      |      4 |  8.87 |          20.2 |           6 |           16368 |
