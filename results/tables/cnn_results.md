# CNN results (ImageNetV2 matched-frequency)

| model           | method   |   bits |   top1 |   top5 |   calib_sec |   n_eval |
|:----------------|:---------|-------:|-------:|-------:|------------:|---------:|
| efficientnet_b0 | AWQ-like |      4 | 0.6208 | 0.835  |           5 |     5000 |
| efficientnet_b0 | FP32     |     32 | 0.6568 | 0.8626 |           0 |     5000 |
| efficientnet_b0 | RTN      |      4 | 0.6076 | 0.8312 |           1 |     5000 |
| efficientnet_b0 | TRIAD    |      4 | 0.6426 | 0.8556 |          24 |     5000 |
| mobilenet_v2    | AWQ-like |      4 | 0.5016 | 0.7486 |           7 |     5000 |
| mobilenet_v2    | FP32     |     32 | 0.5996 | 0.8232 |           0 |     5000 |
| mobilenet_v2    | RTN      |      4 | 0.4004 | 0.6552 |           1 |     5000 |
| mobilenet_v2    | TRIAD    |      4 | 0.569  | 0.8026 |          23 |     5000 |
| mobilevit_s     | AWQ-like |      4 | 0.446  | 0.6702 |          11 |     5000 |
| mobilevit_s     | FP32     |     32 | 0.6686 | 0.8744 |           0 |     5000 |
| mobilevit_s     | RTN      |      4 | 0.3398 | 0.579  |           2 |     5000 |
| mobilevit_s     | TRIAD    |      4 | 0.6264 | 0.8366 |          98 |     5000 |
