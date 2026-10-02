# CosyVoice3 voice-leak measurement report

Generated 2026-10-02T18:52:26+00:00 from `/home/ubuntu/cv3_out/runs`.

Wrong = argmax cosine over all K references is not the requested voice. `agree` = both scorers name the same wrong voice. Leak = agree count > 0 and one-sided Fisher p < 0.01 vs the C=1 cell. CIs are Wilson 95%.

| commit | mode | overlay | K | C | seeds | req | err | CAM++ wrong | WavLM wrong | agree | agree CI | Fisher p | leak | WER mean | WER>0.2 | dur>1.3x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1ea0ee3 | async | - | 2 | 8 | 1,2,3 | 360 | 0 | 0/360 | 0/360 | 0/360 | [0.000, 0.011] |  | no-C1-baseline | 0.0012 | 0 | 6 |
| bbee488 | async | - | 2 | 1 | 1 | 120 | 0 | 0/120 | 0/120 | 0/120 | [0.000, 0.031] |  |  | 0.0012 | 0 | 4 |
| bbee488 | async | - | 2 | 8 | 1,2,3 | 360 | 0 | 0/360 | 1/360 | 0/360 | [0.000, 0.011] | 1.00e+00 | no | 0.0061 | 1 | 5 |
| bbee488 | async | - | 8 | 1 | 1 | 120 | 0 | 0/120 | 0/120 | 0/120 | [0.000, 0.031] |  |  | 0.0057 | 0 | 3 |
| bbee488 | async | - | 8 | 8 | 1,2,3 | 360 | 0 | 0/360 | 0/360 | 0/360 | [0.000, 0.011] | 1.00e+00 | no | 0.0053 | 0 | 5 |
| bbee488 | async | mns4 | 8 | 8 | 1,2,3 | 360 | 0 | 0/360 | 0/360 | 0/360 | [0.000, 0.011] | 1.00e+00 | no | 0.0032 | 0 | 3 |
| bbee488 | async | prefix | 8 | 8 | 1,2,3 | 360 | 335 | 0/25 | 0/25 | 0/25 | [0.000, 0.133] | 1.00e+00 | no | 0.0 | 0 | 0 |
| c85f1a4 | async | - | 2 | 1 | 1 | 120 | 0 | 0/120 | 0/120 | 0/120 | [0.000, 0.031] |  |  | 0.0012 | 0 | 4 |
| c85f1a4 | async | - | 2 | 8 | 1,2,3 | 360 | 0 | 42/360 | 43/360 | 42/360 | [0.087, 0.154] | 2.98e-06 | YES | 0.0066 | 2 | 7 |
| c85f1a4 | async | - | 8 | 1 | 1 | 120 | 0 | 0/120 | 0/120 | 0/120 | [0.000, 0.031] |  |  | 0.0057 | 0 | 3 |
| c85f1a4 | async | - | 8 | 8 | 1,2,3 | 360 | 0 | 95/360 | 92/360 | 90/360 | [0.208, 0.297] | 2.17e-13 | YES | 0.0064 | 0 | 2 |

### Confusion campplus: bbee488 async overlay=- K=8 C=1 (rows requested, cols predicted)

```
               f237     f2961     f5142     f6829     m1188     m7127     m8224     m8230
     f237    15         0         0         0         0         0         0         0    
    f2961     0        15         0         0         0         0         0         0    
    f5142     0         0        15         0         0         0         0         0    
    f6829     0         0         0        15         0         0         0         0    
    m1188     0         0         0         0        15         0         0         0    
    m7127     0         0         0         0         0        15         0         0    
    m8224     0         0         0         0         0         0        15         0    
    m8230     0         0         0         0         0         0         0        15    
```

### Confusion wavlm: bbee488 async overlay=- K=8 C=1 (rows requested, cols predicted)

```
               f237     f2961     f5142     f6829     m1188     m7127     m8224     m8230
     f237    15         0         0         0         0         0         0         0    
    f2961     0        15         0         0         0         0         0         0    
    f5142     0         0        15         0         0         0         0         0    
    f6829     0         0         0        15         0         0         0         0    
    m1188     0         0         0         0        15         0         0         0    
    m7127     0         0         0         0         0        15         0         0    
    m8224     0         0         0         0         0         0        15         0    
    m8230     0         0         0         0         0         0         0        15    
```

### Confusion campplus: bbee488 async overlay=- K=8 C=8 (rows requested, cols predicted)

```
               f237     f2961     f5142     f6829     m1188     m7127     m8224     m8230
     f237    45         0         0         0         0         0         0         0    
    f2961     0        45         0         0         0         0         0         0    
    f5142     0         0        45         0         0         0         0         0    
    f6829     0         0         0        45         0         0         0         0    
    m1188     0         0         0         0        45         0         0         0    
    m7127     0         0         0         0         0        45         0         0    
    m8224     0         0         0         0         0         0        45         0    
    m8230     0         0         0         0         0         0         0        45    
```

### Confusion wavlm: bbee488 async overlay=- K=8 C=8 (rows requested, cols predicted)

```
               f237     f2961     f5142     f6829     m1188     m7127     m8224     m8230
     f237    45         0         0         0         0         0         0         0    
    f2961     0        45         0         0         0         0         0         0    
    f5142     0         0        45         0         0         0         0         0    
    f6829     0         0         0        45         0         0         0         0    
    m1188     0         0         0         0        45         0         0         0    
    m7127     0         0         0         0         0        45         0         0    
    m8224     0         0         0         0         0         0        45         0    
    m8230     0         0         0         0         0         0         0        45    
```

### Confusion campplus: bbee488 async overlay=mns4 K=8 C=8 (rows requested, cols predicted)

```
               f237     f2961     f5142     f6829     m1188     m7127     m8224     m8230
     f237    45         0         0         0         0         0         0         0    
    f2961     0        45         0         0         0         0         0         0    
    f5142     0         0        45         0         0         0         0         0    
    f6829     0         0         0        45         0         0         0         0    
    m1188     0         0         0         0        45         0         0         0    
    m7127     0         0         0         0         0        45         0         0    
    m8224     0         0         0         0         0         0        45         0    
    m8230     0         0         0         0         0         0         0        45    
```

### Confusion wavlm: bbee488 async overlay=mns4 K=8 C=8 (rows requested, cols predicted)

```
               f237     f2961     f5142     f6829     m1188     m7127     m8224     m8230
     f237    45         0         0         0         0         0         0         0    
    f2961     0        45         0         0         0         0         0         0    
    f5142     0         0        45         0         0         0         0         0    
    f6829     0         0         0        45         0         0         0         0    
    m1188     0         0         0         0        45         0         0         0    
    m7127     0         0         0         0         0        45         0         0    
    m8224     0         0         0         0         0         0        45         0    
    m8230     0         0         0         0         0         0         0        45    
```

### Confusion campplus: bbee488 async overlay=prefix K=8 C=8 (rows requested, cols predicted)

```
               f237     f2961     f5142     f6829     m1188     m7127     m8224     m8230
     f237     4         0         0         0         0         0         0         0    
    f2961     0         3         0         0         0         0         0         0    
    f5142     0         0         3         0         0         0         0         0    
    f6829     0         0         0         3         0         0         0         0    
    m1188     0         0         0         0         3         0         0         0    
    m7127     0         0         0         0         0         3         0         0    
    m8224     0         0         0         0         0         0         3         0    
    m8230     0         0         0         0         0         0         0         3    
```

### Confusion wavlm: bbee488 async overlay=prefix K=8 C=8 (rows requested, cols predicted)

```
               f237     f2961     f5142     f6829     m1188     m7127     m8224     m8230
     f237     4         0         0         0         0         0         0         0    
    f2961     0         3         0         0         0         0         0         0    
    f5142     0         0         3         0         0         0         0         0    
    f6829     0         0         0         3         0         0         0         0    
    m1188     0         0         0         0         3         0         0         0    
    m7127     0         0         0         0         0         3         0         0    
    m8224     0         0         0         0         0         0         3         0    
    m8230     0         0         0         0         0         0         0         3    
```

### Confusion campplus: c85f1a4 async overlay=- K=8 C=1 (rows requested, cols predicted)

```
               f237     f2961     f5142     f6829     m1188     m7127     m8224     m8230
     f237    15         0         0         0         0         0         0         0    
    f2961     0        15         0         0         0         0         0         0    
    f5142     0         0        15         0         0         0         0         0    
    f6829     0         0         0        15         0         0         0         0    
    m1188     0         0         0         0        15         0         0         0    
    m7127     0         0         0         0         0        15         0         0    
    m8224     0         0         0         0         0         0        15         0    
    m8230     0         0         0         0         0         0         0        15    
```

### Confusion wavlm: c85f1a4 async overlay=- K=8 C=1 (rows requested, cols predicted)

```
               f237     f2961     f5142     f6829     m1188     m7127     m8224     m8230
     f237    15         0         0         0         0         0         0         0    
    f2961     0        15         0         0         0         0         0         0    
    f5142     0         0        15         0         0         0         0         0    
    f6829     0         0         0        15         0         0         0         0    
    m1188     0         0         0         0        15         0         0         0    
    m7127     0         0         0         0         0        15         0         0    
    m8224     0         0         0         0         0         0        15         0    
    m8230     0         0         0         0         0         0         0        15    
```

### Confusion campplus: c85f1a4 async overlay=- K=8 C=8 (rows requested, cols predicted)

```
               f237     f2961     f5142     f6829     m1188     m7127     m8224     m8230
     f237    41         1         2         0         0         0         0         1    
    f2961     1        35         7         1         0         0         0         1    
    f5142     2         6        30         6         0         0         0         1    
    f6829     0         1         9        26         8         1         0         0    
    m1188     0         1         2         7        29         6         0         0    
    m7127     0         0         4         1        12        26         2         0    
    m8224     0         0         1         0         1         2        41         0    
    m8230     1         1         0         0         0         0         6        37    
```

### Confusion wavlm: c85f1a4 async overlay=- K=8 C=8 (rows requested, cols predicted)

```
               f237     f2961     f5142     f6829     m1188     m7127     m8224     m8230
     f237    41         1         2         0         0         0         0         1    
    f2961     1        35         7         1         0         0         0         1    
    f5142     2         6        30         6         0         0         0         1    
    f6829     0         1         9        26         8         1         0         0    
    m1188     0         1         2         7        29         6         0         0    
    m7127     0         0         4         1        12        27         1         0    
    m8224     0         0         1         0         1         0        43         0    
    m8230     1         1         0         0         2         0         4        37    
```

## Request errors

- `/home/ubuntu/cv3_out/runs/bbee488_async_prefix/K8_C8_s1`: 95 errors, e.g. `HTTP 500: {"error":{"message":"EngineCore encountered an issue. See stack trace (above) for the root cause.","type":"InternalServerError","param":null,"code":500,"request_id":"speech-89d6331189548af2"`
- `/home/ubuntu/cv3_out/runs/bbee488_async_prefix/K8_C8_s2`: 120 errors, e.g. `HTTP 500: {"error":{"message":"Stage-0 has no live replica","type":"InternalServerError","param":null,"code":500,"request_id":"speech-a0d6a26aeeb47301","error_stage_id":0}}`
- `/home/ubuntu/cv3_out/runs/bbee488_async_prefix/K8_C8_s3`: 120 errors, e.g. `HTTP 500: {"error":{"message":"Stage-0 has no live replica","type":"InternalServerError","param":null,"code":500,"request_id":"speech-8c8323c854e904c5","error_stage_id":0}}`
