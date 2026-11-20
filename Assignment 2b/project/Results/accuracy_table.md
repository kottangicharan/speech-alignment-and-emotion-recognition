# Test accuracy (%) / macro-F1 (%)

| Model | Random 70/15/15 (3 seeds) | Speaker OAF→YAF | Speaker YAF→OAF | Speaker-independent mean |
|---|---|---|---|---|
| speech | 100.00 ± 0.00 / 100.00 ± 0.00 | 73.00 / 70.92 | 51.29 / 45.99 | 62.14 ± 15.35 / 58.45 ± 17.63 |
| text | 14.37 ± 0.14 / 4.03 ± 0.80 | 14.29 / 3.57 | 14.29 / 3.57 | 14.29 ± 0.00 / 3.57 ± 0.00 |
| fusion_gated | 99.92 ± 0.14 / 99.92 ± 0.14 | 60.07 / 53.77 | 59.21 / 52.88 | 59.64 ± 0.61 / 53.33 ± 0.63 |
| fusion_concat | 99.44 ± 0.14 / 99.45 ± 0.14 | 56.00 / 51.32 | 56.79 / 49.20 | 56.39 ± 0.56 / 50.26 ± 1.51 |
| speech_spknorm | 99.76 ± 0.24 / 99.76 ± 0.24 | 62.93 / 59.03 | 46.36 / 39.05 | 54.64 ± 11.72 / 49.04 ± 14.13 |
| fusion_gated_spknorm | 100.00 ± 0.00 / 100.00 ± 0.00 | 43.43 / 40.71 | 52.00 / 49.97 | 47.71 ± 6.06 / 45.34 ± 6.54 |
| fusion_concat_spknorm | 100.00 ± 0.00 / 100.00 ± 0.00 | 45.64 / 41.54 | 54.43 / 51.17 | 50.04 ± 6.21 / 46.35 ± 6.81 |

Chance = 14.29 % (7 balanced classes).

# Per-class F1 (%), mean over runs

## Random split

| Model | angry | disgust | fear | happy | neutral | pleasant_surprise | sad |
|---|---|---|---|---|---|---|---|
| speech | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| text | 8.3 | 0.0 | 0.0 | 8.3 | 3.3 | 8.3 | 0.0 |
| fusion_gated | 99.7 | 100.0 | 100.0 | 100.0 | 99.7 | 100.0 | 100.0 |
| fusion_concat | 100.0 | 99.7 | 99.7 | 99.4 | 98.9 | 99.2 | 99.2 |
| speech_spknorm | 100.0 | 99.2 | 100.0 | 100.0 | 100.0 | 99.2 | 100.0 |
| fusion_gated_spknorm | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| fusion_concat_spknorm | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

## Speaker-independent

| Model | angry | disgust | fear | happy | neutral | pleasant_surprise | sad |
|---|---|---|---|---|---|---|---|
| speech | 21.4 | 75.0 | 93.6 | 27.0 | 66.0 | 47.7 | 78.4 |
| text | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 25.0 |
| fusion_gated | 34.2 | 61.3 | 70.5 | 0.0 | 85.4 | 39.9 | 81.9 |
| fusion_concat | 35.1 | 55.1 | 90.4 | 0.9 | 78.8 | 6.2 | 85.4 |
| speech_spknorm | 12.0 | 70.1 | 91.7 | 0.4 | 63.6 | 34.5 | 71.0 |
| fusion_gated_spknorm | 5.6 | 83.2 | 61.1 | 0.3 | 82.2 | 9.8 | 75.2 |
| fusion_concat_spknorm | 13.3 | 81.5 | 81.2 | 0.0 | 63.0 | 41.7 | 43.7 |

# Fusion gate (mean weight on speech, 1 = speech only)

| Model | Split | gate |
|---|---|---|
| fusion_gated | random_s0 | 0.696 |
| fusion_gated | random_s1 | 0.680 |
| fusion_gated | random_s2 | 0.702 |
| fusion_gated | spk_OAF2YAF | 0.671 |
| fusion_gated | spk_YAF2OAF | 0.732 |
| fusion_gated_spknorm | random_s0 | 0.761 |
| fusion_gated_spknorm | random_s1 | 0.776 |
| fusion_gated_spknorm | random_s2 | 0.736 |
| fusion_gated_spknorm | spk_OAF2YAF | 0.742 |
| fusion_gated_spknorm | spk_YAF2OAF | 0.768 |