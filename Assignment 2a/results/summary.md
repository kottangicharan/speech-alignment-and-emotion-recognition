# Alignment summary

## Per variant
| variant | files | words | phones | mean_phone_ms | min_len_phones | unk_words | words_>=1s | speech_log_likelihood | overall_log_likelihood | phone_duration_deviation | snr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| arpa | 6 | 241 | 1063 | 75.823 | 2 | 0 | 0 | -47.760 | -49.278 | 2.975 | 10.178 |
| arpa_adapted | 6 | 241 | 1063 | 75.823 | 2 | 0 | 0 | -47.751 | -49.283 | 2.969 | 10.178 |
| mfa | 6 | 241 | 1046 | 82.022 | 0 | 0 | 2 | -44.768 | -43.043 | 3.779 | 9.662 |
| mfa_nonnative | 6 | 241 | 1032 | 82.456 | 0 | 0 | 2 | -45.591 | -43.749 | 3.640 | 9.459 |

## Word-boundary agreement between variants (ms)
| variant_a | variant_b | mean_boundary_diff_ms | F2BJRLP1 | F2BJRLP2 | F2BJRLP3 | ISLE_SESS0131_BLOCKD02_01_sprt1 | ISLE_SESS0131_BLOCKD02_02_sprt1 | ISLE_SESS0131_BLOCKD02_03_sprt1 |
|---|---|---|---|---|---|---|---|---|
| arpa | arpa_adapted | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| arpa | mfa | 90.547 | 6.600 | 8.500 | 9.700 | 231.500 | 35.000 | 252.000 |
| arpa | mfa_nonnative | 88.550 | 8.900 | 12.200 | 11.700 | 231.500 | 9.000 | 258.000 |
| arpa_adapted | mfa | 90.547 | 6.600 | 8.500 | 9.700 | 231.500 | 35.000 | 252.000 |
| arpa_adapted | mfa_nonnative | 88.550 | 8.900 | 12.200 | 11.700 | 231.500 | 9.000 | 258.000 |
| mfa | mfa_nonnative | 8.809 | 5.200 | 7.400 | 4.200 | 0.000 | 30.000 | 6.000 |

## ISLE minimal pairs
| variant | file | word | word_ms | phones | vowel | vowel_ms |
|---|---|---|---|---|---|---|
| arpa | ISLE_SESS0131_BLOCKD02_01_sprt1 | white | 410 | W AY1 T | AY1 | 150 |
| arpa | ISLE_SESS0131_BLOCKD02_01_sprt1 | bait | 450 | B EY1 T | EY1 | 200 |
| arpa | ISLE_SESS0131_BLOCKD02_02_sprt1 | new | 410 | N Y UW1 | UW1 | 240 |
| arpa | ISLE_SESS0131_BLOCKD02_02_sprt1 | no | 470 | N OW1 | OW1 | 350 |
| arpa | ISLE_SESS0131_BLOCKD02_03_sprt1 | bad | 510 | B AE1 D | AE1 | 220 |
| arpa | ISLE_SESS0131_BLOCKD02_03_sprt1 | bed | 550 | B EH1 D | EH1 | 180 |
| arpa_adapted | ISLE_SESS0131_BLOCKD02_01_sprt1 | white | 410 | W AY1 T | AY1 | 150 |
| arpa_adapted | ISLE_SESS0131_BLOCKD02_01_sprt1 | bait | 450 | B EY1 T | EY1 | 200 |
| arpa_adapted | ISLE_SESS0131_BLOCKD02_02_sprt1 | new | 410 | N Y UW1 | UW1 | 240 |
| arpa_adapted | ISLE_SESS0131_BLOCKD02_02_sprt1 | no | 470 | N OW1 | OW1 | 350 |
| arpa_adapted | ISLE_SESS0131_BLOCKD02_03_sprt1 | bad | 510 | B AE1 D | AE1 | 220 |
| arpa_adapted | ISLE_SESS0131_BLOCKD02_03_sprt1 | bed | 550 | B EH1 D | EH1 | 180 |
| mfa | ISLE_SESS0131_BLOCKD02_01_sprt1 | white | 580 | w aj t | aj | 190 |
| mfa | ISLE_SESS0131_BLOCKD02_01_sprt1 | bait | 2425 | b ej t | ej | 200 |
| mfa | ISLE_SESS0131_BLOCKD02_02_sprt1 | new | 280 | ɲ ʉː | ʉː | 100 |
| mfa | ISLE_SESS0131_BLOCKD02_02_sprt1 | no | 500 | n ow | ow | 380 |
| mfa | ISLE_SESS0131_BLOCKD02_03_sprt1 | bad | 510 | b æ d | æ | 220 |
| mfa | ISLE_SESS0131_BLOCKD02_03_sprt1 | bed | 3030 | b ɛ d | ɛ | 190 |
| mfa_nonnative | ISLE_SESS0131_BLOCKD02_01_sprt1 | white | 580 | ʋ aj ʈ | aj | 190 |
| mfa_nonnative | ISLE_SESS0131_BLOCKD02_01_sprt1 | bait | 2425 | b e t | e | 200 |
| mfa_nonnative | ISLE_SESS0131_BLOCKD02_02_sprt1 | new | 430 | ɲ uː | uː | 250 |
| mfa_nonnative | ISLE_SESS0131_BLOCKD02_02_sprt1 | no | 500 | n əw | əw | 380 |
| mfa_nonnative | ISLE_SESS0131_BLOCKD02_03_sprt1 | bad | 480 | b a d | a | 220 |
| mfa_nonnative | ISLE_SESS0131_BLOCKD02_03_sprt1 | bed | 3030 | b ɛ d | ɛ | 190 |

## Anomalies: 8 (see anomalies.csv)