# Penetration Hard-Constraint Ablation (EXP-13)

Run: clips=2, frames=all
Modes: D (off), A (warmup only), B (S2 only), C (both)

## Per-clip penetration summary (mm)

| Clip | Mode | T | fps | pen_max | pen_mean | pen_p95 | frames≥0.5mm | w_shrinks | s_shrinks | struct | Δqpos vs D (°) |
|------|------|--:|----:|--------:|---------:|--------:|-------------:|----------:|----------:|-------:|---------------:|
| subject_1/seg00 | D-IMx1111 | 166 | 75.2 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0.000 |
| subject_1/seg00 | A-IMx1111 | 166 | 91.5 | 54.05 | 31.20 | 53.70 | 118 | 456 | 0 | 0 | 0.014 |
| subject_1/seg00 | B-IMx1111 | 166 | 129.0 | 50.19 | 30.91 | 49.07 | 118 | 0 | 464 | 0 | 9.566 |
| subject_1/seg00 | C-IMx1111 | 166 | 119.0 | 39.04 | 19.57 | 38.18 | 118 | 428 | 428 | 0 | 12.894 |
| subject_3/seg00 | D-IMx1111 | 494 | 81.2 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0.000 |
| subject_3/seg00 | A-IMx1111 | 494 | 90.1 | 44.20 | 13.12 | 38.31 | 197 | 760 | 0 | 0 | 0.017 |
| subject_3/seg00 | B-IMx1111 | 494 | 105.1 | 48.22 | 13.95 | 44.58 | 197 | 0 | 752 | 0 | 8.073 |
| subject_3/seg00 | C-IMx1111 | 494 | 78.8 | 31.44 | 4.72 | 28.97 | 197 | 367 | 364 | 0 | 12.704 |
