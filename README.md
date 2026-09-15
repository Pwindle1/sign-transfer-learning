# signcanon

Code and results for *Using Sign Phonology to Improve Few-Shot Cross-Lingual Transfer Learning for
Sign Language Recognition* (Matteo Lanza), submitted to the Workshop on Sign Language Processing,
WSLP 2026.

Pose-based sign language recognisers represent the two hands as separate halves of a skeleton
graph, so a left-dominant and a right-dominant signer produce the same sign as mirror images.
Which hand leads is a fact about the signer, not the sign. Dominance canonicalization mirrors any
training clip whose left hand carries more than 1.2 times the right hand's motion energy, swapping
the hand blocks and the left/right body nodes and negating x. The mirror is an exact automorphism
of the skeleton graph, so the operation is lossless. Donors are CTR-GCN models trained on Turkish
Sign Language (AUTSL) and adapted with k labelled clips per sign to six other sign languages.

## Results

Transfer, four standard vs four canonical donors, 90-cell grid (6 recipients x k in {1, 5, 10} x
5 episode seeds), fine-tune accuracy in percent:

| recipient | standard | canonical | gain FT | gain proto |
|---|---|---|---|---|
| BdSLW60 (Bangla) | 38.33 | 47.83 | +9.50 | +9.86 |
| INCLUDE (Indian) | 55.62 | 60.85 | +5.23 | +3.95 |
| LSA64 (Argentinian) | 60.38 | 66.25 | +5.88 | +6.78 |
| Slovo (Russian) | 33.10 | 40.36 | +7.26 | +5.14 |
| SSL400 (Sinhala) | 31.01 | 35.59 | +4.58 | +2.93 |
| WLASL-300 (American) | 13.80 | 17.41 | +3.62 | +2.49 |
| pooled | 38.70 | 44.72 | +6.01 | +5.19 |

Noise floor (pooled same-recipe donor spread) 0.80 fine-tune, 1.42 prototypes; 95% CI on the effect
[+4.62, +7.40] and [+2.73, +7.65]; exact permutation p = 1/70 in both regimes with complete
separation (every canonical donor above every standard donor). On held-out AUTSL signers the
standard donors score 75.8 +/- 1.5 and the canonical donors 75.0 +/- 0.5, an in-language cost of
-0.77 points, 95% CI [-2.77, +1.23]. Transfer grows with donor size at 5.82 points per e-fold, the
standard arm's log-linear fit over 7.9k to 26.2k donor clips.

Donor-training variants with identical data, differing only in which clips are mirrored (paper
Table 2; 54-cell grid, fine-tune mean over the donors of each arm):

| variant | accuracy | share of canonical gain |
|---|---|---|
| natural corpus | 39.01 +/- 1.05 | 0% |
| gloss-blocked | 42.42 +/- 0.88 | 57% |
| signer-blocked | 42.59 +/- 0.19 | 60% |
| random fraction f = 0.568 | 43.69 +/- 0.51 | 79% |
| signer-majority | 44.51 +/- 1.63 | 92% |
| canonical | 44.96 +/- 0.46 | 100% |
| flip augmentation p = 0.5 | 45.31 +/- 1.52 | 106% |
| random fraction f = 0.20 | 46.92 +/- 0.67 | 133% |

A further control mirrors every clip unconditionally, leaving every coupling identical to the
natural corpus; it reaches 44.58 +/- 1.07, 0.38 points from canonical, inside the seed spread of
either arm (Section 6.1).

No-donor baseline (random initialisation, support set only, same 90 cells): 28.10 pooled, above
random selection for every recipient but below the standard donor by 10.6 points on average
(Appendix A.5).

Tracking census over the seven corpora: 0.12 to 65.1% of clips never have a hand detected, and the
share of rule-decidable clips whose decision was forced by a missing hand runs from 0.3% (AUTSL) to
80.7% (LSA64). Restricting the transfer comparison to clips with both hands tracked moves the
fine-tune effect from +5.72 to +4.77 with separation intact.

## Reproduce the numbers from the shipped results

```bash
pip install -e .
python experiments/compare_donors.py \
  --results paper_results/cells/s{0,1,2,3}_standard_rawrecipients.json \
            paper_results/cells/s{90,94,12,13}_canonical_canonrecipients.json \
  --control 0 1 2 3 --treatment 90 94 12 13        # add --method proto for the second regime
python experiments/audit_tracking.py --corpora data/*.npz   # the census, given the staged corpora
```

`paper_results/README.md` maps every file to the table it feeds.

## Contents

```
src/signcanon/      operator, variant recipes, whole-clip loss census (validity.py), splits, episodes, adaptation, statistics
experiments/        train_donor.py, evaluate_transfer.py, compare_donors.py, audit_tracking.py
paper_results/      every cell-level grid and aggregate in the paper, as JSON
splits/             donor split and the support/test clips of every episode
staging/            featurisation, extraction scripts, staging rules, Slovo subset file
SEEDS.md            seed-to-recipe map (paper Table 8)
tests/              33 property tests (PYTHONPATH=src pytest)
```

The recogniser is the official CTR-GCN (Chen et al., 2021), unmodified: clone
https://github.com/Uason-Chen/CTR-GCN and pass its path as `--ctrgcn-repo`. Each corpus is one
`.npz` with `X` float32 (N, 2, 32, 49), `y` int (N,), `sg` str (N,); `staging/` documents how the
published corpora become these files. Pose data is not redistributed.

Donor training: 35 epochs, SGD lr 0.1, Nesterov momentum 0.9, weight decay 4e-4, batch 256, five
warm-up epochs then 0.1x decay at 60% and 85%, label smoothing 0.1, nine AUTSL signers held out.
Recipient adaptation: 30 epochs, SGD lr 0.01, momentum 0.9, weight decay 1e-4, batch 128. Canonical
donors evaluate on canonicalized recipients (`--canonical`); all other arms on raw recipients.
The recipe behind each seed is one `train_donor.py` flag, listed in `SEEDS.md`.

## Licence

MIT for the code in this repository. CTR-GCN and the corpora carry their own licences. Please cite
the paper (`CITATION.cff`).
