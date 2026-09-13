# Seed-to-recipe map (paper Table 8)

Every donor checkpoint is identified by its seed alone; the seed fixes the recipe. Checkpoints are not
part of this repository (see `paper_results/` for every number they produced).

| seeds | recipe | `train_donor.py` flag | recipient frame at evaluation |
|---|---|---|---|
| 0-3 | standard donors (no transform) | *(none)* | raw |
| 90, 94, 12, 13 | dominance canonicalization (the canonical arm throughout) | `--canonical` | canonicalized |
| 91 | all-left mirror-image canonical donor (Section 3 mirror-pair test) | `--all-left` | canonicalized |
| 92-93 | second donor-subsample pair at 7,910 clips (release only) | `--subsample 7910` (+ `--canonical` for 93) | as arm |
| 40-42 | per-epoch flip augmentation, p = 0.5 | `--flip-aug 0.5` | raw |
| 43-45 | unconditional flip (every clip mirrored, no gate) | `--unconditional-flip` | raw |
| 50-57 | random-fraction sweep f ∈ {0, 0.20, 0.40, 0.568}; 50-53 one seed each, 54-55 f = 0.20 and 56-57 f = 0.568 replicates | `--mixture F --construction-seed S` | canonicalized |
| 60-63 | donor-subsample scaling pairs: 60/61 at 7,910 clips, 62/63 at 15,820 (even = standard, odd = canonical) | `--subsample N` (+ `--canonical`) | as arm |
| 70-72 | signer-blocked re-mirroring to the natural left fraction 0.568 | `--signer-blocked 0.568 --construction-seed S` | canonicalized |
| 75-77 | gloss-blocked re-mirroring to 0.568 | `--gloss-blocked 0.568 --construction-seed S` | canonicalized |
| 80-82 | signer-majority-vote canonicalization | `--signer-majority` | canonicalized |
| *(no donor seed)* | no-donor baseline: random initialisation, support set only, episode seeds 0-4 | `evaluate_transfer.py --from-scratch` | raw (and canonicalized, Appendix A.5) |

Construction seeds for the replicate arms: 54/55 use `--construction-seed 1`/`2`, 56/57 likewise;
70/71/72 and 75/76/77 use construction seeds 0/1/2. The training seed (`--seed`) is the table's seed.
Retired: seed 10, a canonicalized donor trained during a build race with the wrong skeleton graph; it
is excluded from every analysis and from this release.
