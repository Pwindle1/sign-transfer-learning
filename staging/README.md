# Staging: from published corpora to the (N, 2, 32, 49) packs

No pose data is redistributed (see the paper's Ethics Statement). These files reproduce the packs.

**Featurisation** (`featurize.py`, dependency-free): MediaPipe Holistic keypoints (75 points,
`model_complexity=2`) → 49 nodes × 2-D, shoulder-centred and shoulder-width-scaled, low-confidence
points (< 0.1) NaN-filled then forward/back-filled, resampled to 32 frames. z and confidences are
discarded.

**Sources.** AUTSL, WLASL-300, INCLUDE and LSA64 come from the OpenHands pose release
(`openhands_data.py`, loader); Slovo, BdSLW60 and SSL400 were extracted by us with MediaPipe
Holistic 0.10.21 (`slovo_extract.py`, `bangla_extract.py`, `ssl400_extract.py`). The four scripts are
the production versions verbatim, with their original absolute paths; edit the path constants at the
top of each before use.

**Staging rules** (paper Section 4), all fixed before any transfer measurement:

| corpus | rule |
|---|---|
| AUTSL (donor) | official train + test partitions; 274 training clips (0.9%) lost in extraction; validation signers unlabeled in the source release, none dropped |
| WLASL-300 | published pack as is |
| INCLUDE | published pack as is (263 classes) |
| LSA64 | published pack as is |
| Slovo | the 120-gloss × 20-clip subset in `slovo_subset120.csv` (2,400 clips, 106 signers) |
| BdSLW60 | trials capped at six per (signer, word) → 4,643 clips, all 18 signers |
| SSL400 | every gloss with ≥ 8 extracted clips → 127 glosses, 2,433 clips |

No subset was selected on tracking quality or on any measured outcome.
