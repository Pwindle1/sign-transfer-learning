# paper_results — every number in the paper, as JSON

All files are the cell-level or aggregate outputs behind the paper's tables and figures, converted
to the schema this repository's scripts read. Nothing here was re-run for the release: these are
the production results, re-keyed.

## cells/ — cell-level grids

One file per (donor seed, recipient frame): `s<seed>_<arm>_<raw|canon>recipients.json`. Each row is
one cell of one regime:

```
donor               seed of the donor checkpoint (-1 = no-donor baseline)
arm                 recipe name (see SEEDS.md)
target              recipient corpus
k                   support clips per class (1, 5, 10)
eval_seed           episode seed (0-4 for the headline grid, 0-2 for the mechanism variants)
method              "full_ft" (fine-tune regime) or "proto" (frozen prototypes)
accuracy            top-1 accuracy in percent
n_classes           classes in the episode
recipient_canonical whether the recipient clips were canonicalized (the canonical donors' native frame)
split               "signer_disjoint" or "clip_only"
```

`MANIFEST.json` lists every file with its cell count and episode seeds. "Native" frames — the ones
every headline and variant number uses — are raw recipients for standard, flip-augmentation,
unconditional-flip and no-donor arms, and canonicalized recipients for every canonicalized donor.
The non-native files (`s0-3 ... canonrecipients`, `s90/94/12/13 ... rawrecipients`) are the
off-diagonal cells of the paper's 2×2 (Table 6). The no-donor files carry `accuracy_by_epochs` and
`epoch_ceiling` (Appendix A.5).

Reproduce the headline (Table 1 / Section 5):

```bash
python experiments/compare_donors.py \
  --results paper_results/cells/s{0,1,2,3}_standard_rawrecipients.json \
            paper_results/cells/s{90,94,12,13}_canonical_canonrecipients.json \
  --control 0 1 2 3 --treatment 90 94 12 13            # add --method proto for the prototype regime
```

Any variant against the natural donors on the 54-cell basis, e.g. the unconditional flip:

```bash
python experiments/compare_donors.py \
  --results paper_results/cells/s{0,1,2,3}_standard_rawrecipients.json \
            paper_results/cells/s{43,44,45}_unconditional_flip_rawrecipients.json \
  --control 0 1 2 3 --treatment 43 44 45 --eval-seeds 0 1 2
```

## in_language/ — the donor's own language

`held_out_signer_probe.json`: per donor, accuracy of a linear probe on frozen features over the nine
held-out AUTSL signers (Section 5, "canonicalization has a small cost in the donor's own language").

## aggregates/ — what the tables print

| file | feeds |
|---|---|
| `headline_4v4_90cell.json`, `headline_robustness.json`, `per_recipient_4v4.json` | Table 1, Section 5 |
| `variant_arms_54cell.json`, `variant_couplings.json` | Table 2, Section 6.1 |
| `tracking_census_and_couplings.json` | Table 3, Section 3, Appendix B.1 (both 0/0 conventions) |
| `donor_size_scaling.json` | Figure 3, Section 5 |
| `guard_icc_table.json` | Table 7, Section 6.2 |
| `selfaudit_fullft_cells.json`, `selfaudit_proto_cells.json` | the well-tracked-only re-scoring, Section 6.2 |
| `zeroing_probe_*.json` | the zeroing paragraph, Section 6.1 |
| `mirror_pair_and_audit_followups.json` | the 0.17-point mirror-pair test, Section 3 |
| `two_implementation_repro.json` | cross-implementation reproducibility, Appendix C |
