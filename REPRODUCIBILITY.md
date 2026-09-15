# Reproducibility notes

## The donor training recipe (audit finding)

The released donor checkpoints and every grid under `paper_results/` were trained with the recipe
now in `experiments/train_donor.py` at HEAD: SGD, learning rate 0.1, Nesterov momentum 0.9, weight
decay 4e-4, batch 256, five epochs of linear warm-up then a 0.1x step decay at 60% and 85% of 35
epochs, cross-entropy with label smoothing 0.1. This matches the paper, Appendix C.2.

An earlier commit of `train_donor.py` (the first release, `173931e`) used a different recipe: weight
decay 5e-4, plain momentum without Nesterov, no warm-up, and cross-entropy without label smoothing.
That version was a re-implementation for the public repository and never trained any released
checkpoint. The training was done by the project's internal driver, whose optimizer and schedule are:

```
opt = torch.optim.SGD(net.parameters(), 0.1, momentum=0.9, nesterov=True, weight_decay=4e-4, foreach=False)
ce  = nn.CrossEntropyLoss(label_smoothing=0.1)
def lr_at(ep, EP):
    if ep < 5: return (ep + 1) / 5                       # five-epoch linear warm-up
    return 0.1 ** sum(ep >= int(EP * f) for f in (0.6, 0.85))
```

which is identical to the current `build_optimizer` and `lr_scale` in `train_donor.py`. The released
`.pt` files store the model `state_dict` only, not the optimizer state, so the recipe cannot be read
back from a checkpoint; the evidence is this driver source together with the per-epoch validation
lines in the training logs, whose accuracy ramps over the first five epochs as the warm-up predicts.

Because the checkpoints came from the current recipe, `train_donor.py` was not reverted. Instead
`tests/test_recipe.py` asserts the optimizer construction (`build_optimizer`) and the `lr_scale`
warm-up-plus-two-decays schedule, so the recipe cannot drift again.

## Flip-augmentation in-language number

`paper_results/in_language/held_out_signer_probe.json` holds the frozen-feature linear-probe accuracy
on the nine held-out AUTSL signers for the standard, canonical and unconditional-flip donors. The
flip-augmentation donors (seeds 40-42) are not in it: their in-language figure in the paper
(80.6 +/- 1.1) is a different instrument, the epoch-34 held-out training-validation accuracy, and is
reported against the standard donors' training-validation accuracy (75.9 +/- 1.2), not against the
linear probe. The two instruments are not interchangeable, so the flip-augmentation seeds are left
out of the probe file rather than mixed into it. There is no `flip_aug_in_language.json` in the
repository and nothing references one.
