"""Train one donor. Every recipe in the paper's seed-to-recipe map (Table 8) is one flag:

  (none)                        standard donor                       seeds 0-3
  --canonical                   dominance canonicalization           seeds 90, 94, 12, 13
  --all-left                    all-left mirror-image donor          seed 91
  --mixture F                   random fraction f of left clips      seeds 50-57 (f in 0, .2, .4, .568)
  --signer-blocked F            whole signers mirrored back to f     seeds 70-72
  --gloss-blocked F             whole glosses mirrored back to f     seeds 75-77
  --signer-majority             signer-majority-vote canonicalization  seeds 80-82
  --flip-aug P                  per-epoch stochastic mirror, p = 0.5 seeds 40-42
  --unconditional-flip          every clip mirrored, no gate         seeds 43-45
  --subsample N                 class-balanced donor subsample       seeds 60-63 (7,910 / 15,820)

Optimizer and schedule are the paper's (Appendix C.2): 35 epochs, SGD lr 0.1 with Nesterov momentum
0.9 and weight decay 4e-4, batch 256, five epochs of linear warm-up, then the learning rate drops by
10x at 60% and 85% of training; cross-entropy with label smoothing 0.1. The donor split holds nine
signers out entirely (28 signers / 26,157 clips on AUTSL) and validates on the first 3,000 clips of
the held signers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from signcanon import (
    all_left,
    canonicalize,
    controlled_mixture,
    gloss_blocked,
    load_corpus,
    signer_blocked,
    signer_majority,
    to_model_input,
    unconditional_flip,
)
from signcanon.data import donor_split, subsample_balanced
from signcanon.model import build_ctrgcn
from signcanon.skeleton import MIRROR_PERM


def lr_scale(epoch: int, epochs: int) -> float:
    if epoch < 5:
        return (epoch + 1) / 5
    return 0.1 ** sum(epoch >= int(epochs * f) for f in (0.6, 0.85))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--ctrgcn-repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--held-out-signers", type=int, default=9)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    recipe = parser.add_mutually_exclusive_group()
    recipe.add_argument("--canonical", action="store_true")
    recipe.add_argument("--all-left", action="store_true")
    recipe.add_argument("--mixture", type=float, default=None, metavar="F")
    recipe.add_argument("--signer-blocked", type=float, default=None, metavar="F")
    recipe.add_argument("--gloss-blocked", type=float, default=None, metavar="F")
    recipe.add_argument("--signer-majority", action="store_true")
    recipe.add_argument("--unconditional-flip", action="store_true")
    parser.add_argument("--flip-aug", type=float, default=0.0, metavar="P")
    parser.add_argument("--construction-seed", type=int, default=0,
                        help="seed for the random choices inside --mixture / --signer-blocked / --gloss-blocked")
    parser.add_argument("--subsample", type=int, default=None)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    corpus = load_corpus(args.data)
    train_idx, held_idx, val_idx = donor_split(corpus.signers, args.held_out_signers, seed=42)
    clips, labels, signers = corpus.clips[train_idx], corpus.labels[train_idx], corpus.signers[train_idx]
    print(f"signer-holdout: train {len(labels)} clips ({len(set(signers.tolist()))} signers) | "
          f"val {len(val_idx)} clips ({args.held_out_signers} held signers)", flush=True)

    if args.subsample:
        if args.signer_blocked is not None or args.gloss_blocked is not None or args.signer_majority:
            raise SystemExit("--subsample drops the signer/gloss alignment the blocked recipes need")
        keep = subsample_balanced(labels, args.subsample, args.seed)
        clips, labels, signers = clips[keep], labels[keep], signers[keep]
        print(f"subsampled -> {len(labels)} clips", flush=True)

    if args.canonical:
        clips = canonicalize(clips)
    elif args.all_left:
        clips = all_left(clips)
    elif args.mixture is not None:
        clips = controlled_mixture(clips, args.mixture, args.construction_seed)
    elif args.signer_blocked is not None:
        clips = signer_blocked(clips, signers, args.signer_blocked, args.construction_seed)
    elif args.gloss_blocked is not None:
        clips = gloss_blocked(clips, labels, args.gloss_blocked, args.construction_seed)
    elif args.signer_majority:
        clips = signer_majority(clips, signers)
    elif args.unconditional_flip:
        clips = unconditional_flip(clips)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_ctrgcn(args.ctrgcn_repo, int(corpus.labels.max() + 1)).to(device)
    inputs = to_model_input(clips)
    targets = torch.tensor(labels)
    val_inputs = to_model_input(corpus.clips[val_idx])
    val_targets = corpus.labels[val_idx]
    optimizer = torch.optim.SGD(
        net.parameters(), 0.1, momentum=0.9, nesterov=True, weight_decay=4e-4, foreach=False
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    perm = torch.tensor(MIRROR_PERM)
    flip_gen = torch.Generator().manual_seed(31337 + args.seed)

    def mirror_batch(x):
        n, t, vc = x.shape
        x4 = x.view(n, t, 49, 3)[:, :, perm, :].clone()
        x4[..., 0] *= -1
        return x4.view(n, t, vc)

    n = len(labels)
    for epoch in range(args.epochs):
        for group in optimizer.param_groups:
            group["lr"] = 0.1 * lr_scale(epoch, args.epochs)
        net.train()
        order = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            batch = order[i : i + args.batch_size]
            x = inputs[batch].float()
            if args.flip_aug > 0:
                mask = torch.rand(len(batch), generator=flip_gen) < args.flip_aug
                if mask.any():
                    x[mask] = mirror_batch(x[mask])
            optimizer.zero_grad()
            loss = criterion(net(x.to(device)), targets[batch].long().to(device))
            loss.backward()
            optimizer.step()
        if epoch < 3 or epoch % 4 == 0 or epoch == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                preds = np.concatenate([
                    net(val_inputs[j : j + 512].float().to(device)).argmax(1).cpu().numpy()
                    for j in range(0, len(val_targets), 512)
                ])
            print(f"epoch {epoch + 1}/{args.epochs} loss {loss.item():.3f} "
                  f"held-signer val {100 * (preds == val_targets).mean():.1f}%", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
