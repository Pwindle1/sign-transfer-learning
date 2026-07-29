from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from signcanon import canonicalize, controlled_mixture, load_corpus, to_model_input
from signcanon.data import signer_disjoint_split
from signcanon.model import build_ctrgcn


def subsample(clips, labels, n, seed=0):
    rng = np.random.default_rng(seed)
    by_class = {}
    for i, c in enumerate(labels):
        by_class.setdefault(int(c), []).append(i)
    quota = max(1, n // len(by_class))
    chosen = np.concatenate(
        [rng.choice(idx, min(quota, len(idx)), replace=False) for idx in by_class.values()]
    )
    return clips[chosen], labels[chosen]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--ctrgcn-repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--held-out-signers", type=int, default=9)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--mixture", type=float, default=None)
    parser.add_argument("--subsample", type=int, default=None)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    corpus = load_corpus(args.data)
    train_idx, *_ = signer_disjoint_split(
        corpus.labels, corpus.signers, args.held_out_signers, 1, seed=42
    )
    clips, labels = corpus.clips[train_idx], corpus.labels[train_idx]
    if args.subsample:
        clips, labels = subsample(clips, labels, args.subsample, args.seed)
    if args.mixture is not None:
        clips = controlled_mixture(clips, args.mixture, args.seed)
    elif args.canonical:
        clips = canonicalize(clips)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_ctrgcn(args.ctrgcn_repo, int(labels.max() + 1)).to(device)
    inputs = to_model_input(clips)
    targets = torch.tensor(labels)
    optimizer = torch.optim.SGD(net.parameters(), 0.1, momentum=0.9, weight_decay=5e-4, foreach=False)
    criterion = torch.nn.CrossEntropyLoss()

    def lr_scale(epoch):
        return 0.1 ** sum(epoch >= int(args.epochs * f) for f in (0.6, 0.85))

    n = len(labels)
    for epoch in range(args.epochs):
        for group in optimizer.param_groups:
            group["lr"] = 0.1 * lr_scale(epoch)
        net.train()
        order = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            batch = order[i : i + args.batch_size]
            optimizer.zero_grad()
            loss = criterion(net(inputs[batch].float().to(device)), targets[batch].long().to(device))
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch + 1}/{args.epochs}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
