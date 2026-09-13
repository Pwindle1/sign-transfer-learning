"""k-shot evaluation of one donor on one or more recipient corpora, or of the no-donor baseline.

Each cell is (recipient, k, episode seed). Support and test draws depend only on those three values,
so every donor - and the no-donor baseline - is scored on exactly the same clips. The paper's
headline grid is --eval-seeds 0 1 2 3 4 (90 cells over six recipients); the mechanism variants use
the default 0 1 2 (54 cells).

--canonical applies the paper's operator to the recipient clips (the canonical donors' native frame);
omit it for standard donors, flip-augmentation donors, the unconditional-flip donors and the
no-donor baseline, which all see raw recipients.

--from-scratch trains from random initialisation on the support set alone through the same fine-tune
code path (empty donor state dict), one trajectory per cell scored at each --budgets epoch count;
the reported accuracy is the most favourable budget (paper Appendix A.5).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from signcanon import canonicalize, episodes, load_corpus, to_model_input
from signcanon.adapt import classify_accuracy, embed, full_finetune, prototype_accuracy
from signcanon.model import build_ctrgcn


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--donor", default=None, help="donor checkpoint (omit with --from-scratch)")
    parser.add_argument("--donor-id", type=int, default=None, help="seed of the donor (-1 for --from-scratch)")
    parser.add_argument("--ctrgcn-repo", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--canonical", action="store_true", help="canonicalize the recipient clips")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", default=["full_ft", "proto"])
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--budgets", type=int, nargs="+", default=[30, 60])
    args = parser.parse_args()

    import torch

    if args.from_scratch:
        donor_state, donor_id, methods = {}, -1 if args.donor_id is None else args.donor_id, ["full_ft"]
    else:
        if args.donor is None or args.donor_id is None:
            raise SystemExit("--donor and --donor-id are required unless --from-scratch")
        donor_state, donor_id, methods = torch.load(args.donor, map_location="cpu"), args.donor_id, args.methods
    build = lambda ncls: build_ctrgcn(args.ctrgcn_repo, ncls)

    encoder = None
    if "proto" in methods:
        encoder = build(226)
        encoder.load_state_dict(donor_state)

    rows = []
    for target_index, target_path in enumerate(args.targets):
        corpus = load_corpus(target_path)
        clips = canonicalize(corpus.clips) if args.canonical else corpus.clips
        inputs = to_model_input(clips)
        embeddings = embed(encoder, inputs) if encoder is not None else None
        for episode in episodes(corpus, tuple(args.ks), tuple(args.eval_seeds)):
            remap = {c: i for i, c in enumerate(episode.classes)}
            support_labels = np.array([remap[int(corpus.labels[i])] for i in episode.support])
            test_labels = np.array([remap[int(corpus.labels[i])] for i in episode.test])
            base = dict(
                donor=donor_id,
                target=corpus.name,
                k=episode.k,
                eval_seed=episode.eval_seed,
                n_classes=len(episode.classes),
                recipient_canonical=bool(args.canonical),
            )
            if "proto" in methods:
                accuracy = prototype_accuracy(
                    embeddings[episode.support], support_labels, embeddings[episode.test], test_labels
                )
                rows.append(base | {"method": "proto", "accuracy": 100 * accuracy})
            if "full_ft" in methods and not args.from_scratch:
                net = full_finetune(
                    donor_state, len(episode.classes), inputs[episode.support], support_labels, build
                )
                accuracy = classify_accuracy(net, inputs[episode.test], test_labels)
                rows.append(base | {"method": "full_ft", "accuracy": 100 * accuracy})
            if args.from_scratch:
                budgets = sorted(set(args.budgets))
                # the random initialisation varies with the cell, not just the episode seed
                torch.manual_seed(4200 + 1000 * target_index + 10 * episode.k + episode.eval_seed)
                test_inputs = inputs[episode.test]
                _, by_epoch = full_finetune(
                    {}, len(episode.classes), inputs[episode.support], support_labels, build,
                    epochs=max(budgets), eval_at=tuple(budgets),
                    eval_fn=lambda net: classify_accuracy(net, test_inputs, test_labels),
                )
                rows.append(base | {
                    "method": "full_ft", "recipe": "no_donor",
                    "accuracy": 100 * max(by_epoch.values()),
                    "accuracy_by_epochs": {str(e): 100 * a for e, a in sorted(by_epoch.items())},
                    "epoch_ceiling": max(budgets),
                })
            Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"{corpus.name}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
