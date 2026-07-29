from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from signcanon import canonicalize, episodes, load_corpus, to_model_input
from signcanon.adapt import classify_accuracy, embed, full_finetune, prototype_accuracy
from signcanon.model import build_ctrgcn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", required=True)
    parser.add_argument("--donor-id", type=int, required=True)
    parser.add_argument("--ctrgcn-repo", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", default=["full_ft", "proto"])
    args = parser.parse_args()

    import torch

    donor_state = torch.load(args.donor, map_location="cpu")
    build = lambda ncls: build_ctrgcn(args.ctrgcn_repo, ncls)
    encoder = build(226)
    encoder.load_state_dict(donor_state)

    rows = []
    for target_path in args.targets:
        corpus = load_corpus(target_path)
        clips = canonicalize(corpus.clips) if args.canonical else corpus.clips
        inputs = to_model_input(clips)
        embeddings = embed(encoder, inputs) if "proto" in args.methods else None
        for episode in episodes(corpus, tuple(args.ks), tuple(args.eval_seeds)):
            remap = {c: i for i, c in enumerate(episode.classes)}
            support_labels = np.array([remap[int(corpus.labels[i])] for i in episode.support])
            test_labels = np.array([remap[int(corpus.labels[i])] for i in episode.test])
            base = dict(
                donor=args.donor_id,
                target=corpus.name,
                k=episode.k,
                eval_seed=episode.eval_seed,
                n_classes=len(episode.classes),
            )
            if "proto" in args.methods:
                accuracy = prototype_accuracy(
                    embeddings[episode.support], support_labels, embeddings[episode.test], test_labels
                )
                rows.append(base | {"method": "proto", "accuracy": 100 * accuracy})
            if "full_ft" in args.methods:
                net = full_finetune(
                    donor_state, len(episode.classes), inputs[episode.support], support_labels, build
                )
                accuracy = classify_accuracy(net, inputs[episode.test], test_labels)
                rows.append(base | {"method": "full_ft", "accuracy": 100 * accuracy})
            Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"{corpus.name}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
