"""OpenHands (Zenodo 6674324) precomputed-pose loader for the rigorous cross-lingual transfer study.
Parity: ALL datasets use MediaPipe Holistic model_complexity=2, 75 keypoints (33 pose + 21 L-hand +
21 R-hand), 3D + confidences. We featurize to the SAME [2,T,49] representation as the toy encoder
(BODY 7 + hands 42, 2D, shoulder-normalised) so donor/target share one input space.

Signer metadata (for leakage-proof signer-disjoint splits):
  LSA64  : filename NNN_SSS_RRR -> class NNN, signer SSS   (10 signers)
  AUTSL  : filename signerXX_sampleYY_color -> signer XX; class from labels csv
  WLASL  : <video_id> -> gloss + signer_id via WLASL_v0.3.json
  GSL    : from split files
  INCLUDE: gloss from path; signer unreliable (flagged clip_only)
"""
from __future__ import annotations
import os, re, glob, pickle, json, pathlib
from collections import defaultdict
import numpy as np
from gdl_bench import _canon_frame, T_FIX
from explore_features import _impute, _resample

OH = pathlib.Path("/Volumes/ExtremeSSD/sign-language-project/data/openhands")
CACHE = pathlib.Path("/private/tmp/claude-501/-Volumes-ExtremeSSD-braille-project/3d5001ec-f602-4b29-914c-2829ebffa233/scratchpad/oh_cache")
CACHE.mkdir(parents=True, exist_ok=True)

# 75-kp layout: pose 0..32, L-hand 33..53, R-hand 54..74
BODY = [0, 11, 12, 13, 14, 15, 16]              # nose, L/R shoulder, L/R elbow, L/R wrist
LH = list(range(33, 54)); RH = list(range(54, 75))
IDX = BODY + LH + RH                             # 49


def featurize_oh(kp, conf=None, conf_thr=0.1):
    """kp (T,75,3) -> [2,32,49] float32. Low-confidence points -> NaN -> imputed."""
    xy = kp[:, IDX, :2].astype(np.float64)
    if conf is not None:
        bad = conf[:, IDX] < conf_thr
        xy[bad] = np.nan
    xy = _canon_frame(xy, rotate=False)         # center on shoulder mid, scale by shoulder width
    xy = _impute(xy.reshape(xy.shape[0], -1)).reshape(-1, 49, 2)
    xy = _resample(xy, T_FIX)
    return np.nan_to_num(xy).transpose(2, 0, 1).astype(np.float32)   # [2,T,49]


def _pkls(sub):
    return [p for p in glob.glob(str(OH / sub / "**/*.pkl"), recursive=True) if "/._" not in p]


def _feat_file(p):
    with open(p, "rb") as f:
        d = pickle.load(f)
    return featurize_oh(np.asarray(d["keypoints"]), np.asarray(d.get("confidences")))


# ---------------- per-dataset loaders ----------------
def load_lsa64():
    files = _pkls("LSA64_Cut")
    X, y, sg = [], [], []
    for p in files:
        m = re.match(r"(\d+)_(\d+)_(\d+)", os.path.basename(p)[:-4])
        if not m: continue
        X.append(_feat_file(p)); y.append(int(m.group(1)) - 1); sg.append(f"s{int(m.group(2))}")
    return np.stack(X), np.array(y), np.array(sg, dtype=object)


def load_include_oh():
    """class = immediate parent folder ('14. Bus'); no per-clip signer -> clip_only."""
    files = _pkls("INCLUDE")
    gl = sorted({pathlib.Path(p).parent.name for p in files}); gi = {g: i for i, g in enumerate(gl)}
    X, y, sg = [], [], []
    for p in files:
        X.append(_feat_file(p)); y.append(gi[pathlib.Path(p).parent.name]); sg.append("NA")
    return np.stack(X), np.array(y), np.array(sg, dtype=object)


def load_wlasl_oh(top_n=300):
    """video_id.pkl joined to WLASL_v0.3.json for gloss + signer_id. Subset to top_n most-frequent glosses."""
    raw = json.loads((pathlib.Path("/Volumes/ExtremeSSD/sign-language-project/data/WLASL/WLASL_v0.3.json")).read_text())
    vid2 = {}
    counts = defaultdict(int)
    for e in raw:
        for i in e["instances"]:
            vid2[str(i["video_id"])] = (e["gloss"].lower().strip(), f"s{i['signer_id']}"); counts[e["gloss"].lower().strip()] += 1
    keep = {g for g, _ in sorted(counts.items(), key=lambda x: -x[1])[:top_n]}
    avail = {os.path.basename(p)[:-4]: p for p in _pkls("WLASL")}
    gl = sorted(keep); gi = {g: i for i, g in enumerate(gl)}
    X, y, sg = [], [], []
    for vid, p in avail.items():
        if vid not in vid2: continue
        g, s = vid2[vid]
        if g not in keep: continue
        X.append(_feat_file(p)); y.append(gi[g]); sg.append(s)
    return np.stack(X), np.array(y), np.array(sg, dtype=object)


def load_autsl_oh():
    """filename 'signerXX_sampleYY_color' -> signer XX; class from local ChaLearn labels (100% join)."""
    import pandas as pd
    from xsign import paths
    lab = pd.concat([pd.read_csv(paths.AUTSL_META / f, header=None, names=["c", "l"])
                     for f in ["train_labels.csv", "validation_labels.csv", "test_labels.csv"]])
    lab = dict(zip(lab["c"], lab["l"]))
    X, y, sg = [], [], []
    for p in _pkls("AUTSL"):
        stem = os.path.basename(p)[:-4]; key = stem.replace("_color", "")
        m = re.match(r"signer(\d+)", stem)
        if key not in lab or not m: continue
        X.append(_feat_file(p)); y.append(int(lab[key])); sg.append(f"s{int(m.group(1))}")
    yy = np.array(y); remap = {c: i for i, c in enumerate(sorted(set(yy)))}
    return np.stack(X), np.array([remap[c] for c in yy]), np.array(sg, dtype=object)


def load_slovo():
    """Flagship: Russian SL word-subset. class=text, signer=user_id (194 signers → leak-proof)."""
    import pandas as pd
    df = pd.read_csv("/Volumes/ExtremeSSD/sign-language-project/data/slovo/subset120.csv")
    pd_ = pathlib.Path("/Volumes/ExtremeSSD/sign-language-project/data/slovo/poses")
    gl = sorted(df["text"].unique()); gi = {g: i for i, g in enumerate(gl)}
    X, y, sg = [], [], []
    for r in df.itertuples():
        p = pd_ / f"{r.attachment_id}.pkl"
        if not p.exists(): continue
        with open(p, "rb") as f: d = pickle.load(f)
        X.append(featurize_oh(np.asarray(d["keypoints"]), np.asarray(d.get("confidences"))))
        y.append(gi[r.text]); sg.append(str(r.user_id))
    return np.stack(X), np.array(y), np.array(sg, dtype=object)


def load_bdslw60():
    """Bangla flagship: pose stems 'U{user}W{word}F_t{trial}'. class=W-number, signer=U-number."""
    pd_ = pathlib.Path("/Volumes/ExtremeSSD/sign-language-project/data/bdslw60/poses")
    files = [p for p in glob.glob(str(pd_ / "*.pkl")) if "/._" not in p]
    X, y, sg = [], [], []
    words = sorted({re.match(r"U\d+(W\d+)", os.path.basename(p)).group(1) for p in files},
                   key=lambda w: int(w[1:]))
    gi = {w: i for i, w in enumerate(words)}
    for p in files:
        m = re.match(r"(U\d+)(W\d+)", os.path.basename(p))
        if not m: continue
        with open(p, "rb") as f: d = pickle.load(f)
        X.append(featurize_oh(np.asarray(d["keypoints"]), np.asarray(d.get("confidences"))))
        y.append(gi[m.group(2)]); sg.append(m.group(1))
    return np.stack(X), np.array(y), np.array(sg, dtype=object)



def load_ssl400(min_clips=8):
    """Sinhala SL (SSL400, Kaggle CC-BY-NC-SA). Stems '<Gloss>__NNN'. class=gloss folder name.
    NO signer IDs exist (documented limitation) -> sg='NA', splits are clip-level (leak=clip_only,
    disclosed). Restrict to glosses with >= min_clips clips (k-shot needs support+test)."""
    pd_ = pathlib.Path("/Volumes/ExtremeSSD/sign-language-project/data/ssl400/poses")
    files = [q for q in glob.glob(str(pd_ / "*.pkl")) if "/._" not in q]
    from collections import Counter
    by = Counter(os.path.basename(q).split("__")[0] for q in files)
    keep = sorted(g for g, n in by.items() if n >= min_clips)
    gi = {g: i for i, g in enumerate(keep)}
    X, y, sg = [], [], []
    for q in files:
        g = os.path.basename(q).split("__")[0]
        if g not in gi: continue
        with open(q, "rb") as f: d = pickle.load(f)
        X.append(featurize_oh(np.asarray(d["keypoints"]), np.asarray(d.get("confidences"))))
        y.append(gi[g]); sg.append("NA")
    return np.stack(X), np.array(y), np.array(sg)

LOADERS = {"lsa64": load_lsa64, "include_oh": load_include_oh,
           "wlasl_oh": load_wlasl_oh, "autsl_oh": load_autsl_oh, "slovo": load_slovo,
           "bdslw60": load_bdslw60, "ssl400": load_ssl400}
HAS_SIGNER = {"lsa64": True, "autsl_oh": True, "wlasl_oh": True, "include_oh": False, "slovo": True,
              "bdslw60": True, "ssl400": False}


def load(dataset):
    cf = CACHE / f"{dataset}.npz"
    if cf.exists():
        d = np.load(cf, allow_pickle=True); return d["X"], d["y"], d["sg"]
    X, y, sg = LOADERS[dataset]()
    np.savez_compressed(cf, X=X.astype(np.float32), y=y, sg=np.array(sg, dtype=object))
    return X.astype(np.float32), y, np.array(sg, dtype=object)


if __name__ == "__main__":
    import sys
    for ds in (sys.argv[1:] or ["lsa64"]):
        X, y, sg = load(ds)
        u = sorted(set(sg.tolist()))
        print(f"{ds:10s} X{X.shape} classes={int(y.max()+1)} signers={len(u)} "
              f"clips/signer≈{len(y)//max(len(u),1)}")
