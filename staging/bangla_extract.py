"""BdSLW60 (Bangla SL) trial segmentation + OpenHands-parity pose extraction.
Each mp4 = all repetitions of one word by one user; annotationW*.txt gives per-trial frame windows:
    User: U1 ... File: U1W1F / Word: W1 / trials: N / then lines "a/b c/d" = trial [a..d] (outer bounds).
We cap trials per (user,word) at TRIAL_CAP (k-shot eval needs breadth, not every repetition), extract
each trial with MediaPipe Holistic model_complexity=2 -> {keypoints (T,75,3), confidences (T,75)}.
Sharded: WORKER_ID / N_WORKERS split the manifest for parallel extraction.

Usage:
  manifest : python bangla_extract.py manifest        -> data/bdslw60/manifest.csv
  extract  : WORKER_ID=0 N_WORKERS=5 python bangla_extract.py extract [every_n]
"""
from __future__ import annotations
import sys, os, re, glob, pickle, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path("/Volumes/ExtremeSSD/sign-language-project/data/bdslw60")
OUT = ROOT / "poses"; OUT.mkdir(parents=True, exist_ok=True)
TRIAL_CAP = int(os.environ.get("BD_TRIAL_CAP", "6"))


def parse_annotations():
    rows = []
    for ap in glob.glob(str(ROOT / "raw" / "*" / "annotation*.txt")):
        if "/._" in ap: continue
        folder = pathlib.Path(ap).parent
        cur_user = cur_file = cur_word = None; want = 0
        for line in open(ap, errors="ignore"):
            line = line.strip()
            m = re.search(r"User:\s*(U\d+).*?File:\s*(\S+)", line)
            if m: cur_user, cur_file = m.group(1), m.group(2); continue
            m = re.match(r"Word:\s*(W\d+)", line)
            if m: cur_word = m.group(1); continue
            m = re.match(r"trials:\s*(\d+)", line)
            if m: want = int(m.group(1)); continue
            m = re.match(r"(\d+)/(\d+)\s+(\d+)/(\d+)", line)
            if m and cur_file and cur_word:
                a, b, c, d = map(int, m.groups())
                begin, end = min(a, b), max(c, d)
                if end > begin:
                    rows.append(dict(file=cur_file, user=cur_user, word=cur_word,
                                     folder=folder.name, begin=begin, end=end))
    df = pd.DataFrame(rows)
    df["trial"] = df.groupby(["file"]).cumcount()
    df = df[df["trial"] < TRIAL_CAP].reset_index(drop=True)
    # resolve video path
    def vp(r):
        p = ROOT / "raw" / r["folder"] / f"{r['file']}.mp4"
        return str(p) if p.exists() else None
    df["path"] = df.apply(vp, axis=1)
    n_miss = df["path"].isna().sum()
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    df.to_csv(ROOT / "manifest.csv", index=False)
    print(f"manifest: {len(df)} trials | users {df['user'].nunique()} | words {df['word'].nunique()} "
          f"| cap {TRIAL_CAP}/user-word | missing-video rows dropped: {n_miss}")


def extract_worker(every_n=2):
    import cv2, mediapipe as mp
    wid = int(os.environ.get("WORKER_ID", "0")); nw = int(os.environ.get("N_WORKERS", "1"))
    df = pd.read_csv(ROOT / "manifest.csv")
    df = df[df.index % nw == wid]
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=2)
    done = fail = 0
    # group by video: read each video once, slice all its trials
    for path, g in df.groupby("path"):
        need = [(r.file, r.trial, int(r.begin), int(r.end)) for r in g.itertuples()
                if not (OUT / f"{r.file}_t{r.trial}.pkl").exists()]
        if not need: done += len(g); continue
        cap = cv2.VideoCapture(path)
        frames = {}
        wanted = sorted({i for _, _, b, e in need for i in range(b, e + 1, every_n)})
        wset = set(wanted); idx = 0; maxf = max(wanted) if wanted else 0
        while idx <= maxf:
            ok, frame = cap.read()
            if not ok: break
            if idx in wset:
                res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                kp = np.full((75, 3), np.nan, np.float32); cf = np.zeros(75, np.float32)
                if res.pose_landmarks:
                    for j, lm in enumerate(res.pose_landmarks.landmark[:33]):
                        kp[j] = (lm.x, lm.y, lm.z); cf[j] = lm.visibility
                for off, hand in ((33, res.left_hand_landmarks), (54, res.right_hand_landmarks)):
                    if hand:
                        for j, lm in enumerate(hand.landmark[:21]):
                            kp[off + j] = (lm.x, lm.y, lm.z); cf[off + j] = 1.0
                frames[idx] = (kp, cf)
            idx += 1
        cap.release()
        for fname, trial, b, e in need:
            ks = [frames[i] for i in range(b, e + 1, every_n) if i in frames]
            if len(ks) < 3: fail += 1; continue
            kp = np.stack([k for k, _ in ks]); cf = np.stack([c for _, c in ks])
            with open(OUT / f"{fname}_t{trial}.pkl", "wb") as f:
                pickle.dump({"keypoints": kp, "confidences": cf}, f, protocol=4)
            done += 1
        if done % 100 < len(need): print(f"[w{wid}] {done} done, {fail} failed", flush=True)
    holistic.close()
    print(f"[w{wid}] DONE: {done} ok, {fail} failed", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "manifest": parse_annotations()
    else: extract_worker(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
