"""Extract the Slovo (Russian SL) word-subset to OpenHands-PARITY 75-keypoint poses, trimmed to each
clip's actual sign frames [begin:end]. MediaPipe Holistic model_complexity=2 (matches the grid pipeline).
Reads data/slovo/subset120.csv -> data/slovo/poses/<attachment_id>.pkl {keypoints (T,75,3), confidences}.
Run: PYTHONPATH=src:scripts $HOME/.xsign-venv/bin/python -u -W ignore scripts/slovo_extract.py [every_n]
"""
from __future__ import annotations
import sys, os, pickle, pathlib
import numpy as np, pandas as pd, cv2, mediapipe as mp

ROOT = pathlib.Path("/Volumes/ExtremeSSD/sign-language-project/data/slovo")
OUT = ROOT / "poses"; OUT.mkdir(parents=True, exist_ok=True)
EVERY_N = int(sys.argv[1]) if len(sys.argv) > 1 else 2


def find_mp4(aid):
    for sub in ("train", "test"):
        p = ROOT / "raw" / sub / f"{aid}.mp4"
        if p.exists(): return p
    return None


def extract(path, begin, end, holistic):
    cap = cv2.VideoCapture(str(path)); kps, confs = [], []; i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if i < begin or (end and i > end) or (i - begin) % EVERY_N:
            i += 1; continue
        i += 1
        res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        kp = np.full((75, 3), np.nan, np.float32); cf = np.zeros(75, np.float32)
        if res.pose_landmarks:
            for j, lm in enumerate(res.pose_landmarks.landmark[:33]):
                kp[j] = (lm.x, lm.y, lm.z); cf[j] = lm.visibility
        for off, hand in ((33, res.left_hand_landmarks), (54, res.right_hand_landmarks)):
            if hand:
                for j, lm in enumerate(hand.landmark[:21]):
                    kp[off + j] = (lm.x, lm.y, lm.z); cf[off + j] = 1.0
        kps.append(kp); confs.append(cf)
    cap.release()
    return (np.stack(kps), np.stack(confs)) if kps else None


def main():
    df = pd.read_csv(ROOT / "subset120.csv")
    print(f"{len(df)} clips to extract (every_n={EVERY_N})", flush=True)
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=2)
    done = fail = 0
    for r in df.itertuples():
        op = OUT / f"{r.attachment_id}.pkl"
        if op.exists(): done += 1; continue
        p = find_mp4(r.attachment_id)
        if p is None: fail += 1; continue
        try:
            res = extract(p, int(r.begin), int(r.end), holistic)
            if res is None: fail += 1; continue
            kp, cf = res
            with open(op, "wb") as f:
                pickle.dump({"keypoints": kp, "confidences": cf}, f, protocol=4)
            done += 1
        except Exception as e:
            fail += 1; print(f"  FAIL {r.attachment_id}: {e}", flush=True)
        if done % 200 == 0: print(f"  {done}/{len(df)} done, {fail} failed", flush=True)
    holistic.close()
    print(f"DONE slovo extract: {done} ok, {fail} failed -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
