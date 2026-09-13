"""SSL400 (Sinhala SL) OpenHands-parity pose extraction.
Shipped CSVs are pose-only (33 kp, no hands) -> unusable for the 49-node manual+body space, so we
re-extract from raw video with the pinned pipeline (MediaPipe Holistic model_complexity=2 -> 75 kp),
keeping cross-dataset parity. One clip per file: Dataset - Original/<Category>/<Gloss>/<Gloss>_NNN.mov
-> data/ssl400/poses/<Gloss>__NNN.pkl {keypoints (T,75,3), confidences (T,75)}.
No signer IDs exist (documented limitation; splits will be clip-level with disclosed caveat).

Usage: WORKER_ID=0 N_WORKERS=6 python ssl400_extract.py [every_n=1]
"""
from __future__ import annotations
import sys, os, glob, pickle, pathlib
import numpy as np

ROOT = pathlib.Path("/Volumes/ExtremeSSD/sign-language-project/data/ssl400")
RAW = ROOT / "Dataset - Original"
OUT = ROOT / "poses"; OUT.mkdir(parents=True, exist_ok=True)


def clips():
    fs = sorted(glob.glob(str(RAW / "*" / "*" / "*.mov")) + glob.glob(str(RAW / "*" / "*" / "*.mp4")))
    return [f for f in fs if "/._" not in f]


def extract(every_n=1):
    import cv2, mediapipe as mp
    wid = int(os.environ.get("WORKER_ID", "0")); nw = int(os.environ.get("N_WORKERS", "1"))
    fs = clips(); fs = [f for i, f in enumerate(fs) if i % nw == wid]
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=2)
    done = fail = skip = 0
    for path in fs:
        p = pathlib.Path(path)
        gloss = p.parent.name.replace(" ", "_")
        stem = p.stem.replace(" ", "_")                      # Gloss_NNN
        of = OUT / f"{gloss}__{stem.split('_')[-1]}.pkl"
        if of.exists(): skip += 1; continue
        cap = cv2.VideoCapture(path)
        ks = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if idx % every_n == 0:
                res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                kp = np.full((75, 3), np.nan, np.float32); cf = np.zeros(75, np.float32)
                if res.pose_landmarks:
                    for j, lm in enumerate(res.pose_landmarks.landmark[:33]):
                        kp[j] = (lm.x, lm.y, lm.z); cf[j] = lm.visibility
                for off, hand in ((33, res.left_hand_landmarks), (54, res.right_hand_landmarks)):
                    if hand:
                        for j, lm in enumerate(hand.landmark[:21]):
                            kp[off + j] = (lm.x, lm.y, lm.z); cf[off + j] = 1.0
                ks.append((kp, cf))
            idx += 1
        cap.release()
        if len(ks) < 3: fail += 1; continue
        kp = np.stack([k for k, _ in ks]); cf = np.stack([c for _, c in ks])
        with open(of, "wb") as f:
            pickle.dump({"keypoints": kp, "confidences": cf}, f, protocol=4)
        done += 1
        if done % 100 == 0: print(f"[w{wid}] {done} done {fail} failed {skip} skipped", flush=True)
    holistic.close()
    print(f"[w{wid}] DONE: {done} ok, {fail} failed, {skip} skipped", flush=True)


if __name__ == "__main__":
    extract(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
