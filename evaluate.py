#!/usr/bin/env python3
import os, sys, argparse, numpy as np

# ----------------- MOT parser -----------------
def parse_mot(path):
    frames = {}
    with open(path, "r") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split(",")
            if len(p) < 6: 
                continue
            fr, tid = int(float(p[0])), int(float(p[1]))
            x, y, w, h = map(float, p[2:6])
            frames.setdefault(fr, []).append((tid,x,y,w,h))
    return frames

# ----------------- IoU -----------------
def iou(a,b):
    x1,y1,w1,h1=a; x2,y2,w2,h2=b
    xa=max(x1,x2); ya=max(y1,y2)
    xb=min(x1+w1, x2+w2); yb=min(y1+h1, y2+h2)
    inter=max(0,xb-xa)*max(0,yb-ya)
    if inter<=0: return 0.0
    return inter/(w1*h1 + w2*h2 - inter)

# ----------------- Matcher -----------------
def match_frame(gts,dets,thr=0.5):
    if not gts or not dets:
        return [], set(range(len(gts))), set(range(len(dets)))
    M=np.zeros((len(gts), len(dets)))
    for i,(_,x,y,w,h) in enumerate(gts):
        for j,(_,X,Y,W,H) in enumerate(dets):
            M[i,j] = iou((x,y,w,h),(X,Y,W,H))
    matches=[]; used_g=set(); used_d=set()
    while True:
        i,j = np.unravel_index(np.argmax(M),M.shape)
        if M[i,j] < thr: break
        matches.append((i,j))
        used_g.add(i); used_d.add(j)
        M[i,:] = -1; M[:,j] = -1
    return matches, set(range(len(gts))) - used_g, set(range(len(dets))) - used_d

# ----------------- Evaluation -----------------
def evaluate(gt, pr, iou_thr=0.5):
    frames = sorted(set(gt)|set(pr))
    TP=FP=FN=IDSW=0; gt_total=0; det_total=0
    last = {}
    for f in frames:
        g = gt.get(f, []); d = pr.get(f, [])
        gt_total += len(g); det_total += len(d)
        matches, umg, umd = match_frame(g,d,iou_thr)
        TP += len(matches); FN += len(umg); FP += len(umd)
        for i,j in matches:
            gid = g[i][0]; pid = d[j][0]
            if gid in last and last[gid] != pid:
                IDSW += 1
            last[gid] = pid
    mota = 1 - (FN + FP + IDSW) / max(1, gt_total)
    idtp=TP; idfp=det_total-TP; idfn=gt_total-TP
    idf1 = (2*idtp) / max(1,(2*idtp + idfp + idfn))
    return {
        "MOTA": round(mota,4),
        "IDF1": round(idf1,4),
        "IDSW": IDSW,
        "FP": FP,
        "FN": FN,
        "GT": gt_total
    }

# ----------------- Main CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="Path to your tracker MOT file")
    ap.add_argument("--dataset", required=True, help="Dataset name (folder under datasets/)")
    ap.add_argument("--root", default="datasets", help="Datasets root folder")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    # find GT file automatically
    gt_dir = os.path.join(args.root, args.dataset, "ground_truth")
    if not os.path.isdir(gt_dir):
        sys.exit(f"[ERROR] Ground-truth folder not found: {gt_dir}")

    gt_files = [f for f in os.listdir(gt_dir) if f.endswith(".txt")]
    if len(gt_files) == 0:
        sys.exit(f"[ERROR] No GT .txt files in {gt_dir}")
    if len(gt_files) > 1:
        sys.exit(f"[ERROR] Multiple GT files found in {gt_dir}\nPlease keep one GT per dataset.")

    gt_file = gt_files[0]
    gt_path = os.path.join(gt_dir, gt_file)

    # parse + eval
    gt = parse_mot(gt_path)
    pr = parse_mot(args.file)
    res = evaluate(gt, pr, args.iou)

    print(f"\nDataset: {args.dataset}")
    print(f"GT file used: {gt_file}")
    print(f"Prediction file: {os.path.basename(args.file)}")
    for k,v in res.items():
        print(f"{k}: {v}")
    print()

if __name__ == "__main__":
    main()
