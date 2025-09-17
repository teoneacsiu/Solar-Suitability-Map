# scripts/evaluate_model.py
# --------------------------------------------------------------
# Evaluează harta de probabilități față de punctele etichetate:
#  - ROC-AUC, PR-AUC
#  - F1/precision/recall la prag optim (F1-max)
#  - Matrice de confuzie + curbe ROC/PR (PNG)
#  - metrics.json pentru HTML
# --------------------------------------------------------------

import argparse
from pathlib import Path
import json
import numpy as np
import rasterio
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, confusion_matrix, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser(description="Evaluate suitability map using labeled points.")
    p.add_argument("--map", required=True, help="GeoTIFF with probabilities (e.g., suitability_adjusted.tif)")
    p.add_argument("--points", required=True, help="NPZ with rows, cols, y (from make_labels.py)")
    p.add_argument("--outdir", default="data/metrics")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.points)
    rows, cols, y_true = data["rows"], data["cols"], data["y"]

    with rasterio.open(args.map) as src:
        arr = src.read(1).astype("float32")

    # extragem scorurile
    scores = arr[rows, cols]
    mask = np.isfinite(scores)
    scores = scores[mask]
    y_true = y_true[mask]

    if len(y_true) < 10:
        raise SystemExit("Too few valid points after masking.")

    # AUC-uri
    roc_auc = float(roc_auc_score(y_true, scores))
    pr_auc  = float(average_precision_score(y_true, scores))

    # prag optim pe F1
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1_vals = (2 * prec * rec) / np.maximum(prec + rec, 1e-9)
    best_idx = int(np.nanargmax(f1_vals))
    best_thr = float(thr[max(0, best_idx-1)]) if best_idx < len(thr) else 0.5

    y_pred = (scores >= best_thr).astype(np.uint8)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    f1  = float(f1_score(y_true, y_pred))
    prc = float(precision_score(y_true, y_pred))
    rcl = float(recall_score(y_true, y_pred))

    # salvăm grafice
    # ROC
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.figure(figsize=(4,3), dpi=150)
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--",lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.legend()
    plt.tight_layout(); plt.savefig(outdir / "roc.png"); plt.close()

    # PR
    plt.figure(figsize=(4,3), dpi=150)
    plt.plot(rec, prec, label=f"AP={pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
    plt.tight_layout(); plt.savefig(outdir / "pr.png"); plt.close()

    # Confusion matrix
    plt.figure(figsize=(3.2,3), dpi=150)
    plt.imshow(cm, cmap="Blues")
    for (i,j),v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0,1], ["Pred 0","Pred 1"]); plt.yticks([0,1], ["True 0","True 1"])
    plt.title(f"th={best_thr:.3f}  F1={f1:.3f}")
    plt.tight_layout(); plt.savefig(outdir / "cm.png"); plt.close()

    metrics = dict(
        n=len(y_true),
        roc_auc=roc_auc, pr_auc=pr_auc,
        best_threshold=best_thr,
        f1=f1, precision=prc, recall=rcl,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        plots=dict(roc="roc.png", pr="pr.png", cm="cm.png"),
    )
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✔ Saved metrics → {outdir/'metrics.json'}")

if __name__ == "__main__":
    main()
