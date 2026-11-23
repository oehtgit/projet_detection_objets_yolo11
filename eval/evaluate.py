"""
Évaluation finale sur le jeu de TEST et comparaison avant / après fine-tuning.

Pour chaque modèle passé en argument :
  1. précision : mAP@0.5, mAP@0.5:0.95, précision, rappel et AP par classe sur le test
     (model.val, mêmes réglages pour tous les modèles : imgsz, conf=0.001, iou=0.7) ;
  2. vitesse : temps par image en batch 1 (le cas du temps réel), après un warmup,
     sur les images du test préchargées en mémoire (le disque n'est pas mesuré).

Deux types de modèles sont acceptés :
  - le modèle COCO d'origine (80 classes, yolo11n.pt) : les labels VOC sont renumérotés
    avec les numéros COCO (même méthode que data/baseline_zeroshot.py), les prédictions
    des 60 classes COCO absentes de VOC sont ignorées ;
  - un modèle fine-tuné (20 classes, train/weights/best.pt) : évalué directement avec
    data/data.yaml.

Sorties (eval/results/) :
  - metrics_test.csv      une ligne par modèle (format commun du projet)
  - ap_par_classe_test.csv  AP@0.5 et AP@0.5:0.95 par classe et par modèle
  - metrics_test.json     tout, plus la configuration de la mesure (GPU, versions)
  - figures/ : comparaison globale, AP par classe avant/après, compromis vitesse/précision

Exemples (Google Colab, GPU) :
    python eval/evaluate.py --device 0
    python eval/evaluate.py --device 0 --models avant=yolo11n.pt apres=train/weights/best.pt
    # compromis vitesse / précision : même modèle, images plus petites à l'inférence
    python eval/evaluate.py --device 0 --models avant=yolo11n.pt apres=train/weights/best.pt \
        apres_416=train/weights/best.pt@416 apres_320=train/weights/best.pt@320
    python eval/evaluate.py --device 0 --models avant=yolo11n.pt \
        lr_faible=train/runs/lr_faible/weights/best.pt imgsz_416=train/runs/imgsz_416/weights/best.pt
"""

import argparse
import csv
import json
import platform
import random
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # pas d'affichage nécessaire (Colab / terminal)
import matplotlib.pyplot as plt
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
from baseline_zeroshot import build_mapping, remap_split  # noqa: E402  (code de la Personne 1, réutilisé)

EVAL_DIR = ROOT / "eval"
OUT_DIR = EVAL_DIR / "results"
FIG_DIR = OUT_DIR / "figures"

# Couleurs fixes : même palette que train/compare_runs.py.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


# ----------------------------------------------------------------------------------------
# Précision
# ----------------------------------------------------------------------------------------
def is_coco_model(model, voc_names):
    """Vrai si le modèle a les 80 classes COCO, faux s'il a exactement nos 20 classes."""
    names = {int(k): v for k, v in model.names.items()}
    if names == {int(k): v for k, v in voc_names.items()}:
        return False
    if len(names) == 80:
        return True
    raise ValueError(
        f"Classes du modèle inattendues ({len(names)} classes) : ni COCO (80), "
        f"ni identiques à data.yaml (20, même ordre)."
    )


def evaluate_accuracy(name, model, data, data_yaml, split, args):
    """Lance model.val() sur le split choisi et renvoie (métriques globales, AP par classe)."""
    voc_names = data["names"]
    common = dict(imgsz=args.imgsz, batch=args.batch, device=args.device, conf=0.001, iou=0.7,
                  plots=True, project=str(OUT_DIR / "val"), name=name, exist_ok=True, verbose=False)

    if is_coco_model(model, voc_names):
        mapping = build_mapping(voc_names, model.names)           # numéro VOC -> numéro COCO
        out_root = Path(data["path"]).parent / (Path(data["path"]).name + "_coco_ids")
        yaml_path = remap_split(data, mapping, model.names, out_root, split)
        metrics = model.val(data=str(yaml_path), split="val", **common)
        index_to_voc = {coco: voc_names[voc] for voc, coco in mapping.items()}
        setting = "COCO 80 classes, labels renumérotés"
    else:
        metrics = model.val(data=str(data_yaml), split=split, **common)
        index_to_voc = {int(k): v for k, v in voc_names.items()}
        setting = "20 classes (fine-tuné)"

    per_class = {}
    for j, c in enumerate(metrics.box.ap_class_index):
        per_class[index_to_voc[int(c)]] = {"AP50": float(metrics.box.ap50[j]),
                                           "AP50-95": float(metrics.box.ap[j])}
    glob = {
        "setting": setting,
        "precision": float(metrics.box.mp),
        "rappel": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }
    return glob, per_class


# ----------------------------------------------------------------------------------------
# Vitesse
# ----------------------------------------------------------------------------------------
def load_speed_images(data, split, n, seed=0):
    """Charge n images du split en mémoire (tirage fixe : mêmes images pour tous les modèles)."""
    img_dir = Path(data[split] if data.get(split) else data["val"])
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    random.Random(seed).shuffle(files)
    return [cv2.imread(str(p)) for p in files[:n]]


def sync(device):
    if torch.cuda.is_available() and str(device) not in ("cpu", "None"):
        torch.cuda.synchronize()


def measure_speed(model, images, args):
    """Temps par image en batch 1, comme dans la démo temps réel.

    - warmup : les premières inférences (allocation mémoire GPU, compilation des noyaux)
      sont plus lentes et ne sont pas comptées ;
    - inference_ms : temps du réseau seul (mesuré par Ultralytics) ;
    - total_ms : temps complet d'une image (prétraitement + réseau + NMS), mesuré à
      l'horloge ; c'est lui qui donne les FPS réellement obtenus.
    """
    kw = dict(imgsz=args.imgsz, device=args.device, conf=0.25, verbose=False)
    for img in images[: args.warmup]:
        model.predict(img, **kw)

    pre, inf, post, total = [], [], [], []
    for img in images:
        sync(args.device)
        t0 = time.perf_counter()
        r = model.predict(img, **kw)[0]
        sync(args.device)
        total.append((time.perf_counter() - t0) * 1000)
        pre.append(r.speed["preprocess"])
        inf.append(r.speed["inference"])
        post.append(r.speed["postprocess"])

    total_ms = float(np.mean(total))
    return {
        "pretraitement_ms": round(float(np.mean(pre)), 2),
        "inference_ms": round(float(np.mean(inf)), 2),
        "postprocess_ms": round(float(np.mean(post)), 2),
        "total_ms": round(total_ms, 2),
        "total_ms_p95": round(float(np.percentile(total, 95)), 2),
        "FPS": round(1000 / total_ms, 1),
    }


# ----------------------------------------------------------------------------------------
# Sorties
# ----------------------------------------------------------------------------------------
def write_csvs(rows, per_class_all, class_names):
    fields = ["modele", "poids", "setting", "imgsz", "precision", "rappel", "mAP50", "mAP50-95",
              "pretraitement_ms", "inference_ms", "postprocess_ms", "total_ms", "total_ms_p95", "FPS"]
    with open(OUT_DIR / "metrics_test.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) and k in
                            ("precision", "rappel", "mAP50", "mAP50-95") else v)
                        for k, v in r.items() if k in fields})

    models = list(per_class_all)
    with open(OUT_DIR / "ap_par_classe_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["classe"] + [f"{m}_{k}" for m in models for k in ("AP50", "AP50-95")])
        for c in class_names:
            row = [c]
            for m in models:
                pc = per_class_all[m].get(c)
                row += [round(pc["AP50"], 4), round(pc["AP50-95"], 4)] if pc else ["", ""]
            w.writerow(row)


def plot_global(rows):
    """Barres groupées : P, R, mAP@0.5, mAP@0.5:0.95 pour chaque modèle."""
    keys = ["precision", "rappel", "mAP50", "mAP50-95"]
    labels = ["Précision", "Rappel", "mAP@0.5", "mAP@0.5:0.95"]
    x = np.arange(len(keys))
    width = 0.8 / len(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (r, color) in enumerate(zip(rows, COLORS)):
        vals = [r[k] for k in keys]
        bars = ax.bar(x + (i - (len(rows) - 1) / 2) * width, vals, width, color=color, label=r["modele"])
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, labels)
    low = min(r[k] for r in rows for k in keys)
    ax.set_ylim(max(0, round(low - 0.15, 1)), 1)  # axe resserré : les écarts de quelques points restent visibles
    ax.set_ylabel("Score sur le test")
    ax.set_title("Avant / après fine-tuning (jeu de test)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", ncol=len(rows), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparaison_globale.png", dpi=150)
    plt.close(fig)


def plot_per_class(per_class_all, ref, new):
    """Écart d'AP@0.5:0.95 par classe entre deux modèles, trié du plus grand gain à la plus grande perte."""
    a, b = per_class_all[ref], per_class_all[new]
    classes = [c for c in a if c in b]
    delta = sorted(((b[c]["AP50-95"] - a[c]["AP50-95"]) * 100, c) for c in classes)
    vals, names = zip(*delta)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, vals, color=[COLORS[2] if v >= 0 else COLORS[7] for v in vals])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Écart d'AP@0.5:0.95 (points), {new} − {ref}")
    ax.set_title("Effet du fine-tuning par classe (jeu de test)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"ap_par_classe_{ref}_vs_{new}.png", dpi=150)
    plt.close(fig)


def plot_speed_accuracy(rows, device_name):
    """Compromis vitesse / précision : FPS en abscisse, mAP@0.5:0.95 en ordonnée."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r, color in zip(rows, COLORS):
        ax.scatter(r["FPS"], r["mAP50-95"], s=80, color=color, zorder=3)
        ax.annotate(f"{r['modele']} ({r['imgsz']} px)", (r["FPS"], r["mAP50-95"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.axvline(30, color="grey", linestyle="--", linewidth=1)
    ax.text(30, ax.get_ylim()[0], " 30 FPS", color="grey", va="bottom", fontsize=8)
    ax.set_xlabel(f"FPS (batch 1, {device_name})")
    ax.set_ylabel("mAP@0.5:0.95 (test)")
    ax.set_title("Compromis vitesse / précision")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vitesse_precision.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------
def parse_models(items):
    """nom=poids ou nom=poids@taille (ex. apres_416=train/weights/best.pt@416)."""
    models = {}
    for item in items:
        name, _, path = item.partition("=")
        if not path:
            name, path = Path(item).stem, item
        path, _, size = path.partition("@")
        models[name] = (path, int(size) if size else None)
    return models


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["avant=yolo11n.pt", "apres=train/weights/best.pt"],
                   help="liste nom=poids ou nom=poids@taille ; le premier sert de référence")
    p.add_argument("--data", default=str(ROOT / "data" / "data.yaml"))
    p.add_argument("--split", default="test")
    p.add_argument("--imgsz", type=int, default=640,
                   help="taille d'image commune ; utiliser --imgsz-auto pour celle de chaque entraînement")
    p.add_argument("--imgsz-auto", action="store_true",
                   help="chaque modèle fine-tuné est évalué à la taille de son entraînement (ex. imgsz_416)")
    p.add_argument("--batch", type=int, default=32, help="batch de model.val (précision seulement)")
    p.add_argument("--device", default=None, help="0 pour le GPU, cpu pour le CPU")
    p.add_argument("--speed-images", type=int, default=300, help="nombre d'images pour la mesure de vitesse")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--no-speed", action="store_true", help="ne mesure que la précision")
    p.add_argument("--replot", action="store_true",
                   help="refait seulement les figures à partir de results/metrics_test.json")
    args = p.parse_args()

    if args.replot:
        info = json.loads((OUT_DIR / "metrics_test.json").read_text())
        rows, per_class_all = info["modeles"], info["par_classe"]
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plot_global(rows)
        for r in rows[1:]:
            plot_per_class(per_class_all, rows[0]["modele"], r["modele"])
        if "FPS" in rows[0]:
            plot_speed_accuracy(rows, info["materiel"])
        print(f"Figures refaites dans {FIG_DIR}")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = check_det_dataset(args.data)
    class_names = [data["names"][i] for i in sorted(data["names"])]
    models = parse_models(args.models)
    for name, (path, _) in models.items():
        if not path.endswith(".pt") or (not Path(path).exists() and Path(path).name != path):
            raise SystemExit(f"Poids introuvables pour '{name}' : {path}")

    device_name = (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() and str(args.device) != "cpu"
                   else f"CPU, {torch.get_num_threads()} threads")
    speed_images = [] if args.no_speed else load_speed_images(data, args.split, args.speed_images)

    base_imgsz = args.imgsz
    rows, per_class_all = [], {}
    for name, (path, size) in models.items():
        model = YOLO(path)
        args.imgsz = size or base_imgsz
        if size is None and args.imgsz_auto and getattr(model, "ckpt", None) and model.ckpt.get("train_args"):
            args.imgsz = int(model.ckpt["train_args"].get("imgsz", base_imgsz))
        print(f"\n=== {name} ({path}) : {len(model.names)} classes, imgsz {args.imgsz} ===")

        glob, per_class = evaluate_accuracy(name, model, data, args.data, args.split, args)
        row = {"modele": name, "poids": path, "imgsz": args.imgsz, **glob}
        if not args.no_speed:
            row.update(measure_speed(model, speed_images, args))
        rows.append(row)
        per_class_all[name] = per_class
        print(f"mAP50 = {glob['mAP50']:.3f} | mAP50-95 = {glob['mAP50-95']:.3f} | "
              f"P = {glob['precision']:.3f} | R = {glob['rappel']:.3f}"
              + ("" if args.no_speed else f" | {row['total_ms']:.1f} ms/image = {row['FPS']} FPS"))

    write_csvs(rows, per_class_all, class_names)
    plot_global(rows)
    ref = rows[0]["modele"]
    for r in rows[1:]:
        plot_per_class(per_class_all, ref, r["modele"])
    if not args.no_speed:
        plot_speed_accuracy(rows, device_name)

    info = {
        "split": args.split, "conf_val": 0.001, "iou_val": 0.7,
        "vitesse": None if args.no_speed else {
            "batch": 1, "images": len(speed_images), "warmup": args.warmup, "conf": 0.25},
        "materiel": device_name, "ultralytics": ultralytics.__version__,
        "torch": torch.__version__, "python": platform.python_version(),
        "modeles": rows, "par_classe": per_class_all,
    }
    (OUT_DIR / "metrics_test.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))

    print(f"\n{'modèle':<16}{'mAP50':>8}{'mAP50-95':>10}{'P':>7}{'R':>7}" + ("" if args.no_speed else f"{'ms/img':>9}{'FPS':>7}"))
    for r in rows:
        line = f"{r['modele']:<16}{r['mAP50']:>8.3f}{r['mAP50-95']:>10.3f}{r['precision']:>7.3f}{r['rappel']:>7.3f}"
        if not args.no_speed:
            line += f"{r['total_ms']:>9.1f}{r['FPS']:>7.1f}"
        print(line)
    print(f"\nRésultats : {OUT_DIR}  (matériel : {device_name})")


if __name__ == "__main__":
    main()
