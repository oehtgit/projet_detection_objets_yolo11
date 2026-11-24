"""
Contrôle qualité et statistiques sur les données VOC (format YOLO).

On vérifie :
  - que chaque image a son fichier de labels
  - que les coordonnées sont entre 0 et 1 et les classes valides
On calcule :
  - images et objets par ensemble (train / val / test)
  - nombre d'objets par classe
  - tailles des objets (petit / moyen / grand, seuils COCO)
  - tailles des images
Les résultats vont dans data/stats/.

Exemple :
    python data/data_eda.py --root /content/datasets/VOC
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

SPLITS = {"train": ["train"], "val": ["val"], "test": ["test"]}


def read_split(root: Path, sub: str, names: dict) -> pd.DataFrame:
    """Lit les labels d'un ensemble et note les problèmes trouvés."""
    rows, problems = [], Counter()
    img_dir, lbl_dir = root / "images" / sub, root / "labels" / sub
    for img in sorted(img_dir.glob("*.jpg")):
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            problems["label manquant"] += 1
            continue
        with Image.open(img) as im:
            W, H = im.size
        lines = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
        if not lines:
            rows.append({"split_dir": sub, "image": img.stem, "W": W, "H": H, "cls": None})
        for p in lines:
            c, x, y, w, h = int(p[0]), *map(float, p[1:5])
            if c not in names:
                problems["classe invalide"] += 1
            if not all(0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0:
                problems["coordonnées hors [0,1]"] += 1
            rows.append({"split_dir": sub, "image": img.stem, "W": W, "H": H, "cls": c,
                         "w_px": w * W, "h_px": h * H})
    return pd.DataFrame(rows), problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/datasets/VOC")
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--out", default="data/stats")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = {int(k): v for k, v in yaml.safe_load(Path(a.data).read_text())["names"].items()}

    frames, problems = [], Counter()
    for split, subs in SPLITS.items():
        for sub in subs:
            df, pb = read_split(root, sub, names)
            df["split"] = split
            frames.append(df)
            problems.update(pb)
    df = pd.concat(frames, ignore_index=True)
    obj = df.dropna(subset=["cls"]).copy()
    obj["cls"] = obj["cls"].astype(int)
    obj["classe"] = obj["cls"].map(names)
    obj["aire"] = obj["w_px"] * obj["h_px"]
    obj["taille"] = pd.cut(obj["aire"], [0, 32**2, 96**2, np.inf], labels=["petit", "moyen", "grand"])
    obj["ratio_l_h"] = obj["w_px"] / obj["h_px"]

    # 1. résumé par ensemble
    imgs = df.drop_duplicates(["split_dir", "image"])
    per_img = obj.groupby(["split", "split_dir", "image"]).size()
    summary = []
    for s in SPLITS:
        n_img = int((imgs["split"] == s).sum())
        n_obj = int((obj["split"] == s).sum())
        empty = int(df[(df["split"] == s) & df["cls"].isna()].shape[0])
        summary.append({"split": s, "images": n_img, "objets": n_obj,
                        "objets/image (moy.)": round(n_obj / n_img, 2),
                        "objets/image (max)": int(per_img[s].max()), "images sans objet": empty})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "resume_splits.csv", index=False)

    # 2. objets par classe
    cls_tab = obj.pivot_table(index="classe", columns="split", values="cls", aggfunc="size", fill_value=0)
    cls_tab = cls_tab[["train", "val", "test"]].sort_values("train", ascending=False)
    cls_tab["part train (%)"] = (100 * cls_tab["train"] / cls_tab["train"].sum()).round(1)
    cls_tab.to_csv(out / "objets_par_classe.csv")

    # 3. tailles des objets
    size_tab = (obj.groupby("split")["taille"].value_counts(normalize=True).unstack()[["petit", "moyen", "grand"]] * 100).round(1)
    size_tab.to_csv(out / "tailles_boites_pct.csv")
    size_cls = (obj[obj.split == "train"].groupby("classe")["taille"].value_counts(normalize=True)
                .unstack()[["petit", "moyen", "grand"]] * 100).round(1).sort_values("petit", ascending=False)
    size_cls.to_csv(out / "tailles_par_classe_train_pct.csv")

    # graphiques
    fig, ax = plt.subplots(figsize=(8, 6))
    cls_tab[["train", "val", "test"]].iloc[::-1].plot.barh(ax=ax, width=0.8)
    ax.set_xlabel("nombre d'objets"); ax.set_title("Pascal VOC : objets par classe et par split")
    plt.tight_layout(); plt.savefig(out / "objets_par_classe.png", dpi=150); plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    tr = obj[obj.split == "train"]
    axs[0].hist(np.sqrt(tr["aire"]), bins=60, color="#4c72b0")
    for t, l in [(32, "32 px"), (96, "96 px")]:
        axs[0].axvline(t, color="k", ls="--", lw=1); axs[0].text(t + 3, axs[0].get_ylim()[1] * 0.9, l)
    axs[0].set_xlabel("√(aire de la boîte) en pixels"); axs[0].set_title("Taille des objets (train)")
    axs[1].hist(np.log2(tr["ratio_l_h"]), bins=60, color="#dd8452")
    axs[1].set_xlabel("log2(largeur / hauteur)"); axs[1].set_title("Forme des boîtes (0 = carré)")
    plt.tight_layout(); plt.savefig(out / "tailles_boites.png", dpi=150); plt.close()

    res = {
        "integrite": dict(problems) or "aucun problème détecté",
        "tailles_images_px": {"largeur_mediane": int(imgs["W"].median()), "hauteur_mediane": int(imgs["H"].median()),
                              "largeur_max": int(imgs["W"].max()), "hauteur_max": int(imgs["H"].max())},
        "desequilibre_train": {"classe_max": cls_tab.index[0], "n_max": int(cls_tab["train"].iloc[0]),
                               "classe_min": cls_tab.index[-1], "n_min": int(cls_tab["train"].iloc[-1]),
                               "rapport": round(cls_tab["train"].iloc[0] / cls_tab["train"].iloc[-1], 1)},
    }
    (out / "resume_donnees.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False), "\n")
    print(cls_tab.to_string(), "\n")
    print(size_tab.to_string(), "\n")
    print(size_cls.head(6).to_string(), "\n")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
