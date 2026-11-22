"""
Graphiques sur les données VOC (format YOLO). Les figures vont dans data/figures/.

Exemple :
    python data/data_viz.py --root /content/datasets/VOC
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np
import pandas as pd
import yaml
from PIL import Image

# couleurs des graphiques
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SPLITC = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
SPLITS = {"train": ["train"], "val": ["val"], "test": ["test"]}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 11.5, "axes.titlecolor": INK, "axes.titlelocation": "left"})


def load(root, names):
    """Lit tous les labels : une ligne par objet."""
    rows = []
    for split, subs in SPLITS.items():
        for sub in subs:
            for f in (root / "labels" / sub).glob("*.txt"):
                lines = [l.split() for l in f.read_text().splitlines() if l.strip()]
                with Image.open(root / "images" / sub / f"{f.stem}.jpg") as im:
                    W, H = im.size
                for p in lines:
                    c, x, y, w, h = int(p[0]), *map(float, p[1:])
                    rows.append(dict(split=split, sub=sub, img=f.stem, W=W, H=H, cls=names[c],
                                     x=x, y=y, w=w, h=h, wpx=w * W, hpx=h * H))
    d = pd.DataFrame(rows)
    d["aire"] = d.wpx * d.hpx
    d["taille"] = pd.cut(d.aire, [0, 32**2, 96**2, np.inf], labels=["petit", "moyen", "grand"])
    return d


def draw_boxes(ax, img_path, objs, title=None):
    """Affiche une image avec ses boîtes et le nom des classes."""
    im = Image.open(img_path); W, H = im.size
    ax.imshow(im); ax.axis("off")
    for _, o in objs.iterrows():
        x0, y0 = (o.x - o.w / 2) * W, (o.y - o.h / 2) * H
        ax.add_patch(mp.Rectangle((x0, y0), o.w * W, o.h * H, fill=False, ec="#eda100", lw=2))
        ax.text(x0 + 2, y0 + 2, o.cls, va="top", fontsize=8, color="black",
                bbox=dict(fc="#eda100", ec="none", pad=1.5))
    if title:
        ax.set_title(title, fontsize=9, color=INK2, fontweight="normal", loc="center")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/datasets/VOC")
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--out", default="data/figures")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = {int(k): v for k, v in yaml.safe_load(Path(a.data).read_text())["names"].items()}
    d = load(root, names)

    # 1. exemples d'images annotées
    ids = ["000005", "000009", "000012", "000019", "000032", "000063"]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, i in zip(axs.flat, ids):
        o = d[d.img == i]
        sub = o["sub"].iloc[0]
        draw_boxes(ax, root / "images" / sub / f"{i}.jpg", o, f"image {i} · {len(o)} objet(s)")
    fig.suptitle("Ce que voit le modèle : une image + des boîtes étiquetées", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    plt.tight_layout(); plt.savefig(out / "fig_exemples.png", dpi=150); plt.close()

    # 2. conversion VOC -> YOLO sur l'image 000001
    o = d[d.img == "000001"]
    im = Image.open(root / "images" / o["sub"].iloc[0] / "000001.jpg"); W, H = im.size
    dog = o[o.cls == "dog"].iloc[0]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 6.2))
    for ax, mode in zip(axs, ["voc", "yolo"]):
        ax.imshow(im); ax.set_xticks([0, W]); ax.set_yticks([0, H])
        x0, y0, bw, bh = (dog.x - dog.w / 2) * W, (dog.y - dog.h / 2) * H, dog.w * W, dog.h * H
        ax.add_patch(mp.Rectangle((x0, y0), bw, bh, fill=False, ec="#eda100", lw=2.5))
        if mode == "voc":
            for (px, py, t, ha) in [(x0, y0, "(xmin, ymin)\n= (48, 240)", "right"),
                                    (x0 + bw, y0 + bh, "(xmax, ymax)\n= (195, 371)", "left")]:
                ax.plot(px, py, "o", ms=8, color="#e34948", mec="white")
                ax.annotate(t, (px, py), xytext=(-8 if ha == "right" else 8, -14 if ha == "right" else 14),
                            textcoords="offset points", ha=ha, fontsize=9, color="white",
                            bbox=dict(fc="#e34948", ec="none", pad=2))
            ax.set_title("Format VOC (XML) : deux coins, en pixels")
        else:
            cx, cy = dog.x * W, dog.y * H
            ax.plot(cx, cy, "o", ms=8, color="#2a78d6", mec="white")
            ax.annotate("centre\n(0,341 ; 0,609)", (cx, cy), xytext=(-12, 18), ha="right", textcoords="offset points",
                        fontsize=9, color="white", bbox=dict(fc="#2a78d6", ec="none", pad=2))
            ax.annotate("", (x0, y0 - 6), (x0 + bw, y0 - 6), arrowprops=dict(arrowstyle="<->", color="#2a78d6", lw=1.8))
            ax.text(x0 + bw / 2, y0 - 12, "largeur = 0,416", ha="center", fontsize=9, color="white",
                    bbox=dict(fc="#2a78d6", ec="none", pad=2))
            ax.annotate("", (x0 + bw + 6, y0), (x0 + bw + 6, y0 + bh), arrowprops=dict(arrowstyle="<->", color="#2a78d6", lw=1.8))
            ax.text(x0 + bw + 10, y0 + bh / 2, "hauteur\n= 0,262", va="center", fontsize=9, color="white",
                    bbox=dict(fc="#2a78d6", ec="none", pad=2))
            ax.set_xticklabels(["0", "1"]); ax.set_yticklabels(["0", "1"])
            ax.set_title("Format YOLO (.txt) : centre + taille, entre 0 et 1")
        ax.spines[:].set_visible(False)
    plt.tight_layout(); plt.savefig(out / "fig_conversion.png", dpi=150); plt.close()

    # 3. le découpage
    imgs = d.drop_duplicates(["sub", "img"]).groupby("split").size().reindex(SPLITS)
    objs = d.groupby("split").size().reindex(SPLITS)
    fig, ax = plt.subplots(figsize=(10, 2.6))
    left = 0
    for s in SPLITS:
        ax.barh(0, imgs[s], left=left, color=SPLITC[s], height=0.55, ec="white", lw=2)
        pct = 100 * imgs[s] / imgs.sum()
        ax.text(left + imgs[s] / 2, -0.45, f"{s}\n{imgs[s]:,} images\n{pct:.0f} %".replace(",", " "),
                ha="center", va="top", color=INK, fontsize=9.5)
        left += imgs[s]
    ax.set_xlim(0, imgs.sum()); ax.set_ylim(-1.35, 0.4); ax.set_yticks([]); ax.spines[:].set_visible(False); ax.set_xticks([])
    ax.set_title(f"Découpage aléatoire stratifié des {imgs.sum():,} images : 70 / 10 / 20 (graine 0)".replace(",", " "))
    plt.tight_layout(); plt.savefig(out / "fig_split.png", dpi=150); plt.close()

    # 4. part de chaque classe dans chaque ensemble
    share = (d.groupby(["split", "cls"]).size() / d.groupby("split").size()).unstack(0)[list(SPLITS)] * 100
    share = share.sort_values("train")
    fig, ax = plt.subplots(figsize=(8, 7))
    yy = np.arange(len(share))
    ax.hlines(yy, share.min(1), share.max(1), color=GRID, lw=3)
    for s in SPLITS:
        ax.plot(share[s], yy, "o", ms=7, color=SPLITC[s], mec="white", mew=1.5, label=s)
    ax.set_yticks(yy, share.index); ax.set_xlabel("part des objets de l'ensemble (%)")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", title="ensemble")
    ax.set_title("Les 3 ensembles ont la même répartition des classes")
    plt.tight_layout(); plt.savefig(out / "fig_classes_par_split.png", dpi=150); plt.close()

    # 5. nombre d'objets par classe (train)
    cnt = d[d.split == "train"].cls.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(cnt.index, cnt.values, color="#2a78d6", height=0.7)
    for i, v in enumerate(cnt.values):
        if i in (0, len(cnt) - 1, len(cnt) - 2):
            ax.text(v + 120, i, f"{v:,}".replace(",", " "), va="center", fontsize=9, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xlabel("nombre d'objets (train)")
    ax.set_title(f"Déséquilibre : '{cnt.index[-1]}' est {cnt.iloc[-1] / cnt.iloc[0]:.1f} fois plus fréquente que '{cnt.index[0]}'".replace(".", ","))
    plt.tight_layout(); plt.savefig(out / "fig_classes_train.png", dpi=150); plt.close()

    # 6. objets par image
    per = d.groupby(["split", "sub", "img"]).size().reset_index(name="n")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    tr = per[per.split == "train"].n.clip(upper=10).value_counts().sort_index()
    ax.bar(tr.index, 100 * tr.values / tr.sum(), color="#2a78d6", width=0.7)
    ax.set_xticks(range(1, 11), [str(i) for i in range(1, 10)] + ["10+"])
    ax.set_xlabel("nombre d'objets dans l'image"); ax.set_ylabel("% des images (train)")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(f"{100 * (per[per.split == 'train'].n == 1).mean():.0f} % des images n'ont qu'un seul objet ; moyenne {per[per.split=='train'].n.mean():.1f}")
    plt.tight_layout(); plt.savefig(out / "fig_objets_par_image.png", dpi=150); plt.close()

    # 7. tailles des objets, par ensemble et par classe
    sz = (d.groupby("split").taille.value_counts(normalize=True).unstack()[["petit", "moyen", "grand"]] * 100).reindex(list(SPLITS))
    szc = (d[d.split == "train"].groupby("cls").taille.value_counts(normalize=True).unstack()[["petit", "moyen", "grand"]] * 100).sort_values("petit")
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [1, 1.6]})
    TC = {"petit": "#e34948", "moyen": "#eda100", "grand": "#1baf7a"}
    for ax, tab, title in [(axs[0], sz, "Par ensemble"), (axs[1], szc, "Par classe (train)")]:
        left = np.zeros(len(tab))
        for t in ["petit", "moyen", "grand"]:
            ax.barh(tab.index, tab[t], left=left, color=TC[t], ec="white", lw=1, label=t, height=0.7)
            left += tab[t].values
        ax.set_xlim(0, 100); ax.set_xlabel("% des objets"); ax.set_title(title)
    axs[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.suptitle("Taille des objets : surtout grands ; 'bottle' a le plus de petits objets (< 32×32 px)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    plt.tight_layout(); plt.savefig(out / "fig_tailles.png", dpi=150); plt.close()

    # 8. taille des images
    im = d.drop_duplicates(["sub", "img"])
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    c = im.groupby(["W", "H"]).size().reset_index(name="n")
    ax.scatter(c.W, c.H, s=np.clip(c.n / 15, 4, 400), color="#2a78d6", alpha=0.55, ec="white", lw=0.5)
    ax.axvline(640, color="#e34948", ls="--", lw=1); ax.axhline(640, color="#e34948", ls="--", lw=1)
    ax.text(630, 120, "taille d'entrée\nYOLO = 640", color="#e34948", ha="right", fontsize=9)
    ax.set_xlim(0, 680); ax.set_ylim(0, 680); ax.set_xlabel("largeur (px)"); ax.set_ylabel("hauteur (px)")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("Toutes les images font au plus 500 px : YOLO les agrandit")
    plt.tight_layout(); plt.savefig(out / "fig_taille_images.png", dpi=150); plt.close()

    # quelques chiffres à afficher
    print(imgs.to_dict(), objs.to_dict())
    print("1 objet:", round(100 * (per[per.split == 'train'].n == 1).mean(), 1), "%")
    print(sz.round(1).to_string())
    print("images 500 max:", im.W.max(), im.H.max(), "% 500x375 ou 375x500:",
          round(100 * (((im.W == 500) & (im.H == 375)) | ((im.W == 375) & (im.H == 500))).mean(), 1))


if __name__ == "__main__":
    main()
