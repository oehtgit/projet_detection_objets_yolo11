"""
Partie DONNÉES (Amina) — préparation de Pascal VOC pour YOLO.

Deux modes :
  A) --drive-id <ID>  : télécharge le zip DÉJÀ PRÊT depuis le Google Drive partagé (rapide).
  B) sinon (ou si A échoue) : télécharge les archives officielles et fait toute la préparation.

Préparation (mode B), étape par étape :
  1. Téléchargement des 3 archives VOC (2007 trainval, 2007 test, 2012 trainval) et décompression.
  2. Lecture des listes officielles ImageSets/Main/{train,val,test}.txt : elles donnent les
     21 503 images annotées pour la détection (VOC2007 + VOC2012).
  3. Pour chaque image : lecture du XML, on écarte les objets 'difficult', conversion de chaque
     boîte (xmin, ymin, xmax, ymax en pixels) en format YOLO normalisé (x_centre, y_centre, w, h).
  4. Découpage 70 % train / 10 % val / 20 % test, aléatoire STRATIFIÉ (graine fixe) :
     chaque image est rangée selon la classe la plus rare qu'elle contient, et chaque groupe est
     découpé 70/10/20, pour que les 3 ensembles aient les mêmes proportions de classes.
     Écriture : images/<split>/xxx.jpg et labels/<split>/xxx.txt (une ligne par objet).
     (--split officiel garde l'ancien découpage officiel de VOC.)
  5. Écriture du fichier de configuration du dataset (data/data.yaml).

Usage (Colab) :
    python data/prepare_data.py --out /content/datasets/VOC                    # mode B
    python data/prepare_data.py --out /content/datasets/VOC --drive-id <ID>    # mode A, repli sur B
    python data/prepare_data.py --out /content/datasets/VOC --zip              # B + crée VOC_pret.zip
"""
import argparse
import random
from collections import Counter
import shutil
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import yaml

URL = "https://ultralytics.com/assets/"
ARCHIVES = ["VOCtrainval_06-Nov-2007.zip", "VOCtest_06-Nov-2007.zip", "VOCtrainval_11-May-2012.zip"]

# Ordre officiel des 20 classes VOC (noms tels qu'écrits dans les XML)
VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]
# Noms écrits dans le .yaml : orthographe COCO pour 6 classes, afin que le fine-tuning réutilise
# la tête de classification COCO (Ultralytics associe les classes par leur nom).
COCO_SPELLING = {"aeroplane": "airplane", "diningtable": "dining table", "motorbike": "motorcycle",
                 "pottedplant": "potted plant", "sofa": "couch", "tvmonitor": "tv"}

# (année, liste officielle) -> dossier de sortie
SPLITS = [("2012", "train", "train2012"), ("2012", "val", "val2012"),
          ("2007", "train", "train2007"), ("2007", "val", "val2007"), ("2007", "test", "test2007")]


# ---------------------------------------------------------------- étape 1 : téléchargement
def download_official(raw: Path):
    raw.mkdir(parents=True, exist_ok=True)
    # fichier présent si l'archive a déjà été décompressée (évite de retélécharger)
    marker = {"VOCtrainval_06-Nov-2007.zip": "VOCdevkit/VOC2007/ImageSets/Main/train.txt",
              "VOCtest_06-Nov-2007.zip": "VOCdevkit/VOC2007/ImageSets/Main/test.txt",
              "VOCtrainval_11-May-2012.zip": "VOCdevkit/VOC2012/ImageSets/Main/train.txt"}
    for name in ARCHIVES:
        z = raw / name
        if (raw / marker[name]).exists():
            print(f"[1] {name} déjà présent")
            continue
        print(f"[1] téléchargement {name}")
        urllib.request.urlretrieve(URL + name, z)
        print(f"[1] décompression {name}")
        with zipfile.ZipFile(z) as f:
            f.extractall(raw)
        z.unlink()


# ---------------------------------------------------------------- étape 3 : conversion
def convert_annotation(xml_path: Path):
    """Retourne la liste des lignes YOLO 'classe xc yc w h' (objets 'difficult' exclus)."""
    root = ET.parse(xml_path).getroot()
    W = int(root.find("size/width").text)
    H = int(root.find("size/height").text)
    lines, n_difficult = [], 0
    for obj in root.iter("object"):
        name = obj.find("name").text.strip()
        d = obj.find("difficult")
        if d is not None and d.text.strip() == "1":
            n_difficult += 1
            continue
        b = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        # VOC numérote les pixels à partir de 1 -> on retire 1 au centre
        xc = ((xmin + xmax) / 2 - 1) / W
        yc = ((ymin + ymax) / 2 - 1) / H
        w = (xmax - xmin) / W
        h = (ymax - ymin) / H
        lines.append(f"{VOC_CLASSES.index(name)} {xc} {yc} {w} {h}")
    return lines, n_difficult


# ---------------------------------------------------------------- étape 4 : découpage
def stratified_split(items, ratios=(0.7, 0.1, 0.2), seed=0):
    """items : liste de (identifiant, classes présentes). Retourne {identifiant: 'train'|'val'|'test'}.

    Stratification simple pour la détection (plusieurs classes par image) : on range chaque image
    dans le groupe de la classe la PLUS RARE qu'elle contient, puis on découpe chaque groupe 70/10/20.
    Les classes rares sont ainsi présentes dans les 3 ensembles en bonne proportion.
    """
    freq = Counter(c for _, cls in items for c in set(cls))          # nb d'images par classe
    groups = {}
    for i, cls in items:
        key = min(set(cls), key=lambda c: freq[c]) if cls else -1     # -1 : image sans objet
        groups.setdefault(key, []).append(i)
    rng = random.Random(seed)                                          # graine fixe = reproductible
    assign = {}
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        n = len(ids)
        n_train, n_val = round(ratios[0] * n), round(ratios[1] * n)
        for j, i in enumerate(ids):
            assign[i] = "train" if j < n_train else "val" if j < n_train + n_val else "test"
    return assign


# ---------------------------------------------------------------- étapes 2 à 4
def build_yolo(raw: Path, out: Path, move: bool = True, mode: str = "aleatoire",
               ratios=(0.7, 0.1, 0.2), seed: int = 0):
    # 2-3. lire toutes les images annotées et convertir leurs annotations
    data = []                        # (identifiant, chemin de l'image, lignes YOLO, dossier officiel)
    n_diff = 0
    for year, liste, sub in SPLITS:
        dev = raw / "VOCdevkit" / f"VOC{year}"
        for i in (dev / "ImageSets" / "Main" / f"{liste}.txt").read_text().split():
            lines, nd = convert_annotation(dev / "Annotations" / f"{i}.xml")
            data.append((i, dev / "JPEGImages" / f"{i}.jpg", lines, sub))
            n_diff += nd
    print(f"[2-3] {len(data)} images, {sum(len(d[2]) for d in data)} objets, {n_diff} 'difficult' écartés")

    # 4. découpage
    if mode == "officiel":
        official = {"train2012": "train", "val2012": "train", "train2007": "train",
                    "val2007": "val", "test2007": "test"}
        assign = {i: official[sub] for i, _, _, sub in data}
    else:
        items = [(i, [int(l.split()[0]) for l in lines]) for i, _, lines, _ in data]
        assign = stratified_split(items, ratios, seed)

    # 4bis. écriture
    total = {}
    for i, src, lines, _ in data:
        sp = assign[i]
        img_out, lbl_out = out / "images" / sp, out / "labels" / sp
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        (lbl_out / f"{i}.txt").write_text("\n".join(lines) + "\n" if lines else "")
        if src.exists():
            (shutil.move if move else shutil.copy2)(src, img_out / f"{i}.jpg")
        t = total.setdefault(sp, {"images": 0, "objets": 0})
        t["images"] += 1
        t["objets"] += len(lines)
    n = len(data)
    for sp in ("train", "val", "test"):
        print(f"[4] {sp:5s} {total[sp]['images']:6d} images ({100 * total[sp]['images'] / n:4.1f} %)"
              f"  {total[sp]['objets']:6d} objets")
    return total


# ---------------------------------------------------------------- étape 5 : configuration
def write_yaml(out: Path, yaml_path: Path):
    cfg = {
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: COCO_SPELLING.get(c, c) for i, c in enumerate(VOC_CLASSES)},
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        "# Pascal VOC au format YOLO — généré par src/prepare_data.py\n"
        "# découpage 70/10/20 stratifié (graine 0) | val : choix de best.pt | test : score final\n"
        + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"[5] configuration écrite : {yaml_path}")


# ---------------------------------------------------------------- mode A : Drive
def from_drive(file_id: str, out: Path) -> bool:
    try:
        subprocess.run(["pip", "install", "-q", "gdown"], check=True)
        z = out.parent / "VOC_pret.zip"
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gdown", file_id, "-O", str(z)], check=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(out.parent)
        z.unlink()
        ok = (out / "labels" / "test").exists()
        print("[A] données récupérées depuis le Drive" if ok else "[A] zip incomplet")
        return ok
    except Exception as e:  # quota Google, fichier supprimé, réseau...
        print(f"[A] échec du téléchargement Drive ({e}) -> repli sur la source officielle")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/datasets/VOC")
    ap.add_argument("--raw", default=None, help="dossier des archives brutes (défaut : <out>_brut)")
    ap.add_argument("--drive-id", default=None, help="ID Google Drive du zip VOC_pret.zip")
    ap.add_argument("--yaml", default="data/data.yaml")
    ap.add_argument("--zip", action="store_true", help="crée VOC_pret.zip à côté du dossier de sortie")
    ap.add_argument("--split", default="aleatoire", choices=["aleatoire", "officiel"],
                    help="aleatoire = 70/10/20 stratifié (défaut) ; officiel = découpage officiel VOC")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.1, 0.2], help="train val test")
    ap.add_argument("--seed", type=int, default=0, help="graine du tirage aléatoire")
    ap.add_argument("--keep-raw", action="store_true", help="copier au lieu de déplacer les images")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    raw = Path(a.raw).resolve() if a.raw else out.parent / (out.name + "_brut")

    if not (a.drive_id and from_drive(a.drive_id, out)):
        download_official(raw)
        build_yolo(raw, out, move=not a.keep_raw, mode=a.split, ratios=tuple(a.ratios), seed=a.seed)
        if not a.keep_raw:
            shutil.rmtree(raw, ignore_errors=True)
    write_yaml(out, Path(a.yaml))

    if a.zip:
        z = out.parent / "VOC_pret"
        print(f"[zip] création de {z}.zip ...")
        shutil.make_archive(str(z), "zip", root_dir=out.parent, base_dir=out.name)
        print(f"[zip] {z}.zip : {Path(str(z) + '.zip').stat().st_size / 1e9:.2f} Go")


if __name__ == "__main__":
    main()
