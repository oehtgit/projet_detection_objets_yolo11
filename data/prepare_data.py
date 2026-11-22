"""
Préparation des données Pascal VOC pour YOLO.

Étapes :
  1. télécharger les 3 archives VOC et les décompresser
  2. lire les listes officielles : 21 503 images annotées (VOC2007 + VOC2012)
  3. convertir chaque XML en label YOLO (on enlève les objets "difficult")
  4. découper en 70 % train / 10 % val / 20 % test (stratifié, graine fixe)
  5. écrire data/data.yaml

Avec --drive-id, on récupère directement le zip déjà prêt sur le Drive.
Si ça ne marche pas, on refait toute la préparation.

Exemples (Colab) :
    python data/prepare_data.py --out /content/datasets/VOC
    python data/prepare_data.py --out /content/datasets/VOC --drive-id <ID>
    python data/prepare_data.py --out /content/datasets/VOC --zip
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

# Les 20 classes VOC, dans l'ordre officiel (indice = position dans la liste)
VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]
# 6 classes n'ont pas le même nom dans COCO. On garde le nom COCO pour que
# le fine-tuning réutilise ce que le modèle a déjà appris.
COCO_SPELLING = {"aeroplane": "airplane", "diningtable": "dining table", "motorbike": "motorcycle",
                 "pottedplant": "potted plant", "sofa": "couch", "tvmonitor": "tv"}

# Les 5 listes officielles de VOC : (année, liste, nom)
SPLITS = [("2012", "train", "train2012"), ("2012", "val", "val2012"),
          ("2007", "train", "train2007"), ("2007", "val", "val2007"), ("2007", "test", "test2007")]


# ----- Étape 1 : téléchargement
def download_official(raw: Path):
    raw.mkdir(parents=True, exist_ok=True)
    # si ce fichier existe, l'archive est déjà décompressée
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


# ----- Étape 3 : conversion XML -> YOLO
def convert_annotation(xml_path: Path):
    """Lit un XML et renvoie les lignes YOLO : classe xc yc w h."""
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
        # VOC compte les pixels à partir de 1, d'où le -1
        xc = ((xmin + xmax) / 2 - 1) / W
        yc = ((ymin + ymax) / 2 - 1) / H
        w = (xmax - xmin) / W
        h = (ymax - ymin) / H
        lines.append(f"{VOC_CLASSES.index(name)} {xc} {yc} {w} {h}")
    return lines, n_difficult


# ----- Étape 4 : découpage
def stratified_split(items, ratios=(0.7, 0.1, 0.2), seed=0):
    """Découpe stratifié : chaque image va dans le groupe de sa classe la plus rare,
    puis chaque groupe est coupé en 70/10/20. Renvoie {image: 'train', 'val' ou 'test'}."""
    freq = Counter(c for _, cls in items for c in set(cls))          # nb d'images par classe
    groups = {}
    for i, cls in items:
        key = min(set(cls), key=lambda c: freq[c]) if cls else -1     # classe la plus rare
        groups.setdefault(key, []).append(i)
    rng = random.Random(seed)                                          # même graine = même découpage
    assign = {}
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        n = len(ids)
        n_train, n_val = round(ratios[0] * n), round(ratios[1] * n)
        for j, i in enumerate(ids):
            assign[i] = "train" if j < n_train else "val" if j < n_train + n_val else "test"
    return assign


# ----- Étapes 2 à 4 : lire, convertir, découper, ranger
def build_yolo(raw: Path, out: Path, move: bool = True, mode: str = "aleatoire",
               ratios=(0.7, 0.1, 0.2), seed: int = 0):
    # lire et convertir toutes les images
    data = []                        # (id, image, lignes YOLO, liste d'origine)
    n_diff = 0
    for year, liste, sub in SPLITS:
        dev = raw / "VOCdevkit" / f"VOC{year}"
        for i in (dev / "ImageSets" / "Main" / f"{liste}.txt").read_text().split():
            lines, nd = convert_annotation(dev / "Annotations" / f"{i}.xml")
            data.append((i, dev / "JPEGImages" / f"{i}.jpg", lines, sub))
            n_diff += nd
    print(f"[2-3] {len(data)} images, {sum(len(d[2]) for d in data)} objets, {n_diff} 'difficult' écartés")

    # découpage (officiel ou 70/10/20)
    if mode == "officiel":
        official = {"train2012": "train", "val2012": "train", "train2007": "train",
                    "val2007": "val", "test2007": "test"}
        assign = {i: official[sub] for i, _, _, sub in data}
    else:
        items = [(i, [int(l.split()[0]) for l in lines]) for i, _, lines, _ in data]
        assign = stratified_split(items, ratios, seed)

    # écriture des images et des labels
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


# ----- Étape 5 : fichier data.yaml
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
        "# Pascal VOC au format YOLO, généré par data/prepare_data.py\n"
        "# découpage 70/10/20 stratifié (graine 0)\n"
        + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"[5] configuration écrite : {yaml_path}")


# ----- Récupération du zip sur le Drive
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
    except Exception as e:  # quota Google, fichier supprimé, pas de réseau
        print(f"[A] échec du téléchargement Drive ({e}) -> repli sur la source officielle")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/datasets/VOC")
    ap.add_argument("--raw", default=None, help="dossier des archives (par défaut : <out>_brut)")
    ap.add_argument("--drive-id", default=None, help="ID du zip sur Google Drive")
    ap.add_argument("--yaml", default="data/data.yaml")
    ap.add_argument("--zip", action="store_true", help="crée aussi un zip des données")
    ap.add_argument("--split", default="aleatoire", choices=["aleatoire", "officiel"],
                    help="aleatoire (70/10/20) ou officiel (découpage VOC)")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.1, 0.2], help="train val test")
    ap.add_argument("--seed", type=int, default=0, help="graine du hasard")
    ap.add_argument("--keep-raw", action="store_true", help="copier les images au lieu de les déplacer")
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
