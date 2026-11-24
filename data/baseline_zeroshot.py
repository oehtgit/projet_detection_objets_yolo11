"""
Baseline : YOLO11n pré-entraîné sur COCO, testé tel quel (sans adaptation) sur notre jeu de test.

Problème : COCO a 80 classes et VOC 20, avec des numéros différents.
Si on lance model.val() directement, les numéros ne correspondent pas et la mAP est fausse.
Solution : on renumérote les labels VOC avec les numéros COCO, puis on évalue normalement.
Les prédictions des classes COCO qui ne sont pas dans VOC sont ignorées.

Exemple :
    python data/baseline_zeroshot.py --model yolo11n.pt --data data/data.yaml --split test
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

# noms VOC -> noms COCO (quand ils sont différents)
VOC2COCO_ALIASES = {
    "aeroplane": "airplane",
    "diningtable": "dining table",
    "motorbike": "motorcycle",
    "pottedplant": "potted plant",
    "sofa": "couch",
    "tvmonitor": "tv",
}


def build_mapping(voc_names: dict, coco_names: dict) -> dict:
    """Renvoie {numéro VOC: numéro COCO} pour les 20 classes."""
    coco_inv = {name: idx for idx, name in coco_names.items()}
    mapping = {}
    for i, name in voc_names.items():
        target = VOC2COCO_ALIASES.get(name, name)
        if target not in coco_inv:
            raise KeyError(f"Classe VOC '{name}' introuvable dans COCO (cherché '{target}')")
        mapping[int(i)] = coco_inv[target]
    assert len(set(mapping.values())) == len(mapping), "deux classes VOC sur la même classe COCO"
    return mapping


def link_or_copy(src: Path, dst: Path):
    """Lie chaque image au nouveau dossier (pas de copie), sinon on copie.
    Pas de lien de dossier : Ultralytics le suivrait et relirait les anciens labels."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if not f.is_file():
            continue
        target = dst / f.name
        if target.exists():
            continue
        try:
            os.link(f, target)
        except OSError:
            shutil.copy2(f, target)


def remap_split(data: dict, mapping: dict, coco_names: dict, out_root: Path, split: str = "val") -> Path:
    """Crée une copie du jeu avec les labels renumérotés en classes COCO."""
    key = split if data.get(split) else "val"
    print(f"[remap] split évalué : {key}")
    val_dirs = data[key] if isinstance(data[key], list) else [data[key]]
    rel_dirs = []
    for vdir in val_dirs:
        vdir = Path(vdir)
        rel = vdir.relative_to(Path(data["path"]))          # ex: images/test
        lbl_src = Path(str(vdir).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
        img_dst = out_root / rel
        lbl_dst = Path(str(img_dst).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
        link_or_copy(vdir, img_dst)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        n_files, n_boxes = 0, 0
        for f in lbl_src.glob("*.txt"):
            lines = []
            for line in f.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                parts[0] = str(mapping[int(parts[0])])
                lines.append(" ".join(parts))
                n_boxes += 1
            (lbl_dst / f.name).write_text("\n".join(lines) + ("\n" if lines else ""))
            n_files += 1
        print(f"[remap] {rel}: {n_files} fichiers, {n_boxes} boîtes réindexées")
        rel_dirs.append(str(rel))

    yaml_path = out_root / "voc_coco_ids.yaml"
    cfg = {
        "path": str(out_root),
        "train": rel_dirs[0],  # requis par Ultralytics, non utilisé ici
        "val": rel_dirs if len(rel_dirs) > 1 else rel_dirs[0],
        "names": {int(k): v for k, v in coco_names.items()},
    }
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return yaml_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default=None, help="0 pour GPU, cpu pour CPU")
    p.add_argument("--split", default="test", help="split évalué (test si présent, sinon val)")
    p.add_argument("--out", default="data/stats")
    args = p.parse_args()

    model = YOLO(args.model)
    coco_names = model.names                       # 80 classes COCO
    data = check_det_dataset(args.data)            # lit data.yaml (données préparées par prepare_data.py)
    voc_names = data["names"]
    mapping = build_mapping(voc_names, coco_names)
    print("[mapping VOC->COCO]", {voc_names[k]: coco_names[v] for k, v in mapping.items()})

    out_root = Path(data["path"]).parent / (Path(data["path"]).name + "_coco_ids")
    yaml_path = remap_split(data, mapping, coco_names, out_root, args.split)

    metrics = model.val(data=str(yaml_path), imgsz=args.imgsz, batch=args.batch,
                        device=args.device, split="val", plots=True,
                        project=str(Path.cwd() / "runs" / "val"), name="baseline_zeroshot", exist_ok=True)

    # résultats par classe, avec les noms VOC
    coco2voc = {v: voc_names[k] for k, v in mapping.items()}
    per_class = {}
    for j, c in enumerate(metrics.box.ap_class_index):
        per_class[coco2voc.get(int(c), coco_names[int(c)])] = {
            "AP50": float(metrics.box.ap50[j]), "AP50-95": float(metrics.box.ap[j])}

    res = {
        "model": args.model, "setting": "zero-shot COCO -> VOC (labels remappés)",
        "imgsz": args.imgsz,
        "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
        "speed_ms": metrics.speed, "per_class": per_class,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_file = Path(args.out) / "baseline_zeroshot.json"
    out_file.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nmAP50 = {res['mAP50']:.4f} | mAP50-95 = {res['mAP50-95']:.4f} -> {out_file}")


if __name__ == "__main__":
    main()
