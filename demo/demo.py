"""
Démo de détection d'objets en temps réel.

Fonctionne sur une image, un dossier d'images, une vidéo ou la webcam, avec un ou deux
modèles. Avec deux modèles (--compare), chaque image est affichée côte à côte :
modèle d'origine à gauche, modèle fine-tuné à droite, avec les FPS de chacun.

Les FPS affichés sont ceux réellement obtenus image par image (lecture exclue,
prétraitement + réseau + NMS inclus), lissés sur les dernières images.

Exemples :
    # webcam en temps réel (sur un ordinateur portable, pas sur Colab)
    python demo/demo.py --source 0 --show
    # vidéo, résultat enregistré dans demo/outputs/ (fonctionne sur Colab)
    python demo/demo.py --source video.mp4 --model train/weights/best.pt --device 0
    # comparaison côte à côte avant / après fine-tuning
    python demo/demo.py --source video.mp4 --compare yolo11n.pt train/weights/best.pt --device 0
    # dossier d'images
    python demo/demo.py --source /content/datasets/VOC/images/test --max-frames 20
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "demo" / "outputs"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Les 20 classes VOC (noms COCO) : avec le modèle COCO d'origine, on n'affiche que celles-ci
# pour que la comparaison avec le modèle fine-tuné soit honnête.
VOC_CLASSES = {"airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "dining table", "dog", "horse", "motorcycle", "person", "potted plant", "sheep",
               "couch", "train", "tv"}


class Detector:
    def __init__(self, weights, label, args):
        self.model = YOLO(weights)
        self.label = label
        self.kw = dict(imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)
        if len(self.model.names) == 80 and not args.all_coco:
            self.kw["classes"] = [i for i, n in self.model.names.items() if n in VOC_CLASSES]
        self.times = deque(maxlen=30)  # FPS lissé sur 30 images
        # warmup : les premières inférences sont lentes (allocation mémoire, noyaux GPU)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            self.model.predict(dummy, **self.kw)

    def __call__(self, frame):
        t0 = time.perf_counter()
        result = self.model.predict(frame, **self.kw)[0]
        self.times.append(time.perf_counter() - t0)
        out = result.plot(line_width=2)
        fps = len(self.times) / sum(self.times)
        draw_banner(out, f"{self.label} | {fps:5.1f} FPS | {len(result.boxes)} objets")
        return out


def draw_banner(img, text):
    scale = max(0.5, img.shape[1] / 1000)
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(img, (0, 0), (w + 16, h + 16), (0, 0, 0), -1)
    cv2.putText(img, text, (8, h + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def frames_from(source, max_frames):
    """Renvoie (générateur d'images, fps de la source, nom de sortie, est_une_video)."""
    if source.isdigit():  # webcam
        cap = cv2.VideoCapture(int(source))
        return read_capture(cap, max_frames), 30, "webcam", True
    path = Path(source)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXT)[:max_frames or None]
        return ((cv2.imread(str(p)), p.name) for p in files), None, path.name, False
    if path.suffix.lower() in IMG_EXT:
        return iter([(cv2.imread(str(path)), path.name)]), None, path.stem, False
    cap = cv2.VideoCapture(source)  # fichier vidéo ou URL de flux
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    return read_capture(cap, max_frames), fps, path.stem or "flux", True


def read_capture(cap, max_frames):
    if not cap.isOpened():
        raise SystemExit("Impossible d'ouvrir la source vidéo / webcam.")
    i = 0
    while max_frames == 0 or i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame, f"{i:06d}"
        i += 1
    cap.release()


def side_by_side(images):
    h = min(im.shape[0] for im in images)
    images = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in images]
    return np.hstack(images)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="image, dossier, vidéo, URL ou numéro de webcam (0)")
    p.add_argument("--model", default="yolo11n.pt", help="poids utilisés (un seul modèle)")
    p.add_argument("--compare", nargs=2, metavar=("AVANT", "APRES"),
                   help="deux poids affichés côte à côte (remplace --model)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25, help="seuil de confiance d'affichage")
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", default=None, help="0 pour le GPU, cpu pour le CPU")
    p.add_argument("--show", action="store_true", help="affiche une fenêtre (q pour quitter)")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--max-frames", type=int, default=0, help="0 = toutes les images")
    p.add_argument("--all-coco", action="store_true", help="modèle COCO : afficher les 80 classes")
    args = p.parse_args()

    if args.compare:
        detectors = [Detector(args.compare[0], "avant", args), Detector(args.compare[1], "apres", args)]
    else:
        detectors = [Detector(args.model, Path(args.model).stem, args)]

    frames, src_fps, out_name, is_video = frames_from(args.source, args.max_frames)
    suffix = "_comparaison" if args.compare else ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    writer, n = None, 0

    for frame, frame_name in frames:
        if frame is None:
            continue
        outs = [d(frame) for d in detectors]
        out = outs[0] if len(outs) == 1 else side_by_side(outs)
        n += 1

        if not args.no_save:
            if is_video:
                if writer is None:
                    path = OUT_DIR / f"{out_name}{suffix}.mp4"
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             src_fps, (out.shape[1], out.shape[0]))
                writer.write(out)
            elif Path(args.source).is_file():  # une seule image
                cv2.imwrite(str(OUT_DIR / f"{out_name}{suffix}.jpg"), out)
            else:  # dossier d'images
                (OUT_DIR / f"{out_name}{suffix}").mkdir(exist_ok=True)
                cv2.imwrite(str(OUT_DIR / f"{out_name}{suffix}" / frame_name), out)

        if args.show:
            cv2.imshow("Detection YOLO11 (q pour quitter)", out)
            if cv2.waitKey(1 if is_video else 0) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"{n} images traitées.")
    for d in detectors:
        if d.times:
            print(f"  {d.label:<10} {1 / np.mean(d.times):5.1f} FPS moyens (30 dernières images)")
    if not args.no_save and n:
        print(f"Résultat dans {OUT_DIR}")


if __name__ == "__main__":
    main()
