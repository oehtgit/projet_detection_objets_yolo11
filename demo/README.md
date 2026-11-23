# Démo de détection en temps réel

`demo.py` applique un modèle sur une image, un dossier d'images, une vidéo ou la webcam, et
affiche sur chaque image les boîtes détectées, le nombre d'objets et les FPS réellement
obtenus. Avec `--compare`, les deux modèles sont affichés côte à côte (avant à gauche,
après à droite).

## Lancement

```bash
pip install ultralytics
# webcam (sur un ordinateur, pas sur Colab : Colab n'a pas accès à la webcam via OpenCV)
python demo/demo.py --source 0 --show --model train/weights/best.pt
# vidéo : le résultat est enregistré dans demo/outputs/<nom>.mp4 (fonctionne sur Colab)
python demo/demo.py --source ma_video.mp4 --model train/weights/best.pt --device 0
# comparaison côte à côte avant / après
python demo/demo.py --source ma_video.mp4 --compare yolo11n.pt train/weights/best.pt --device 0
# quelques images du test
python demo/demo.py --source /content/datasets/VOC/images/test --max-frames 20 \
    --compare yolo11n.pt train/weights/best.pt --device 0
```

Options utiles : `--conf 0.25` (seuil d'affichage), `--imgsz 640`, `--max-frames N`,
`--no-save`, `--all-coco` (avec le modèle COCO, afficher les 80 classes au lieu des 20 de VOC).

## Choix

- Avec le modèle COCO d'origine, seules les 20 classes VOC sont affichées par défaut, pour
  comparer honnêtement avec le modèle fine-tuné qui ne connaît que celles-ci.
- Les FPS sont lissés sur les 30 dernières images ; la lecture de la vidéo n'est pas comptée,
  le prétraitement, le réseau et la NMS le sont.
- 5 inférences de warmup sont faites avant la première image, sinon les premiers FPS
  affichés sont très bas.
