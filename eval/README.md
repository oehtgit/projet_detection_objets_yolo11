# Évaluation et comparaison avant / après fine-tuning

## Objectif

Mesurer sur le **jeu de test** (4 300 images, jamais utilisé pour l'entraînement ni pour le
choix des hyperparamètres) la précision et la vitesse du modèle d'origine (YOLO11n COCO) et
du modèle fine-tuné (`train/weights/best.pt`), dans les mêmes conditions, puis les comparer.

## Fichiers

| Fichier | Rôle |
|---|---|
| `evaluate.py` | Évalue chaque modèle sur le test (précision + vitesse), écrit les tableaux et les figures |
| `results/metrics_test.csv` | Une ligne par modèle : P, R, mAP@0.5, mAP@0.5:0.95, temps par image, FPS (GPU Tesla T4) |
| `results/ap_par_classe_test.csv` | AP@0.5 et AP@0.5:0.95 par classe et par modèle |
| `results/metrics_test.json` | Tout, plus les conditions de mesure (GPU T4, versions, réglages) |
| `results/figures/` | `comparaison_globale.png`, `ap_par_classe_<ref>_vs_<modèle>.png`, `vitesse_precision.png` |
| `results/figures/exemples/` | Images du test, modèle d'origine à gauche et fine-tuné à droite (`demo/demo.py --compare`) |
| `evaluation_colab.ipynb` | Notebook Colab : données, évaluation, démo |
| `results/val/<modèle>/` | Sorties d'Ultralytics : courbes PR, matrice de confusion |

## Lancement (Google Colab, GPU T4)

Le notebook `evaluation_colab.ipynb` enchaîne toutes les étapes (dépôt, données, évaluation,
démo). En ligne de commande :

```bash
pip install ultralytics
# données prêtes dans /content/datasets/VOC (voir data/README.md)
# évaluation finale : avant, après, et après à deux tailles d'image plus petites
python eval/evaluate.py --device 0 --models avant=yolo11n.pt apres=train/weights/best.pt \
    apres_416=train/weights/best.pt@416 apres_320=train/weights/best.pt@320
# refaire seulement les figures à partir de results/metrics_test.json
python eval/evaluate.py --replot
```

Le premier modèle de la liste sert de référence pour les écarts par classe. La syntaxe
`poids@taille` évalue le même modèle avec des images plus petites à l'inférence.

## Méthode

**Mêmes conditions pour tous les modèles.** Même jeu de test, même taille d'image (640 px
sauf `--imgsz-auto`), mêmes seuils (`conf=0.001`, `iou=0.7`, les valeurs d'Ultralytics pour
le calcul de la mAP), même GPU, mêmes images pour la vitesse.

**Modèle d'origine (80 classes COCO).** Ses numéros de classes ne correspondent pas à ceux
de `data.yaml`. On réutilise la fonction de la Personne 1 (`data/baseline_zeroshot.py`) : les labels du
test sont renumérotés avec les numéros COCO, et les prédictions des 60 classes COCO absentes
de VOC sont ignorées. On doit retrouver la baseline de `data/stats/baseline_zeroshot.json`
(mAP@0.5:0.95 = 0,626), ce qui sert de contrôle.

**Modèle fine-tuné (20 classes).** Il est évalué directement avec `data.yaml`.

**Vitesse.** Le `val()` d'Ultralytics mesure le temps en batch 32 : ce n'est pas la situation
du temps réel, où les images arrivent une par une. On mesure donc séparément, en **batch 1**,
sur 300 images du test préchargées en mémoire (le disque n'est pas compté), après 20
inférences de warmup (les premières sont plus lentes : allocation mémoire et initialisation
du GPU). On relève :
- `inference_ms` : réseau seul ;
- `total_ms` : prétraitement + réseau + NMS, mesuré à l'horloge ;
- `FPS = 1000 / total_ms` : images par seconde réellement traitables ;
- `total_ms_p95` : 95 % des images sont traitées en moins de ce temps (stabilité).

On considère généralement qu'un système est « temps réel » au-delà de 25 à 30 FPS.

## Démo

`demo/demo.py` applique les modèles sur une vidéo ou la webcam et affiche les FPS en direct
(voir `demo/README.md`) ; avec `--compare`, les deux modèles sont côte à côte.

## Limites

- Un seul run par configuration : un écart de 1 à 2 points peut venir en partie du hasard.
- Les FPS dépendent du matériel : ils se comparent entre eux (même machine), pas à d'autres
  ordinateurs ou GPU.
- La réduction de taille d'image n'est testée qu'à l'inférence ; un modèle entraîné
  directement à 320 px pourrait faire un peu mieux à cette taille.
- Notre découpage 70/10/20 n'est pas le découpage officiel de VOC : les scores ne se
  comparent pas directement à ceux des articles.
