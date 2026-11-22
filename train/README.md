# Entraînement et fine-tuning (Personne 3)

## Objectif

Adapter YOLO11n, préentraîné sur COCO, au jeu de données du projet par transfer learning,
puis étudier l'effet des principaux hyperparamètres et conserver les meilleurs poids.

## Fichiers

| Fichier | Rôle |
|---|---|
| `train.py` | Lance les expériences de fine-tuning (une configuration de référence + une variante par hyperparamètre) |
| `compare_runs.py` | Compare les runs : tableau récapitulatif, courbes, copie des meilleurs poids |
| `runs/<expérience>/` | Créé par Ultralytics : `results.csv`, `results.png`, `args.yaml`, `weights/best.pt`, `weights/last.pt` |
| `results_finetuning.csv` | Une ligne par expérience (meilleur epoch sur la validation), transmis à la Personne 4 |
| `figures/` | `comparaison_map.png` et `losses_<meilleur run>.png` |
| `weights/best.pt` | Poids du meilleur run, utilisés pour l'évaluation et la démo |

## Installation et lancement (Google Colab, GPU)

```bash
pip install ultralytics
# test rapide de la chaîne complète (8 images, quelques secondes)
python train/train.py --data coco8.yaml --epochs 3
# toutes les expériences sur le jeu de données du projet
python train/train.py --data data/data.yaml
# ou seulement certaines
python train/train.py --data data/data.yaml --exp baseline freeze_backbone
# comparaison des runs
python train/compare_runs.py
```

Si la mémoire GPU est insuffisante : `--batch 8`.

## Méthode

On part des poids `yolo11n.pt` préentraînés sur COCO. Ultralytics remplace la tête de
détection pour l'adapter au nombre de classes du `data.yaml` (transfer learning, Cours 3
p.90-100). Chaque expérience ne change **qu'un seul** hyperparamètre par rapport à la
référence :

| Expérience | Changement | Justification |
|---|---|---|
| `baseline` | SGD, lr0 = 0.01, 20 epochs, imgsz 640, batch 16, aucune couche gelée, patience 5, seed 0 | Fine-tuning complet (stratégie 2, Cours 3 p.96) |
| `freeze_backbone` | `freeze=10` | Backbone gelé, seuls neck et tête entraînés (stratégie 1, Cours 3 p.95 et p.98 ; Fiche 09) |
| `lr_faible` | `lr0=0.001` | Learning rate faible lors du fine-tuning (Cours 3 p.97 ; Cours 2 p.94) |
| `imgsz_416` | `imgsz=416` | Compromis précision / vitesse (taille des feature maps, Cours 3 p.48-49) |
| `epochs_10` | `epochs=10` | Effet de la durée d'entraînement (Cours 3 p.105-107) |
| `sans_augmentation` | mosaic, flip, translation, échelle et couleurs désactivés | Apport des augmentations (Fiche 04 ; invariance, Cours 3 p.12-13) |

Choix techniques :
- l'optimiseur est fixé à SGD : avec `optimizer="auto"`, Ultralytics choisit lui-même le
  learning rate et ignore `lr0`, ce qui empêcherait de le comparer ;
- les runs sont comparés sur le **jeu de validation** ; le jeu de test reste réservé à
  l'évaluation finale (Cours 3 p.104) ;
- le meilleur epoch de chaque run est celui qu'Ultralytics retient pour `best.pt`
  (critère 0.1 × mAP@0.5 + 0.9 × mAP@0.5:0.95) ;
- 20 epochs suffisent car les poids COCO connaissent déjà les 20 classes VOC ; `patience=5` arrête l'entraînement si la validation ne s'améliore plus (early stopping,
  Fiche 07) ;
- la courbe des losses entraînement / validation du meilleur run permet de détecter un
  éventuel surapprentissage (Cours 3 p.107).

## Résultats

_À compléter après les runs : tableau `results_finetuning.csv`, figures, interprétation._
