# Données (Amina MENACER)

## Objectif

Préparer le jeu de données Pascal VOC (2007 + 2012) pour YOLO : téléchargement, conversion des
annotations au format YOLO, découpage 70 / 10 / 20 en train / validation / test, contrôle
qualité, analyse exploratoire, et mesure de la performance du modèle COCO **avant** adaptation
(baseline zero-shot), qui sert de point de départ au fine-tuning (`train/`).

## Fichiers

| Fichier | Rôle |
|---|---|
| `prepare_data.py` | Téléchargement, conversion XML → YOLO, découpage 70/10/20 stratifié, écriture de `data.yaml`, zip optionnel |
| `data.yaml` | Description du jeu pour YOLO : chemins train / val / test et noms des 20 classes. **C'est le fichier passé à `train/train.py`** |
| `data_eda.py` | Contrôle qualité et statistiques (tableaux dans `stats/`) |
| `data_viz.py` | Figures de l'analyse exploratoire (dans `figures/`) |
| `baseline_zeroshot.py` | mAP de YOLO11n pré-entraîné sur COCO, sans adaptation, sur le test |
| `preparation_donnees_colab.ipynb` | Notebook Colab qui lance toute la préparation et copie le zip sur Google Drive |
| `stats/` | `resume_splits.csv`, `objets_par_classe.csv`, `tailles_*.csv`, `resume_donnees.json`, `baseline_zeroshot.json` |
| `figures/` | 8 figures : exemples annotés, conversion VOC → YOLO, découpage, classes, tailles d'objets et d'images |

## Installation et lancement (Google Colab, CPU suffisant)

```bash
pip install ultralytics
# préparation complète : télécharge VOC (~2,8 Go), convertit, découpe -> /content/datasets/VOC + data/data.yaml
python data/prepare_data.py --out /content/datasets/VOC
# ou récupération du zip déjà prêt sur Google Drive (repli automatique sur la préparation complète)
python data/prepare_data.py --out /content/datasets/VOC --drive-id <ID_DU_ZIP>
# contrôle qualité, statistiques et figures
python data/data_eda.py
python data/data_viz.py
# baseline : modèle COCO sans adaptation sur le test (GPU conseillé : --device 0)
python data/baseline_zeroshot.py --device 0
```

Puis l'entraînement : `python train/train.py --data data/data.yaml`.

Important : les données doivent rester sur le disque local de Colab (`/content`), jamais
directement sur Google Drive (des milliers de petits fichiers y sont très lents).

## Méthode

**Données.** 21 503 images annotées pour la détection (listes officielles `ImageSets/Main` de
VOC2007 et VOC2012), 20 classes (personnes, véhicules, animaux, objets d'intérieur). Les 20
classes existent toutes dans COCO, ce qui permet de mesurer la performance avant adaptation.

**Conversion.** Chaque boîte VOC (deux coins en pixels, fichier XML) devient une ligne YOLO
`classe x_centre y_centre largeur hauteur`, normalisée entre 0 et 1 :
`x_centre = ((xmin + xmax)/2 − 1)/W`, `largeur = (xmax − xmin)/W` (idem en y avec H). Les objets
marqués `difficult` sont écartés (convention officielle VOC) : 52 090 objets gardés, 10 109
écartés. Vérification : conversion identique à celle d'Ultralytics sur les 21 503 images.

**Noms des classes.** Six classes sont écrites avec l'orthographe COCO (`aeroplane` → `airplane`,
`sofa` → `couch`, `tvmonitor` → `tv`…). Ultralytics associe les classes par leur nom au
fine-tuning : les 20 classes réutilisent ainsi la tête de classification COCO.

**Découpage 70 / 10 / 20 stratifié** (graine 0, donc reproductible) : chaque image est rangée
dans le groupe de la classe la plus rare qu'elle contient, puis chaque groupe est découpé
70 / 10 / 20. Les trois ensembles ont ainsi les mêmes proportions de classes (écart < 1 point
pour chaque classe).

| Ensemble | Part | Images | Objets | Rôle |
|---|---|---|---|---|
| train | 70 % | 15 052 | 36 545 | apprentissage |
| val | 10 % | 2 151 | 5 132 | early stopping, choix de `best.pt`, comparaison des runs |
| test | 20 % | 4 300 | 10 413 | score final uniquement |

Ce découpage n'est pas le découpage officiel de VOC (test = VOC2007 test) : nos scores se
comparent entre eux (avant / après fine-tuning, même test), pas directement aux articles.
L'option `--split officiel` reproduit le découpage officiel si besoin.

## Résultats

**Contrôle qualité** : aucun label manquant, aucune coordonnée hors de [0, 1], aucune classe
invalide, aucune image sans objet.

**Analyse exploratoire** (`figures/`, `stats/`) :
- déséquilibre des classes : `person` = 34 % des objets du train (12 451), `dining table` = 721,
  soit 17,3 fois moins ;
- 46 % des images ne contiennent qu'un objet (moyenne 2,4, maximum 39) ;
- surtout de grands objets : 64 % grands (> 96×96 px), 6 % petits (< 32×32 px) ; classes avec le
  plus de petits objets : bottle (20 %), car (12 %), sheep (11 %), potted plant (10 %) ;
- images de 500 px de côté au plus : YOLO (640 px) les agrandit.

**Baseline zero-shot** (YOLO11n COCO, sans adaptation, test 70/10/20, imgsz 640) :

| mAP50 | mAP50-95 | Précision | Rappel |
|---|---|---|---|
| 0,809 | 0,626 | 0,801 | 0,736 |

Classes les plus faibles (AP50) : potted plant 0,53, chair 0,55, dining table 0,66, boat 0,67.
C'est le point de départ que le fine-tuning doit dépasser, sur le même jeu de test.
