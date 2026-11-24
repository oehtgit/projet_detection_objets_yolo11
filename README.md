# Détection d'objets en temps réel avec YOLO

Projet 4 du module d'apprentissage profond (M2, UHA), groupe 4 : Amina Menacer,
Matteo Prades, Théo Crétel, Mesrour Lounis.

## Description

Nous partons de **YOLO11n préentraîné sur COCO** (80 classes) et l'adaptons par
**fine-tuning** au jeu **Pascal VOC** (20 classes). Nous évaluons la précision (mAP@0.5,
mAP@0.5:0.95, précision, rappel) et la vitesse d'inférence (ms/image, FPS), comparons le
modèle avant et après adaptation, et testons le système sur des images, des vidéos et la
webcam.

Résultat principal, sur le jeu de test (4 300 images jamais utilisées pour l'entraînement) :

| Modèle | Précision | Rappel | mAP@0.5 | mAP@0.5:0.95 | FPS (GPU T4) |
|---|---|---|---|---|---|
| Avant : YOLO11n COCO, sans adaptation | 0,800 | 0,736 | 0,809 | 0,626 | 76 |
| **Après : YOLO11n fine-tuné** | **0,816** | **0,770** | **0,845** | **0,661** | **85** |

Le fine-tuning apporte **+3,6 points de mAP@0.5 et +3,5 points de mAP@0.5:0.95**, sans
perte de vitesse. Le modèle tourne à 85 images par seconde : le temps réel est atteint.

## Organisation du dépôt

Chaque dossier a son propre README, avec le détail de la méthode et des résultats.

| Dossier | Contenu | Détails |
|---|---|---|
| `data/` | Préparation de Pascal VOC, `data.yaml`, analyse exploratoire, baseline zero-shot | [data/README.md](data/README.md) |
| `train/` | Fine-tuning, étude des hyperparamètres, meilleurs poids `train/weights/best.pt` | [train/README.md](train/README.md) |
| `eval/` | Évaluation finale sur le test, vitesse, comparaison avant / après | [eval/README.md](eval/README.md) |
| `demo/` | Démo sur images, vidéo et webcam, avec comparaison côte à côte | [demo/README.md](demo/README.md) |
| `presentation/` | Présentation de la soutenance (PDF et PowerPoint) | [presentation_YOLO11n.pdf](presentation/presentation_YOLO11n.pdf) |

## Dépendances

- Python 3.8 ou plus récent
- [Ultralytics](https://docs.ultralytics.com) (YOLO11), qui installe PyTorch, OpenCV,
  NumPy, Matplotlib et PyYAML
- `gdown` pour récupérer les données préparées depuis Google Drive
- Un GPU est fortement conseillé pour l'entraînement (Google Colab, GPU T4)

Versions utilisées pour les résultats : Ultralytics 8.4.216, PyTorch 2.14.1, Python 3.13
(Google Colab, GPU Tesla T4).

```bash
pip install ultralytics gdown
```

## Installation et exécution (Google Colab, GPU T4)

```bash
# 1. Code
git clone https://github.com/oehtgit/projet_detection_objets_yolo11
cd projet_detection_objets_yolo11
pip install ultralytics gdown

# 2. Données prêtes (Pascal VOC au format YOLO, 70/10/20, 2,2 Go) dans /content/datasets/VOC
mkdir -p /content/datasets
gdown 1JD_19qsvIUdjzPuSI2k6zhK-OIr2wMbs -O /content/datasets/VOC_pret.zip
cd /content/datasets && unzip -q VOC_pret.zip && rm VOC_pret.zip && cd -
# (ou préparation complète depuis les archives officielles : python data/prepare_data.py --out /content/datasets/VOC)

# 3. Baseline : modèle COCO sans adaptation, sur le test
python data/baseline_zeroshot.py --device 0

# 4. Fine-tuning (environ 4 min par epoch sur T4), puis comparaison des expériences
python train/train.py --data data/data.yaml
python train/compare_runs.py

# 5. Évaluation finale sur le test : précision et vitesse, avant / après
python eval/evaluate.py --device 0 --models avant=yolo11n.pt apres=train/weights/best.pt

# 6. Démo (vidéo ; webcam : --source 0 --show sur un ordinateur)
python demo/demo.py --source ma_video.mp4 --compare yolo11n.pt train/weights/best.pt --device 0
```

Les poids fine-tunés sont fournis dans `train/weights/best.pt` : les étapes 5 et 6 peuvent
être lancées sans refaire l'entraînement.

## Données et préparation

Détails : [data/README.md](data/README.md).

- **Pascal VOC 2007 + 2012** : 21 503 images annotées, 20 classes (personnes, véhicules,
  animaux, objets d'intérieur). Les 20 classes existent toutes dans COCO, ce qui permet de
  mesurer le modèle avant adaptation.
- **Conversion** des annotations XML au format YOLO (`classe x_centre y_centre largeur
  hauteur`, normalisés), objets « difficult » écartés. Les classes portent leur nom COCO
  (`aeroplane` → `airplane`…).
- **Découpage stratifié 70 / 10 / 20** (graine 0) : 15 052 images d'entraînement, 2 151 de
  validation (choix des hyperparamètres), 4 300 de test (score final uniquement).
- **Analyse** : classes déséquilibrées (`person` = 34 % des objets), surtout de grands objets
  (6 % de petits objets, surtout bottle et potted plant).

## Méthode

1. **Baseline** ([data/README.md](data/README.md)) : YOLO11n COCO évalué tel quel. Ses
   numéros de classes (80) diffèrent de ceux de VOC (20) : les labels sont renumérotés avec
   les numéros COCO avant l'évaluation.
2. **Fine-tuning par transfer learning** ([train/README.md](train/README.md)) : on repart
   des poids COCO ; Ultralytics reprend les poids COCO des 20 classes dans la nouvelle tête
   de détection. Six expériences, un seul hyperparamètre changé à la fois, comparées sur la
   validation : learning rate, gel du backbone, taille d'image, augmentations, durée
   d'entraînement.
3. **Évaluation finale** ([eval/README.md](eval/README.md)) : avant et après sur le même jeu
   de test, avec les mêmes réglages ; vitesse mesurée image par image (batch 1), comme en
   temps réel.
4. **Démo** ([demo/README.md](demo/README.md)) : détection sur vidéo ou webcam avec
   affichage des FPS.

## Résultats et interprétation

**Étude des hyperparamètres** (validation, mAP@0.5:0.95 ; modèle de départ : 0,634) :

| Expérience | mAP@0.5:0.95 | Enseignement |
|---|---|---|
| lr0 = 0,01 | 0,484 | Un learning rate trop fort détruit les poids préentraînés (chute à 0,17 à l'epoch 3) |
| lr0 = 0,001 | 0,655 | Un learning rate faible améliore le modèle (Cours 3 p.97) |
| Backbone gelé | 0,658 | Aussi bien que le fine-tuning complet : les features COCO conviennent à VOC |
| Images de 416 px | 0,622 | Moins précis, surtout sur les petits objets |
| Sans augmentation | 0,657 | Puis surapprentissage : la loss d'entraînement baisse, la mAP de validation aussi |
| **20 epochs complètes (lr0 = 0,001)** | **0,666** | **Meilleur modèle** : la mAP remonte après le warmup et monte encore à l'epoch 20 |

**Avant / après sur le test** (tableau en haut de page) :

- le gain mesuré sur la validation (+3,2 points) se retrouve sur le test (+3,5 points) : il
  se généralise à des images inédites ;
- 17 classes sur 20 progressent, surtout les plus faibles au départ (dining table +10,7
  points, chair +8,0, car +8,0). Le modèle connaissait déjà ces objets : il s'adapte surtout
  aux conventions d'annotation de Pascal VOC ;
- la vitesse ne baisse pas (85 FPS après, 76 avant) : seule la dernière couche passe de 80 à 20
  classes. Sur GPU, réduire la taille d'image ne fait pas gagner de vitesse mais fait perdre
  de la précision : on garde 640 px.

**Limites** : un seul run par configuration (un écart de 1 à 2 points peut en partie venir du
hasard) ; découpage 70/10/20 différent du découpage officiel de VOC, donc scores non
comparables directement à ceux des articles ; FPS propres au GPU utilisé.

## Contributions individuelles

| Membre | Rôle | Tâches réalisées |
|---|---|---|
| **Amina Menacer** | Données et annotation | Choix et justification de Pascal VOC ; téléchargement, conversion au format YOLO, découpage stratifié 70/10/20 ; contrôle qualité ; analyse exploratoire et figures ; `data.yaml` ; baseline zero-shot sur le test ; partie « Données » |
| **Matteo Prades** | Modèle préentraîné et baseline | Prise en main de YOLO11n ; évaluation du modèle préentraîné sur la validation et le test (référence utilisée pour les comparaisons d'entraînement) ; mesure de vitesse sur GPU ; notebook Colab de la baseline ; partie « Modèle et baseline » |
| **Théo Crétel** | Fine-tuning et optimisation | Code d'entraînement et de comparaison (`train/`) ; six expériences de fine-tuning (learning rate, gel du backbone, taille d'image, augmentations, durée) ; suivi des courbes de loss et de mAP ; choix et sauvegarde des meilleurs poids ; partie « Entraînement et fine-tuning » |
| **Mesrour Lounis** | Évaluation, démo et intégration | Évaluation finale sur le test (mAP, précision, rappel) ; mesure de la vitesse (ms/image, FPS) ; comparaison avant / après (tableaux, figures, exemples) ; démo sur images, vidéo et webcam ; coordination du rapport et de la présentation |
