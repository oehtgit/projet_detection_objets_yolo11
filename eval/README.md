# Évaluation du modèle préentraîné (Mattéo PRADES)

## Protocole

YOLO11n préentraîné sur COCO est évalué sans fine-tuning sur le Pascal VOC du
projet, décrit par `data/data.yaml`, avec une taille d'image de **640**.
Les résultats de validation permettent de situer les expériences d'entraînement ;
ceux de test servent à la comparaison finale.

Les identifiants des 20 classes VOC diffèrent de ceux des 80 classes COCO.
Le script `data/baseline_zeroshot.py` utilise le dataset décrit par le YAML
d'origine, mais crée un YAML intermédiaire et réindexe ses annotations en
identifiants COCO. Les images et les boîtes restent les mêmes. Une validation
directe des poids COCO sur le YAML VOC interpréterait mal les identifiants.
Les poids fine-tunés, eux, doivent être évalués avec le YAML VOC d'origine.

## Résultats avant fine-tuning

Mesures réalisées localement sur la RTX 4060 Laptop GPU avec le dataset 70/10/20,
`yolo11n.pt`, 640 pixels et `batch=16` :

| Split | Images | Objets | Précision | Rappel | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| val | 2 151 | 5 132 | 0,7846 | 0,7558 | 0,8073 | 0,6340 |
| test | 4 300 | 10 413 | 0,8015 | 0,7359 | 0,8093 | 0,6262 |

Les valeurs détaillées, dont l'AP50 et l'AP50-95 des 20 classes, sont dans
`resultats/val/baseline_metrics.json` et `resultats/test/baseline_metrics.json`.
Sur test, les AP50 les plus faibles concernent `potted plant` (0,5265),
`chair` (0,5498) et `dining table` (0,6587).

Ultralytics a mesuré sur test **2,28 ms d'inférence par image**, moyenne calculée
avec des lots de 16, plus 1,25 ms de prétraitement et 1,37 ms de post-traitement.
Ces chiffres ne comprennent pas toute la chaîne de lecture et ne sont pas une
mesure de latence ou de FPS pour une vidéo traitée image par image.

L'ancienne baseline de `data/stats/baseline_zeroshot.json` donne des scores test
proches ; elle ne précise cependant pas dans son JSON le split ni les versions.
Les scores dépendent du découpage et ne sont pas directement comparables à ceux
obtenus avec le découpage officiel VOC. Un éventuel recouvrement entre COCO et
VOC limite aussi leur interprétation comme scores sur des images inédites.

### Conditions de mesure

- Python 3.13.14, Ultralytics 8.4.216, PyTorch 2.9.1+cu128, CUDA 12.8.
- GPU NVIDIA GeForce RTX 4060 Laptop GPU, 8 Go ; validation en lots de 16.
- Poids `yolo11n.pt`, SHA-256 :
  `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- Archive récupérée `VOC_pret_70_10_20.zip`, SHA-256 :
  `a049d8f1fa21caa3e95da2634040ca1134bcc99cfac2efd99ff28c345301ae07`.
  Le nombre d'images et d'annotations correspond aux statistiques du projet.
- Les configurations partagée et locale ont été vérifiées identiques pour les
  splits et les noms de classes ; seul le chemin racine du dataset diffère.

## Reproduire sur Windows

Installer les données comme indiqué dans `data/README.md`. Copier
`data/data.yaml` vers `data/data.local.yaml`, puis remplacer uniquement `path`
par le chemin Windows du dossier `VOC` extrait. Ce fichier local est ignoré par
Git ; ne pas publier un chemin propre à sa machine.

Avec un environnement virtuel contenant Ultralytics, PyYAML et PyTorch avec
CUDA, depuis la racine du dépôt :

```powershell
.\.venv\Scripts\python.exe data\baseline_zeroshot.py --model yolo11n.pt --data data\data.local.yaml --split val --imgsz 640 --batch 16 --device 0 --out eval\resultats\val
.\.venv\Scripts\python.exe data\baseline_zeroshot.py --model yolo11n.pt --data data\data.local.yaml --split test --imgsz 640 --batch 16 --device 0 --out eval\resultats\test
```

Les JSON de résultats sont écrasés par une nouvelle exécution. Le script écrit
aussi des figures et des exemples d'images dans `runs/val/baseline_zeroshot/` ;
ce dossier est ignoré par Git et réutilisé d'un lancement à l'autre.

Le script crée un dossier voisin `VOC_coco_ids` et réutilise les images qui y
existent déjà. Pour éviter de mélanger des versions de données, utiliser un
emplacement de dataset neuf lorsque le dataset change. Vérifier également que
le split demandé est présent : le script se rabat sur `val` s'il est absent.

## Reproduire sur Colab

Ouvrir `eval/baseline_colab.ipynb` dans Google Colab, activer un GPU et exécuter
les cellules dans l'ordre. Le notebook récupère le dépôt et l'archive de données,
vérifie leurs conditions d'utilisation et évalue le split test à 640 pixels.
Il permet de télécharger les métriques et les conditions de mesure. Le runtime
Colab est temporaire ; les scores et la vitesse obtenus dépendent de son matériel
et des versions installées.

## Comparaison après fine-tuning

Lorsque `train/weights/best.pt` sera disponible, la personne 4 pourra l'évaluer
sur **test**, avec la même configuration VOC, les mêmes images, 640 pixels et
les mêmes paramètres de validation. Les métriques sur test doivent être
comparées à la ligne test ci-dessus ; celles sur val, à la ligne val.
`train/results_finetuning.csv` contient des résultats de validation, pas de test.

Pour comparer les temps, conserver aussi le même matériel, les mêmes versions,
le même batch et les mêmes réglages de précision. La comparaison après
fine-tuning reste à faire : les poids entraînés n'étaient pas encore disponibles.
