"""
Fine-tuning de YOLO11n sur le jeu de données du projet (Personne 3).

Principe (Cours 3, p.90-100 : transfer learning) :
on part des poids de YOLO11n préentraînés sur COCO ("yolo11n.pt"). Ultralytics remplace
automatiquement la tête de détection si le nombre de classes du data.yaml est différent
de 80, puis on ré-entraîne le réseau sur nos données.

Chaque expérience ne modifie qu'UN seul hyperparamètre par rapport à la configuration
de référence (BASE), pour pouvoir attribuer la différence de résultat à ce paramètre.
Les comparaisons se font sur le jeu de validation (Cours 3, p.104) ; le jeu de test
reste réservé à l'évaluation finale (Personne 4).

Exemples :
    python train/train.py --data coco8.yaml --epochs 3            # test rapide de la chaîne
    python train/train.py --data data/data.yaml                    # toutes les expériences
    python train/train.py --data data/data.yaml --exp baseline lr_faible
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

RUNS_DIR = Path(__file__).resolve().parent / "runs"

# Configuration de référence.
BASE = {
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    # Optimiseur fixé explicitement : avec optimizer="auto" (défaut), Ultralytics choisit
    # lui-même l'optimiseur et le learning rate et ignore lr0 (Fiche 05 : SGD).
    "optimizer": "SGD",
    "lr0": 0.01,
    # Stratégie 2 du cours (fine-tuning) : aucune couche gelée (Cours 3, p.96).
    "freeze": None,
    # Early stopping : arrêt si la validation ne s'améliore plus pendant 15 epochs (Fiche 07).
    "patience": 15,
    # Graine fixe pour que les runs soient comparables et reproductibles.
    "seed": 0,
}

# Une expérience = les paramètres qui changent par rapport à BASE.
EXPERIMENTS = {
    "baseline": {},
    # Stratégie 1 du cours (feature extraction) : backbone (10 premières couches) gelé,
    # seuls le neck et la tête sont entraînés (Cours 3, p.95 et p.98 ; Fiche 09).
    "freeze_backbone": {"freeze": 10},
    # Learning rate 10x plus faible : "règle d'or" du fine-tuning (Cours 3, p.97 ;
    # Cours 2, p.94).
    "lr_faible": {"lr0": 0.001},
    # Images plus petites : feature maps plus petites (Cours 3, p.48-49), donc plus rapide
    # mais petits objets moins bien vus.
    "imgsz_416": {"imgsz": 416},
    # Moins d'epochs : effet de la durée d'entraînement (Cours 3, p.105-107).
    "epochs_20": {"epochs": 20},
    # Sans augmentation de données : mesure l'apport des augmentations (Fiche 04 ;
    # invariance par translation, Cours 3 p.12-13 et p.27).
    "sans_augmentation": {
        "mosaic": 0.0,
        "fliplr": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning de YOLO11n")
    parser.add_argument("--data", required=True, help="chemin du data.yaml (ex. data/data.yaml)")
    parser.add_argument("--model", default="yolo11n.pt", help="poids préentraînés de départ")
    parser.add_argument(
        "--exp",
        nargs="+",
        default=list(EXPERIMENTS),
        choices=list(EXPERIMENTS),
        help="expériences à lancer (par défaut : toutes)",
    )
    parser.add_argument("--epochs", type=int, help="force le nombre d'epochs (test rapide)")
    parser.add_argument("--batch", type=int, help="force la taille de batch (si mémoire GPU insuffisante)")
    parser.add_argument("--device", default=None, help="ex. 0 pour le GPU, cpu sinon")
    return parser.parse_args()


def main():
    args = parse_args()

    for name in args.exp:
        params = {**BASE, **EXPERIMENTS[name]}
        if args.epochs is not None:
            params["epochs"] = args.epochs
        if args.batch is not None:
            params["batch"] = args.batch

        print(f"\n=== Expérience : {name} ===")
        print({k: v for k, v in params.items()})

        # On recharge les poids préentraînés à chaque expérience : chaque run part
        # du même point de départ.
        model = YOLO(args.model)
        model.train(
            data=args.data,
            project=str(RUNS_DIR),
            name=name,
            exist_ok=True,
            device=args.device,
            **params,
        )
        # Ultralytics sauvegarde dans train/runs/<name>/ :
        #   results.csv (losses et métriques par epoch), args.yaml (hyperparamètres),
        #   weights/best.pt (meilleur epoch sur la validation) et weights/last.pt.


if __name__ == "__main__":
    main()
