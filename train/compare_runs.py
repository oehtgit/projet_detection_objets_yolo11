"""
Comparaison des expériences de fine-tuning (Personne 3).

Lit train/runs/<expérience>/results.csv et args.yaml produits par train.py, puis :
  1. écrit train/results_finetuning.csv : une ligne par expérience, meilleur epoch
     sur la validation (format commun transmis à la Personne 4) ;
  2. trace la mAP@0.5:0.95 de validation par epoch pour toutes les expériences
     (train/figures/comparaison_map.png) ;
  3. trace les losses d'entraînement et de validation du meilleur run pour vérifier
     l'absence de surapprentissage (Cours 3, p.107) (train/figures/losses_<run>.png) ;
  4. copie les poids du meilleur run dans train/weights/best.pt (pour la Personne 4).

Utilisation :
    python train/compare_runs.py
"""

import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas d'affichage nécessaire (Colab / terminal)
import matplotlib.pyplot as plt
import yaml

TRAIN_DIR = Path(__file__).resolve().parent
RUNS_DIR = TRAIN_DIR / "runs"
FIG_DIR = TRAIN_DIR / "figures"
WEIGHTS_DIR = TRAIN_DIR / "weights"
SUMMARY_CSV = TRAIN_DIR / "results_finetuning.csv"

MAP50 = "metrics/mAP50(B)"
MAP5095 = "metrics/mAP50-95(B)"
PRECISION = "metrics/precision(B)"
RECALL = "metrics/recall(B)"

# Couleurs fixes par ordre d'expérience (palette catégorielle accessible).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def read_results(csv_path):
    """Lit results.csv d'Ultralytics et renvoie {colonne: [valeurs float]}."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    # Certaines versions d'Ultralytics ajoutent des espaces dans les noms de colonnes.
    rows = [{k.strip(): v for k, v in row.items()} for row in rows]
    return {col: [float(r[col]) for r in rows] for col in rows[0]}


def fitness(res, i):
    """Critère utilisé par Ultralytics pour choisir best.pt : 0.1*mAP50 + 0.9*mAP50-95."""
    return 0.1 * res[MAP50][i] + 0.9 * res[MAP5095][i]


def load_runs():
    runs = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        csv_path = run_dir / "results.csv"
        if not csv_path.exists():
            continue
        res = read_results(csv_path)
        with open(run_dir / "args.yaml") as f:
            args = yaml.safe_load(f)
        best = max(range(len(res["epoch"])), key=lambda i: fitness(res, i))
        runs.append({"name": run_dir.name, "dir": run_dir, "res": res, "args": args, "best": best})
    return runs


def write_summary(runs):
    fields = [
        "experience", "freeze", "optimizer", "lr0", "imgsz", "batch", "epochs_prevus",
        "epochs_effectues", "meilleur_epoch", "precision", "rappel", "mAP50", "mAP50-95",
    ]
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in runs:
            res, b, a = r["res"], r["best"], r["args"]
            writer.writerow({
                "experience": r["name"],
                "freeze": a.get("freeze") if a.get("freeze") is not None else "aucun",
                "optimizer": a.get("optimizer"),
                "lr0": a.get("lr0"),
                "imgsz": a.get("imgsz"),
                "batch": a.get("batch"),
                "epochs_prevus": a.get("epochs"),
                "epochs_effectues": len(res["epoch"]),
                "meilleur_epoch": int(res["epoch"][b]),
                "precision": round(res[PRECISION][b], 4),
                "rappel": round(res[RECALL][b], 4),
                "mAP50": round(res[MAP50][b], 4),
                "mAP50-95": round(res[MAP5095][b], 4),
            })


def plot_map(runs):
    fig, ax = plt.subplots(figsize=(8, 5))
    for color, r in zip(COLORS, runs):
        ax.plot(r["res"]["epoch"], r["res"][MAP5095], color=color, linewidth=2, label=r["name"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0.5:0.95 (validation)")
    ax.set_title("Comparaison des expériences de fine-tuning")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparaison_map.png", dpi=150)
    plt.close(fig)


def plot_losses(run):
    """Losses train vs validation sur le même graphique (Cours 3, p.107)."""
    res = run["res"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, loss in zip(axes, ["box_loss", "cls_loss", "dfl_loss"]):
        ax.plot(res["epoch"], res["train/" + loss], color=COLORS[0], linewidth=2, label="entraînement")
        ax.plot(res["epoch"], res["val/" + loss], color=COLORS[1], linewidth=2, label="validation")
        ax.set_title(loss)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Losses du meilleur run : {run['name']}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"losses_{run['name']}.png", dpi=150)
    plt.close(fig)


def main():
    runs = load_runs()
    if not runs:
        raise SystemExit(f"Aucun run trouvé dans {RUNS_DIR}. Lancer d'abord train/train.py.")

    FIG_DIR.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)

    write_summary(runs)
    plot_map(runs)

    best_run = max(runs, key=lambda r: fitness(r["res"], r["best"]))
    plot_losses(best_run)
    shutil.copy(best_run["dir"] / "weights" / "best.pt", WEIGHTS_DIR / "best.pt")

    print(f"{'expérience':<20}{'mAP50':>8}{'mAP50-95':>10}{'meilleur epoch':>16}")
    for r in runs:
        b = r["best"]
        print(f"{r['name']:<20}{r['res'][MAP50][b]:>8.3f}{r['res'][MAP5095][b]:>10.3f}"
              f"{int(r['res']['epoch'][b]):>16}")
    print(f"\nMeilleur run : {best_run['name']} -> poids copiés dans {WEIGHTS_DIR / 'best.pt'}")
    print(f"Tableau : {SUMMARY_CSV}")
    print(f"Figures : {FIG_DIR}")


if __name__ == "__main__":
    main()
