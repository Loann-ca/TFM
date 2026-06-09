import optuna
import subprocess
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_unet3d_baseline.py")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "optuna_3d")
TRAIN_DEVICE = os.environ.get("TRAIN_DEVICE", "auto")


def _sqlite_url(db_path):
    return "sqlite:///" + db_path.replace("\\", "/")

def objective(trial):
    """Búsqueda expandida de hiperparámetros con Optuna para U-Net 3D."""
    
    # Parámetros expandidos
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    base_channels = trial.suggest_categorical("base_channels", [16, 24, 32, 48])
    
    # Balance de loss: asegurar que sumen ~1.0
    bce_weight = trial.suggest_float("bce_weight", 0.2, 0.8)
    dice_weight = 1.0 - bce_weight
    
    # Regularización
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    
    # Scheduler
    lr_patience = trial.suggest_categorical("lr_patience", [3, 5, 10])
    
    # Batch size acorde a base_channels y VRAM
    if base_channels <= 16:
        batch_size = 4
    elif base_channels <= 32:
        batch_size = 3
    else:
        batch_size = 2
    
    # Nombre único para cada trial
    trial_id = trial.number
    save_dir = os.path.join(RESULTS_DIR, f"trial_{trial_id}_lr_{lr:.1e}_ch_{base_channels}")
    
    cmd = [
        "python",
        TRAIN_SCRIPT,
        "--device", TRAIN_DEVICE,
        "--lr", str(lr),
        "--base_channels", str(base_channels),
        "--batch_size", str(batch_size),
        "--output_dir", OUTPUT_DIR,
        "--save_dir", save_dir,
        "--epochs", "100",
        "--bce_weight", str(bce_weight),
        "--dice_weight", str(dice_weight),
        "--weight_decay", str(weight_decay),
        "--lr_patience", str(lr_patience),
        "--patience", "30",
    ]
    
    print(f"\n[Trial {trial_id}] Running with:")
    print(f"  lr={lr:.1e}, channels={base_channels}, batch={batch_size}")
    print(f"  bce_weight={bce_weight:.2f}, dice_weight={dice_weight:.2f}")
    print(f"  weight_decay={weight_decay:.1e}, lr_patience={lr_patience}")
    
    try:
        subprocess.run(cmd, check=True)
        
        # Leer resultado final
        summary_path = os.path.join(save_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                result = json.load(f)
            # Use foreground Dice (nodule-only) as objective; fall back to overall Dice
            # for old checkpoints that pre-date the fg_dice metric.
            val_dice = result.get("best_val_fg_dice", result.get("best_val_dice", 0.0))
            print(f"[Trial {trial_id}] Best val fg_Dice: {val_dice:.4f}")
            return val_dice
        else:
            print(f"[Trial {trial_id}] Error: summary.json no encontrado")
            return 0.0
    except subprocess.CalledProcessError as e:
        print(f"[Trial {trial_id}] Training failed: {e}")
        return 0.0


# Crear estudio con almacenamiento persistente
os.makedirs(RESULTS_DIR, exist_ok=True)
storage = _sqlite_url(os.path.join(RESULTS_DIR, "optuna_study.db"))
study = optuna.create_study(
    direction="maximize",
    study_name="unet3d_hp_search",
    storage=storage,
    load_if_exists=True
)

print("Starting expanded hyperparameter search...")
print(f"Database: {storage}")
print(f"Train device: {TRAIN_DEVICE}")
study.optimize(objective, n_trials=20, show_progress_bar=True)

print("\n" + "="*60)
print("BEST RESULT:")
print("="*60)
print(f"Best val Dice: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
print(f"Best trial: {study.best_trial.number}")
print("="*60)