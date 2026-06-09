import optuna
import subprocess
import os
import json
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_unet2d_baseline.py")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "optuna_2d")


def _sqlite_url(db_path):
    return "sqlite:///" + db_path.replace("\\", "/")

def objective(trial):
    """Búsqueda expandida de hiperparámetros con Optuna para U-Net 2D."""
    
    # Parámetros expandidos
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    base_channels = trial.suggest_categorical("base_channels", [16, 24, 32, 48])
    
    # Patch size importante en 2D para receptive field
    patch_size = trial.suggest_categorical("patch_size", [64, 96, 128, 160])
    
    # Balance de loss: asegurar que sumen ~1.0
    bce_weight = trial.suggest_float("bce_weight", 0.2, 0.8)
    dice_weight = 1.0 - bce_weight
    
    # Regularización
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    
    # Scheduler
    lr_patience = trial.suggest_categorical("lr_patience", [3, 5, 10])
    
    # Batch size acorde a base_channels y VRAM
    # 2D puede usar batch más grande que 3D
    if base_channels <= 16:
        batch_size = 8
    elif base_channels <= 32:
        batch_size = 6
    else:
        batch_size = 4
    
    # Nombre único para cada trial
    trial_id = trial.number
    save_dir = os.path.join(
        RESULTS_DIR,
        f"trial_{trial_id}_lr_{lr:.1e}_ch_{base_channels}_ps_{patch_size}",
    )
    
    cmd = [
        sys.executable,
        TRAIN_SCRIPT,
        "--lr", str(lr),
        "--base_channels", str(base_channels),
        "--patch_size", str(patch_size),
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
    
    print(f"\n[Trial {trial_id}] Running 2D U-Net with:")
    print(f"  lr={lr:.1e}, channels={base_channels}, patch_size={patch_size}")
    print(f"  batch={batch_size}")
    print(f"  bce_weight={bce_weight:.2f}, dice_weight={dice_weight:.2f}")
    print(f"  weight_decay={weight_decay:.1e}, lr_patience={lr_patience}")
    
    try:
        subprocess.run(cmd, check=True)
        
        # Leer resultado final
        summary_path = os.path.join(save_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                result = json.load(f)
            val_dice = result.get("best_val_dice", 0.0)
            print(f"[Trial {trial_id}] Best val Dice: {val_dice:.4f}")
            return val_dice
        else:
            print(f"[Trial {trial_id}] Error: summary.json no encontrado en {save_dir}")
            return 0.0
    except subprocess.CalledProcessError as e:
        print(f"[Trial {trial_id}] Training failed: {e}")
        return 0.0


# Crear estudio con almacenamiento persistente
os.makedirs(RESULTS_DIR, exist_ok=True)
storage = _sqlite_url(os.path.join(RESULTS_DIR, "optuna_study_2d.db"))
study = optuna.create_study(
    direction="maximize",
    study_name="unet2d_hp_search",
    storage=storage,
    load_if_exists=True
)

print("="*60)
print("Starting expanded 2D U-Net hyperparameter search...")
print(f"Database: {storage}")
print(f"Trials: 20 (may resume from existing if interrupted)")
print("="*60)

study.optimize(objective, n_trials=20, show_progress_bar=True)

print("\n" + "="*60)
print("BEST 2D U-NET RESULT:")
print("="*60)
print(f"Best val Dice: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
print(f"Best trial: {study.best_trial.number}")
print("="*60)
