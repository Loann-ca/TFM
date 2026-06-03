import optuna
import subprocess
import os

def objective(trial):

    lr = trial.suggest_categorical("lr", [1e-3, 3e-4])
    base_channels = trial.suggest_categorical("base_channels", [16, 32])

    batch_size = 4 if base_channels == 16 else 2

    save_dir = f"optuna/lr_{lr}_ch_{base_channels}"

    cmd = [
        "python",
        "train_unet3d_baseline.py",
        "--lr", str(lr),
        "--base_channels", str(base_channels),
        "--batch_size", str(batch_size),
        "--save_dir", save_dir,
        "--epochs", "80",
    ]

    subprocess.run(cmd, check=True)

    # leer resultado final
    import json
    with open(os.path.join(save_dir, "summary.json")) as f:
        result = json.load(f)

    return result["best_val_dice"]


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=4)

print("Best result:")
print(study.best_params)
print(study.best_value)