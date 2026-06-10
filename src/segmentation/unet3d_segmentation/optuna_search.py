import optuna
import torch
from train_unet3d_baseline import train_model


BASE_CONFIG = {
    "data_dir": r"D:\TFM\output\preprocessed", 
    "base_channels": 32,
    "lr": 3e-4,
    "epochs": 40,
}

def objective(trial):

    # -------------------
    # SEARCH SPACE
    # -------------------
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])

    dice_weight = trial.suggest_float("dice_weight", 0.6, 0.9)
    bce_weight = 1.0 - dice_weight

    # -------------------
    # CONFIG FINAL
    # -------------------
    config = dict(BASE_CONFIG)
    config.update({
        "batch_size": batch_size,
        "dice_weight": dice_weight,
        "bce_weight": bce_weight,
    })

    # -------------------
    # TRAIN + EVAL
    # -------------------
    val_dice = train_model(config)
    return val_dice


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("BEST RESULT:")
print(study.best_params)
print(study.best_value)