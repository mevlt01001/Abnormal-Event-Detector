import os
from model import AnomalyDetector
from data.dataset import get_dataset
from train.trainer import BinaryTrainer, MultiClassTrainer
from evaluation.evaluator import Evaluator

# ─── 1. Experiments Setup ────────────────────────────────────────────────
experiments = [
    # [dataset_key, mode, custom_name]
    ("ucf", "binary", "UCF_binary"),
    ("ucf", "multiclass", "UCF_multiclass"),
    ("xdv", "binary", "XDV_binary"),
    ("xdv", "multiclass", "XDV_multiclass"),
    ("combined", "binary", "UCF_XDV_Combined_binary"),
    ("combined", "multiclass", "UCF_XDV_Combined_multiclass"),
    ("combined", "exact_match", "UCF_XDV_Combined_exact_match"),
]

# ─── 2. Hyperparameters ──────────────────────────────────────────────────
K_TOP = 2
EPOCH = 30
BATCH_SIZE = 48
LR = 0.001
WEIGHT_DECAY = 0.005
EARLY_STOP_PATIENCE = 8
GRAD_CLIP = 1.0

MODEL_HIDDEN_DIMS = (344, 256)
MODEL_DROPOUT_RATES = (0.3, 0.4)

TOLERANCE_SEC = 2.0
PADDING_SEC = 1.0

# ─── 3. Training Loop ────────────────────────────────────────────────
for ds_key, mode, exp_name in experiments:
    print(f"\n{'='*80}\n Starting: {exp_name} (Dataset: {ds_key}, Mode: {mode})\n{'='*80}")
    
    train_dataset = get_dataset(ds_key, split="train", mode=mode, data_dir='data/s3d_features')
    val_dataset   = get_dataset(ds_key, split="test", mode=mode, data_dir='data/s3d_features')
    DS_NAME = f"{exp_name}"

    model = AnomalyDetector(
        backbone_name="s3d",
        num_classes=train_dataset.num_classes,
        hidden_dims=MODEL_HIDDEN_DIMS,
        dropout_rates=MODEL_DROPOUT_RATES,
        class_names=train_dataset.classes,
    )

    if mode == "binary":
        trainer = BinaryTrainer(
            model=model,
            k_top=K_TOP,
            smoothness_weight=0.0001,
            sparsity_weight=0.0001,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            weight_decay=WEIGHT_DECAY,
            early_stop_patience=EARLY_STOP_PATIENCE,
            gradient_clip=GRAD_CLIP,
            save_dir=f"checkpoints/{DS_NAME}",
            tensorboard_dir=f"runs/{DS_NAME}",
        )
    else:  # multiclass or exact_match
        pos_weight = float(max(3.0, train_dataset.num_classes - 1))
        trainer = MultiClassTrainer(
            model=model,
            k_top=2,
            smoothness_weight=0.0001,
            sparsity_weight=0.0001,
            pos_weight=pos_weight,
            iou_threshold=0.30,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            weight_decay=WEIGHT_DECAY,
            early_stop_patience=EARLY_STOP_PATIENCE,
            gradient_clip=GRAD_CLIP,
            save_dir=f"checkpoints/{DS_NAME}",
            tensorboard_dir=f"runs/{DS_NAME}",
        )
    
    result = trainer.fit(train_dataset, val_dataset)

    # ─── 4. Değerlendirme & Production Event Simulation ─────────────────────
    evaluator = Evaluator(
        model=f"checkpoints/{DS_NAME}/best_model.pt",
        dataset_or_loader=val_dataset,
        batch_size=BATCH_SIZE,
    )

    eval_results = evaluator.evaluate(
        mode=val_dataset.mode,
        iou_threshold=0.30,
        k_top_segments=K_TOP,
        topk_accuracies=(1, 2, 3, 4, 5),
        tolerance_sec=TOLERANCE_SEC,
        padding_sec=PADDING_SEC,
        sweep_thresholds=True,
        save_report_dir=f"results/{DS_NAME}",
        print_summary=True,
    )
