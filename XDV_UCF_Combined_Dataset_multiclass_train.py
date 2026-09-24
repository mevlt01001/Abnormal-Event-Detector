from data.dataset import XDV_UCF_Combined_Dataset
from train.trainer import MultiClassTrainer
from evaluation.evaluator import Evaluator
from model import AnomalyDetector

# ─── 1. Dataset ─────────────────────────────────────────────────────────
train_ds = XDV_UCF_Combined_Dataset(
    split="train",
    data_dir="data/s3d_features",
    mode="multiclass",
)

val_ds = XDV_UCF_Combined_Dataset(
    split="test",
    data_dir="data/s3d_features",
    mode="multiclass",
)

DS_NAME = "UCF_XDV_Combined_multiclass"

# ─── 2. Model ────────────────────────────────────────────────────────────────
model = AnomalyDetector(
    backbone_name="s3d",
    num_classes=train_ds.num_classes,
    hidden_dims=(344, 256),
    dropout_rates=(0.3, 0.4),
    class_names=train_ds.classes,
)

# ─── 3. Trainer ──────────────────────────────────────────────────────────────
trainer = MultiClassTrainer(
    model=model,
    k_top=1,                    # Top-K MIL pooling: top-1 segment (max-pooling)
    smoothness_weight=0.0001,   # Temporal smoothness regularization ağırlığı
    sparsity_weight=0.0001,     # Temporal sparsity regularization ağırlığı
    pos_weight=3.0,             # Pozitif sınıf BCE ağırlığı (sınıf dengesizliği için)
    iou_threshold=0.30,         # mAP hesabında kullanılan tIoU eşiği (standart: 0.30)
    # BaseTrainer kwargs ↓
    epochs=10,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=0.01,
    early_stop_patience=2,
    gradient_clip=1.0,          # Gradyan kırpma maksimum normu
    save_dir=f"checkpoints/{DS_NAME}",
    tensorboard_dir=f"runs/{DS_NAME}",
)

result = trainer.fit(train_ds, val_ds)

# ─── 4. Evaluator ────────────────────────────────────────────────────────────
evaluator = Evaluator(
    model=f"checkpoints/{DS_NAME}/best_model.pt",  # eğitilmiş checkpoint
    dataset_or_loader=val_ds,
    batch_size=64,
)

eval_results = evaluator.evaluate(
    mode=val_ds.mode,        # "auto" | "binary" | "multiclass"
    iou_threshold=0.30,       # Threshold sweep ve mAP için tIoU eşiği (standart: 0.30)
    k_top_segments=1,         # Video seviyesi skor: top-1 segment (max-pool)
    topk_accuracies=(1, 3, 5),# Top-K sınıflandırma doğruluğu için K değerleri
    tolerance_sec=.5,        # Segment birleştirme zaman toleransı (saniye)
    padding_sec=1.0,          # Tespit edilen segmentlere eklenen güvenlik payı (saniye)
    sweep_thresholds=True,    # 0.05'ten 0.95'e kadar eşik taraması
    save_report_dir=f"results/{DS_NAME}",  # JSON + MD raporları
    print_summary=True,       # Konsola özet yazdır
)