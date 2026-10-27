# Abnormal Event Detector

This repository trains the model behind 3S's **unusual activity detection** — the part of the
system that catches fights, falls, accidents and panic without anyone having written a rule for them.

The model learns what normal activity looks like and flags the moments that deviate from it. Two
properties make it practical for real sites:

- **Cheap to adapt.** Training a new site or a new kind of incident needs only video-level labels —
  "this video is normal", "this one contains an incident" — never frame-by-frame annotation.
- **Stable in the field.** The deployed model does not learn on site, so it cannot drift in an
  uncontrolled way. Operators' true / false verdicts are collected by the application; they are the
  material for periodic, controlled fine-tuning by the engineering team, which lets the product improve
  over time without turning operators into annotators.

This repository is a git submodule of 3S (Smart Surveillance System) and is **not needed at run time**: the
application only uses the exported model, served by Triton as `anomaly_detector`.

---

## What is here

| Path | Purpose |
| :--- | :--- |
| `model/` | The detector: a video backbone pre-trained on Kinetics-400 (S3D or Swin3D variants) and the anomaly scoring head. |
| `data/`, `utils/` | Dataset loading, annotation parsing, video processing and feature extraction. |
| `train/` | Trainers for binary, multi-class and exact-match training, and the loss functions. |
| `evaluation/` | Evaluator and metrics. |
| `*_train.py`, `all_train.py` | One script per dataset and mode; `all_train.py` runs the full experiment matrix. |
| `checkpoints/` | Trained models, one directory per experiment (UCF-Crime, XD-Violence and both combined; binary, multi-class, exact-match). |
| `inference_test_videos.py` | Scores every video in `Test_Videos/` with every checkpoint and writes timeline plots, incident clips and a report per model under `results/`. |
| `triton_anomaly_test.py` | Checks the exported model through the Triton server, exactly as the application calls it. |
| `scripts/`, `kaggle_feature_extraction_script.py`, `organize_s3d.py` | Dataset preparation and large-scale feature extraction. |
| `tests/` | Unit tests (`pytest`). |
| `old_files/` | Earlier prototypes, kept on the development machine for reference; excluded from the repository (`.gitignore`) and not maintained. |

---

## Training

```bash
pip install -r requirements.txt
python all_train.py                    # every experiment in the matrix
python UCF_Dataset_binary_train.py     # or a single dataset / mode
```

Video features are extracted once and cached, so repeated experiments do not process the videos again.
Each experiment stores its trained model under `checkpoints/<experiment>/`.

---

## Checking a model

```bash
python inference_test_videos.py        # all checkpoints on the videos in Test_Videos/
```

```bash
python triton_anomaly_test.py          # the deployed model, through the Triton server
```

---

## How it reaches the product

The application sends Triton clips of **16 frames at 224 × 224 pixels** from every camera, several times
per second and batched across cameras, and receives one score per clip. The score is compared with the
warning and critical thresholds set in the interface; a score above the warning threshold opens an event.

How well the deployed model performs inside the full system — incidents caught, missed and false alarms on
labelled videos — is measured by the application's accuracy harness
(`WebInterface/tests/accuracy_check.py` in the 3S repository). Use it to compare a retrained
model with the current one before deploying it.
