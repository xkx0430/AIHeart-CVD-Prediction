# ❤️ AIHeart-CVD-Prediction

AIHeart-CVD-Prediction is a survival modeling toolkit for cardiovascular outcomes in Chinese adults. It provides a practical workflow from tabular clinical data to risk prediction, including PyTorch training/inference and ONNX deployment.

## 🎯 Prediction Targets & Users

AIHeart-CVD-Prediction is designed to predict CVD, MI, and STROKE risk, and is intended for clinical AI researchers, epidemiology and survival-analysis teams, data scientists building tabular deep-learning pipelines, and so on.

## ✅ What You Get

- Fast ONNXRuntime inference for deployment (`predict.py`)
- A training pipeline with validation and early stopping (`train.py`)
- Sample-level risk inference from PyTorch checkpoints (`infer.py`)
- ONNX export with optional PyTorch-vs-ONNX parity check (`onnx.py`)
- Risk output files with `event,time,risk` columns

## 🧩 Model Types

- `Simplified` models use 15 predictors, which is more favorable for broad clinical translation and deployment. 
- `Full` models use 22 predictors and provides stronger disease-risk prediction performance.

## 🌐 Web Portal

You can also use the online portal: [Cardiovascular Disease Risk Assessment](http://cvd_test.myweihp.win:30212/).

The website provides a visual submission workflow and a file-upload interface for obtaining prediction results.

## 🚀 Quick Start

### 1) Install dependencies

Recommended Python: `3.10+`

```bash
pip install -r requirements.txt
```

### 2) Run ONNXRuntime inference (Recommend)

```bash
python predict.py \
  --onnx_file ckpts/CVD_Men_Simplified/transformer_model_t0.onnx \
  --data_file datasets/example_cvd_dataset.csv \
  --model_type Simplified \
  --sex 0 \
  --t0 10 \
  --output_csv ckpts/CVD_Men_Simplified/predict_results.csv
```

### 3) Train a model

```bash
python train.py \
  --data_file datasets/example_cvd_dataset.csv \
  --model_type Simplified \
  --sex 0 \
  --ckpt_dir ckpts/CVD_Men_Simplified \
  --ckpt_name transformer_model_t0.pth \
  --epochs 200 \
  --batch_size 2048 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --val_ratio 0.2 \
  --patience 30 \
  --embed_dim 128 \
  --num_prompts 16 \
  --top_k 4 \
  --t0 10 \
  --seed 42
```

### 4) Run checkpoint inference (PyTorch)

```bash
python infer.py \
  --data_file datasets/example_cvd_dataset.csv \
  --ckpt_file ckpts/CVD_Men_Simplified/transformer_model_t0.pth \
  --model_type Simplified \
  --sex 0 \
  --t0 10 \
  --output_csv ckpts/CVD_Men_Simplified/infer_results.csv
```

### 5) Export to ONNX

```bash
python onnx.py \
  --ckpt_file ckpts/CVD_Men_Simplified/transformer_model_t0.pth \
  --onnx_file ckpts/CVD_Men_Simplified/transformer_model_t0.onnx \
  --model_type Simplified \
  --embed_dim 128 \
  --num_prompts 16 \
  --top_k 4 \
  --opset 17 \
  --batch_size 1 \
  --t0 10 \
  --verify
```

## 📥 Input Data Requirements

Supported dataset formats:

The data content format should follow `datasets/example_dataset.csv` (including required column names and compatible value types).

- `.csv`
- `.xls`
- `.xlsx`

Required label columns:

- `Event`: `0` = no, `1` = yes (CVD, MI, STROKE)
- `Time`

Core predictors:

| Item | Value space / meaning |
|---|---|
| `Sex` | Discrete: `0` = male, `1` = female, used only for subgroup filtering |
| `Age` | Continuous: `[40, 79]` |
| `Estimated Glomerular Filtration Rate` | Continuous: `[15, 140]  mL/min/1.73m²` |
| `Total Cholesterol` | Continuous: `[2, 11]  mmol/L` |
| `High-density Lipoprotein Cholesterol` | Continuous: `[0.5, 4]  mmol/L` |
| `Systolic Blood Pressure` | Continuous: `[70, 200]  mmHg` |
| `Body Mass Index` | Continuous: `[18.5, 39.9]  kg/m²` |
| `Sleep Duration` | Continuous: `[5, 10]  hours` |
| `County-level Area-Deprivation Index` | Continuous: ` `, refer to [doi:10.1136/jech-2024-223570](https://doi.org/10.1136/jech-2024-223570) |
| `Antihypertensive Treatment` | Discrete: `0` = no, `1` = yes |
| `Lipid Lowering Treatment` | Discrete: `0` = no, `1` = yes |
| `Diabetes Mellitus` | Discrete: `0` = no, `1` = yes |
| `Current Smoker` | Discrete: `0` = no, `1` = yes |
| `Northern China Residence` | Discrete: `0` = no, `1` = yes, divided by the Yangtze River |
| `Alcohol Consumption` | Discrete: `0` = no, `1` = yes |
| `Urban/Rural Residence` | Discrete: `0` = urban, `1` = rural |

Additional predictors required by `Full`:

| Item | Value space / meaning |
|---|---|
| `Fasting Glucose` | Continuous: `[3, 20]  mmol/L` |
| `2-hour Postprandial Glucose` | Continuous: `[3, 30]  mmol/L` |
| `Waist Circumference` | Continuous: `[50, 130]  cm` |
| `Urinary Albumin-to-Creatinine Ratio` | Continuous: `[0, 25000]  mg/g` |
| `HbA1c` | Continuous: `[4, 10]  %` |
| `Family History of CVD` | Discrete: `0` = no, `1` = yes |
| `Glucose Lowering Treatment` | Discrete: `0` = no, `1` = yes |

## 📤 Outputs

Training artifacts (`ckpt_dir`):

- `transformer_model_t0.pth`
- `mean.npy`, `std.npy`
- `base_event_times.npy`, `base_hazards.npy`

Inference artifacts:

- CSV with `event,time,risk`
- Printed C-index in terminal

## 📦 Pretrained Models

You can download pretrained model files from either source and place them under the `ckpts/` directory:

- Google Drive: [AIHeart-CVD-Prediction Models](https://drive.google.com/drive/folders/1jS7zJhuF1ZqtQ3I4kKz_pnIdQpzksPfs?usp=drive_link)
- Dropbox: [AIHeart-CVD-Prediction Models](https://www.dropbox.com/scl/fo/xh96to67pmfiaeuyys96s/APtUk9OutGuzaa3ib_HQ7dc?rlkey=tyyfntbk4d56zpb9pqiee66w2&st=erqgr3sp&dl=0)

## 📁 Project Structure

```text
AIHeart-CVD-Prediction/
├── train.py                 # Train model and save checkpoint artifacts
├── infer.py                 # PyTorch checkpoint inference
├── onnx.py                  # Export checkpoint to ONNX (optional parity check)
├── predict.py               # ONNXRuntime inference
├── preprocess.py            # Input validation + feature engineering
├── proprocess.py            # Survival utilities (C-index, baseline hazard, etc.)
├── loss.py                  # Cox partial likelihood loss
├── models/
│   ├── transformer.py       # Core models: TrainAMFormer / InferAMFormer
│   ├── mlp.py
│   └── kan.py
├── datasets/                # Example datasets
├── ckpts/                   # Pre-exported ONNX models
└── requirements.txt
```

## ❓ Troubleshooting

- `Failed data processing`: check file extension and required columns.
- `No samples after Sex filter`: verify that the selected `Sex` value exists.
- `mean.npy or std.npy not found`: run training first or verify checkpoint folder contents.
- ONNX input-name mismatch: `predict.py` tries `features/t0` first, then falls back to input order.
