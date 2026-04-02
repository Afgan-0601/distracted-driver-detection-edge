# Distracted Driver Detection — Edge AI System

A real-time **inside-cabin driver behaviour classifier** built with MobileNetV2, deployed as a FastAPI service, and designed to run on edge devices (Jetson Nano, Raspberry Pi 4).

Classifies 10 driver behaviours from a single dashboard camera image and computes a **composite risk score** with sustained-distraction tracking.

---

## Classes

| ID | Key | Behaviour | Base Risk |
|----|-----|-----------|-----------|
| 0  | c0  | Safe driving | 0 |
| 1  | c1  | Texting (right hand) | 95 |
| 2  | c2  | Phone call (right hand) | 80 |
| 3  | c3  | Texting (left hand) | 95 |
| 4  | c4  | Phone call (left hand) | 80 |
| 5  | c5  | Radio / dashboard controls | 30 |
| 6  | c6  | Drinking | 65 |
| 7  | c7  | Reaching behind | 70 |
| 8  | c8  | Hair / makeup | 55 |
| 9  | c9  | Talking to passenger | 20 |

---

## Risk Engine

`src/utils/risk_calculator.py` computes a **composite risk score (0–100)** per frame:

```
composite_risk = base_risk × confidence × sustained_multiplier
                 → EMA smoothed → mapped to SAFE / LOW / MEDIUM / HIGH / CRITICAL
```

- **Sustained multiplier** — increases risk the longer the driver stays distracted (piecewise-linear: 0 s → ×1.0, 10 s → ×2.0, 20 s → ×2.5)
- **EMA smoothing** — reduces jitter between frames
- **Stateful per session** — call `reset()` between driver sessions

---

## Project Structure

```
distracted-driver-detection-edge/
├── src/
│   ├── config.py                  # thresholds, class mappings, hyperparameters
│   ├── utils/risk_calculator.py   # core risk engine
│   ├── data/                      # dataset, transforms
│   ├── models/classifier.py       # MobileNetV2 + ONNX export
│   ├── training/                  # trainer, evaluator
│   └── pipeline/detection_pipeline.py  # end-to-end inference (PyTorch + ONNX)
├── api/
│   ├── main.py                    # FastAPI app
│   ├── routes/detection.py        # POST /predict, /predict/batch
│   └── routes/health.py           # GET /health
├── scripts/
│   ├── train.py                   # train from CLI
│   └── export_model.py            # export .pt → .onnx
├── notebooks/
│   ├── train_colab.ipynb          # Google Colab training
│   └── train_kaggle.ipynb         # Kaggle training (dataset pre-mounted)
├── tests/                         # 29 unit tests
├── Dockerfile                     # multi-stage production image
└── deployment/docker-compose.yml
```

---

## Quick Start

### 1. Setup environment

```bash
conda create -n ddd python=3.11 -y
conda activate ddd
pip install -r requirements.txt
```

### 2. Add dataset

Download from [Kaggle — State Farm Distracted Driver Detection](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection) and place at:

```
data/raw/imgs/train/c0/  *.jpg
data/raw/imgs/train/c1/  *.jpg
...
data/raw/imgs/train/c9/  *.jpg
```

### 3. Train

```bash
# Local (CPU / MPS on Mac)
python scripts/train.py --epochs 20 --device mps --workers 0

# With GPU
python scripts/train.py --epochs 20 --device cuda --batch-size 64 --workers 4
```

### 4. Export to ONNX

```bash
python scripts/export_model.py --verify
```

### 5. Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API docs available at: `http://localhost:8000/docs`

---

## Training on Kaggle (Free GPU)

1. Go to the [competition page](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection) → **Code → New Notebook**
2. Enable GPU: **Settings → Accelerator → GPU T4 x2**
3. Upload `notebooks/train_kaggle.ipynb` or run cells manually — dataset is pre-mounted
4. Download `best_model.pt` and `driver_classifier.onnx` from `/kaggle/working/`
5. Place weights in `models/weights/`

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/predict` | Upload driver image → classification + risk |
| `POST` | `/api/v1/predict/batch` | Offline batch evaluation from predictions list |
| `POST` | `/api/v1/session/reset` | Reset sustained-distraction timer |
| `GET`  | `/api/v1/classes` | List all classes with base risk scores |
| `GET`  | `/health` | Liveness probe |

**Example request:**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
     -F "file=@driver_image.jpg"
```

**Example response:**
```json
{
  "class_id": 1,
  "class_key": "c1",
  "label": "texting_right",
  "confidence": 0.9241,
  "low_confidence": false,
  "risk": {
    "composite_risk": 97.5,
    "risk_level": "CRITICAL",
    "alert": true,
    "sustained_seconds": 6.2,
    "sustained_multiplier": 1.53
  }
}
```

---

## Docker

```bash
# Build and run
docker compose -f deployment/docker-compose.yml up --build

# Or directly
docker build -t distracted-driver-api .
docker run -p 8000:8000 -v ./models/weights:/app/models/weights distracted-driver-api
```

---

## Tests

```bash
pytest tests/ -v
# 29 tests: risk calculator logic, model architecture, ONNX export
```

---

## Model

| Property | Value |
|----------|-------|
| Backbone | MobileNetV2 (ImageNet pretrained) |
| Parameters | ~3.4 M |
| Input | 224 × 224 RGB |
| Output | 10-class softmax |
| Export | ONNX opset 17 |
| Training strategy | 2-phase: frozen backbone (warm-up) → full fine-tune |

---

## Dataset

[State Farm Distracted Driver Detection](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection) — 22,424 labelled inside-cabin driver images across 10 classes.

---

## Tech Stack

- **Model**: PyTorch 2.2 + torchvision (MobileNetV2)
- **Inference**: ONNX Runtime (edge), PyTorch (server)
- **API**: FastAPI + Uvicorn + Pydantic v2
- **Containerisation**: Docker (multi-stage)
- **CI**: GitHub Actions
