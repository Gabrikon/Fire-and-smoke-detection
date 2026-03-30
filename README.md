# Hybrid CNN-VLM Fire Detection for Oil & Gas Facilities

Real-time fire/smoke detection using **YOLO26** + contextual safety advisory via **GPT-4o**, streamed through a **Streamlit** webcam dashboard.

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USER/fire-detection-app.git
cd fire-detection-app

python -m venv venv
source venv/bin/activate          # Linux / Mac
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Train YOLO26 on Google Colab

1. Prepare your dataset (see Phase 1 below)
2. Zip it and upload to Google Drive
3. Open Google Colab → Runtime → Change runtime type → **T4 GPU**
4. Paste cells from `colab_training.py` one by one, run them
5. After training, `fire_yolo_best.pt` is saved to your Drive

### 3. Add weights & run

```bash
mkdir -p weights
# Download fire_yolo_best.pt from your Google Drive → place in weights/

# Verify
python -c "from ultralytics import YOLO; print(YOLO('weights/fire_yolo_best.pt').names)"

# Launch
streamlit run app.py
```

### 4. Use it

- Click **START** → allow camera access
- Enter your **OpenAI API key** in the sidebar
- Point webcam at a fire image on your phone to test
- After 5 consecutive fire frames → GPT-4o advisory appears

---

## Project Structure

```
fire-detection-app/
├── app.py                  # Streamlit dashboard (webcam + GPT-4o)
├── vlm_reasoner.py         # GPT-4o Vision integration
├── config.py               # All tunable settings
├── prepare_dataset.py      # Dataset merging / splitting
├── colab_training.py       # YOLO26 training (paste into Colab)
├── vlm_eval.py             # GPT-4o evaluation script
├── weights/
│   └── fire_yolo_best.pt   # Your trained model (git-ignored)
├── requirements.txt
├── packages.txt            # System deps for Streamlit Cloud
├── .streamlit/config.toml
├── .gitignore
└── README.md
```

---

## Phase 1: Prepare the Dataset

### Download sources

| Dataset | URL |
|---------|-----|
| D-Fire (~21k images) | https://github.com/gaiasd/DFireDataset |
| FASDD | https://www.kaggle.com/datasets/riondsilva21/fire-and-smoke-dataset-fasdd |
| Supplementary | Search Kaggle for "industrial fire detection" |

### Merge and split

Edit source paths in `prepare_dataset.py`, then:

```bash
python prepare_dataset.py consolidate --output consolidated_dataset
```

Creates `consolidated_dataset/` with `images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml`.

### Optional: subset to 20,000

```bash
python prepare_dataset.py subset consolidated_dataset 20000 --output subset_dataset
```

### Optional: refine with Roboflow

1. Upload to https://app.roboflow.com → Object Detection project
2. Review/fix bounding boxes
3. Export as YOLOv8 format (compatible with YOLO26)

### Zip & upload to Drive

```bash
cd consolidated_dataset && zip -r ../fire_smoke_dataset.zip . && cd ..
# Upload fire_smoke_dataset.zip to Google Drive
```

---

## Phase 2: Train YOLO26 on Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Runtime → T4 GPU
3. Copy cells from `colab_training.py` into the notebook
4. Run all cells
5. Download `fire_yolo_best.pt` from Drive

### About YOLO26

YOLO26 (September 2025) is Ultralytics' latest release featuring:
- **NMS-free end-to-end inference** (no post-processing bottleneck)
- **DFL removal** for simpler, faster exports
- **ProgLoss + STAL** for better small-object detection
- **MuSGD optimizer** for stable convergence
- Up to **43% faster on CPU** than YOLO11

Model sizes: `yolo26n`, `yolo26s`, `yolo26m` (recommended), `yolo26l`, `yolo26x`

---

## Phase 3: Deploy

### Option A — Streamlit Community Cloud (free)

1. Push to GitHub (weights are git-ignored)
2. In `app.py`, uncomment the `gdown` auto-download block
3. Go to https://share.streamlit.io → New app → select repo
4. Add secrets: `OPENAI_API_KEY = "sk-..."`

### Option B — AWS EC2

```bash
# Launch t3.micro Ubuntu, open port 8501
sudo apt update && sudo apt install -y python3-pip python3-venv ffmpeg
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## How It Works

```
Webcam (browser) → WebRTC → YOLO26 (every Nth frame, ~20ms)
                                    ↓
                            fire detected?
                            yes → buffer it
                                    ↓
                            5 consecutive fires + 30s cooldown?
                            yes → GPT-4o (background thread, ~3s)
                                    ↓
                            JSON advisory → dashboard panel
```

- YOLO26 runs on every sampled frame (fast, never blocks video)
- GPT-4o is rate-limited — only fires after N consecutive detections + cooldown
- GPT-4o runs in a daemon thread so video stays smooth
- Thread-safe state via Lock-protected dict

---

## Configuration

Edit `config.py` or use sidebar sliders:

| Setting | Default | What it does |
|---------|---------|--------------|
| Confidence Threshold | 0.35 | YOLO min confidence |
| Frame Skip | 3 | Process every Nth frame |
| Consecutive Frames | 5 | Fire frames before GPT-4o triggers |
| VLM Cooldown | 30s | Min gap between GPT-4o calls |

---

## Evaluation for Paper

```bash
export OPENAI_API_KEY="sk-..."
python vlm_eval.py --images test_images/ --model weights/fire_yolo_best.pt
```

Training auto-generates: `confusion_matrix.png`, `PR_curve.png`, `F1_curve.png`, `results.csv`

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Camera not working | Check browser permissions. Use Chrome/Edge. |
| Black screen on deployed app | STUN servers are configured in config.py |
| `No module named 'av'` | `sudo apt install ffmpeg && pip install av` |
| OOM on Streamlit Cloud | Use `yolo26n.pt`, increase frame_skip |
| GPT-4o never triggers | Check API key, consecutive frames slider, cooldown |

---

## License

Research use. Cite appropriately if used in publications.
