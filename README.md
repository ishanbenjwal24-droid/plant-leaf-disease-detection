# Plant Leaf Disease Detection Using ML and DL

An educational, full-stack semester project that compares a feature-based machine-learning model with a convolutional neural network on PlantVillage leaf images. Uploads are processed in memory; prediction history stores results and timing metadata, never the uploaded image.

The app is not a verified plant diagnosis. PlantVillage images were collected in controlled conditions, so results may not generalize to field photographs.

## Models and stack

- **Traditional ML:** HOG shape descriptors + HSV color statistics + LinearSVC.
- **Deep learning:** CNN with training-only image augmentation, built with TensorFlow/Keras.
- **Backend:** FastAPI, SQLAlchemy, SQLite.
- **Frontend:** React, TypeScript, Vite.
- **Evaluation:** shared deterministic train/validation/test manifest, accuracy, macro precision/recall/F1, per-class scores, confusion matrix, model size, and average model inference time.

## Project layout

```text
backend/app/       FastAPI endpoints, preprocessing, inference, SQLite
frontend/          React user interface
training/          dataset preparation and ML/DL training
data/              local dataset configuration, split manifest, SQLite DB (ignored)
artifacts/         models, class names, metadata, evaluation metrics (ignored)
docs/              report, presentation outline, API and design documents
tests/             preprocessing, API, and database tests
```

## Requirements

- Python 3.11 recommended (Python 3.10–3.12 should work with compatible wheels).
- Node.js 20 or newer with npm.
- Windows, macOS, or Linux. TensorFlow uses CPU on native Windows; GPU training on Windows requires WSL2 and separate GPU setup. See the [TensorFlow installation guide](https://www.tensorflow.org/install/pip).
- The PlantVillage dataset, obtained separately from the [dataset repository](https://github.com/spMohanty/PlantVillage-Dataset).

The dataset root must directly contain one subdirectory per class:

```text
PlantVillage/
  Apple___Apple_scab/
    image1.JPG
  Apple___Black_rot/
    image2.JPG
```

## Setup on Windows PowerShell

Run commands from this project directory:

```powershell
py -3.11 -m venv "$env:LOCALAPPDATA\venvs\plant-leaf"
& "$env:LOCALAPPDATA\venvs\plant-leaf\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `DATASET_DIR` to the folder containing the class folders, for example:

```text
DATASET_DIR=C:/datasets/PlantVillage
```

Prepare data and train the models:

```powershell
python -m training.prepare_data --max-per-class 200
python -m training.train_ml
python -m training.train_cnn
```

Run the backend in one PowerShell window:

```powershell
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Run the frontend in a second window:

```powershell
Set-Location frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:5173`. The interactive API documentation is at `http://127.0.0.1:8000/docs`.

Using a short venv path under `%LOCALAPPDATA%` also avoids Windows path-length issues during TensorFlow installation. If PowerShell blocks virtual environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that window and activate again.

## Setup on macOS/Linux

From the project directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `DATASET_DIR` in `.env`, then run:

```bash
python -m training.prepare_data --max-per-class 200
python -m training.train_ml
python -m training.train_cnn
```

In a terminal from the project root, start the API:

```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:5173`.

## Data preparation and reproducibility

`training.prepare_data` validates supported JPEG, PNG, WebP, and BMP files, reports unreadable files, hashes images to find exact duplicates, excludes duplicate groups before splitting, and skips classes with fewer than three unique readable images. It samples up to 200 images per class by default and uses seed 42 to make a class-stratified, approximately 70/15/15 split. Increase the cap with `--max-per-class 500`.

The script creates:

- `data/splits.csv`: shared relative file paths, SHA-256 hashes, class labels, and train/validation/test split.
- `data/duplicates.csv`: duplicate paths and hashes excluded from the split.
- `data/unreadable_files.csv`: image files that could not be decoded.
- `data/dataset_config.json`: dataset path, seed, class counts, split settings, and a split-manifest fingerprint.
- `artifacts/class_names.json` and `artifacts/metadata.json`: model label order and preprocessing metadata.

The CNN applies random flips, rotations, and zoom only while training. Validation and test data do not receive augmentation. Both models evaluate against the same held-out test rows. Metrics are written only after a real training run; the UI shows “not trained” or “not available” before that.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Backend state and trained-model availability |
| `GET` | `/api/classes` | Dataset class labels, if prepared |
| `GET` | `/api/metrics` | Real evaluation metrics or unavailable state |
| `POST` | `/api/predict` | Predict with one model; multipart `file` and `model=ml\|cnn` |
| `POST` | `/api/compare` | Run both models on one multipart `file` |
| `GET` | `/api/history` | Recent prediction metadata |
| `DELETE` | `/api/history` | Clear prediction metadata |

Example from PowerShell using `curl.exe`:

```powershell
curl.exe -F "file=@C:/images/leaf.jpg" http://127.0.0.1:8000/api/compare
curl.exe -F "file=@C:/images/leaf.jpg" -F "model=ml" http://127.0.0.1:8000/api/predict
```

The API accepts JPEG, PNG, WebP, and BMP images up to 10 MB (configurable). It checks image dimensions and decoded format, applies EXIF orientation, and converts to RGB. “Score” means an uncalibrated model output, not a probability.

## Model artifact and database management

After training, `artifacts/` contains `ml_model.joblib`, `cnn_model.keras`, class names, model metadata, and each model’s metrics JSON. The backend loads models lazily and gives a setup message if artifacts are missing. Each trained model and metrics file records the split-manifest fingerprint; if you regenerate a dataset split, old model results are marked unavailable until retrained. `DATA_DIR`, `ARTIFACTS_DIR`, `DATABASE_PATH`, `MAX_UPLOAD_MB`, `MAX_IMAGE_SIDE`, `MAX_IMAGE_PIXELS`, and `CORS_ORIGINS` can be set in `.env`.

SQLite stores only timestamp, prediction mode, model names/classes/scores/versions, and inference timing. Uploaded images remain in memory and are not saved. See [database schema](docs/database_schema.md).

## Tests

Install dependencies first, then from the project root:

```bash
python -m pytest
```

These tests use generated images and temporary SQLite files; they do not claim real model accuracy. Frontend type-check/build:

```bash
cd frontend
npm run typecheck
npm run build
```

## College submission materials

- [Project report](docs/project_report.md) includes fill-in sections for actual measurements; no result values are invented.
- [Presentation outline](docs/presentation_outline.md) provides a 9-slide structure.
- [Architecture diagram](docs/architecture.mmd) and [API examples](docs/api_examples.md).
- Cite the PlantVillage dataset and the research paper linked from its repository in your submission. Record the actual dataset version and your measured test results.
