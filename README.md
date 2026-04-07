# Still working on it!!
## ArXiv-Research-Data-subset : Explainable Classifier

This project is a designed to classify research paper abstracts and visually explain *why* the model made specific predictions using LIME, SHAP, and native BERT Attention mapping.

## Project Structure

- `frontend/`: React application built with Vite and vanilla CSS glassmorphism.
- `backend/`: FastAPI application exposing `/predict` and `/explain` endpoints. Contains the Explainable AI wrapper logic.
- `model_training/`: Python scripts for data processing and training the BERT model.
- `data/`: Directory for storing raw datasets and processed JSON files.
- `venv/`: Python virtual environment containing the backend and ML dependencies.

## Setup Instructions

### 1. Requirements

- Node.js & npm (for the frontend)
- Python 3.8+ (for the backend & model training)

### 2. Backend & Model Setup

Activate your Python virtual environment (assuming it is created at `./venv`):
```bash
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

If you haven't installed the dependencies:
```bash
pip install fastapi uvicorn pydantic torch transformers lime shap datasets scikit-learn numpy
```

To run the backend server natively:
```bash
cd backend
uvicorn main:app --reload --port 8000
```
*(Note: Initial load may take a moment as HuggingFace `distilbert-base-uncased` pulls down locally if not cached).*

### 3. Frontend Setup

Open a separate terminal and navigate to the frontend:
```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:5173`.

## Model Pipeline & Training

If you wish to train the model from scratch rather than testing the untuned base DistilBERT model on the UI:

1. Download the raw dataset from [HuggingFace (CShorten/ML-ArXiv-Papers)](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers/tree/main) and place the `ML-Arxiv-Papers.csv` strictly inside the `data/` folder.
2. Run data processing to clean and create multi-hot vectors:
   ```bash
   cd model_training
   python process_data.py
   ```
3. Run the DistilBERT Fine-Tuning pipeline:
   ```bash
   python text_classifier.py --train
   ```
   > **Note on Checkpoints**: Training takes time. The script is configured to save a valid checkpoint at the end of *every epoch* (e.g., `checkpoint-6616`). If you cancel the script mid-way (`Ctrl+C`), progress for the *current* incomplete epoch is lost, but previous epochs are safely saved dynamically on your hard drive!

4. Once completed, inside `backend/explainability.py`, switch `model_path="distilbert-base-uncased"` to point to the new `final/` training artifact output folder.
