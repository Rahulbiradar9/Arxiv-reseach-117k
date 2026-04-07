<div align="center">

# Explainable ArXiv Paper Classifier

**Multi-Label Research Paper Classification with Interpretable AI**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)

</div>

---

A full-stack application that classifies research paper abstracts into ArXiv categories and visually explains *why* the model made its predictions — using LIME, SHAP, and native BERT Attention mapping.

---

## Screenshots

<!-- Add your model training screenshots and reports below -->

### Training Results

<!--
  Place your training screenshots here. Example:
  ![Training Loss Curve](./screenshots/training_loss.png)
  ![Training Accuracy](./screenshots/training_accuracy.png)
-->

<br>

### Classification Report

<!--
  Place your classification report / confusion matrix here. Example:
  ![Classification Report](./screenshots/classification_report.png)
  ![Confusion Matrix](./screenshots/confusion_matrix.png)
-->

<br>

### Application Interface

<!--
  Place your UI screenshots here. Example:
  ![App Interface](./screenshots/app_ui.png)
  ![Explainability View](./screenshots/explainability.png)
-->

<br>

---

## Features

- **Multi-Label Classification** — Classifies abstracts into multiple ArXiv categories simultaneously using DistilBERT.
- **LIME Explanations** — Highlights words that positively or negatively influenced each prediction.
- **SHAP Analysis** — Shapley-value-based feature attribution for fine-grained interpretability.
- **Attention Mapping** — Visualizes native DistilBERT attention weights across input tokens.
- **Real-Time Inference** — FastAPI backend with instant prediction and explanation endpoints.
- **Clean UI** — Minimal React interface with a white-themed glassmorphism design.

---

## Architecture

```
Arxiv-reseach-100k/
│
├── frontend/                          React + Vite application
│   ├── src/
│   │   ├── App.jsx                    Main application shell
│   │   ├── components/
│   │   │   └── ExplainabilityViewer.jsx   LIME / SHAP / Attention visualizer
│   │   ├── index.css                  Global styles and design system
│   │   └── main.jsx                   React entry point
│   └── index.html
│
├── backend/                           FastAPI server
│   ├── main.py                        API routes — /predict, /explain
│   └── explainability.py              LIME, SHAP, Attention wrapper logic
│
├── model_training/                    ML pipeline
│   ├── process_data.py                Data cleaning and multi-hot encoding
│   └── text_classifier.py            DistilBERT fine-tuning script
│
├── data/                              Raw and processed datasets
└── bert-multi-label-model/            Saved model artifacts
```

---

## Tech Stack

| Layer             | Technology                                  |
|-------------------|---------------------------------------------|
| Frontend          | React 18, Vite, Vanilla CSS                 |
| Backend           | FastAPI, Uvicorn, Pydantic                   |
| ML Model          | DistilBERT (HuggingFace Transformers)        |
| Explainability    | LIME, SHAP, Native Attention                 |
| Data Processing   | Scikit-learn, NumPy, HuggingFace Datasets    |

---

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+

### Backend

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn pydantic torch transformers lime shap datasets scikit-learn numpy

# Start the API server
cd backend
uvicorn main:app --reload --port 8000
```

> Initial launch may take a moment while HuggingFace downloads and caches `distilbert-base-uncased`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:5173`.

---

## Model Training

To fine-tune your own DistilBERT model instead of using the base model:

1. Download the dataset from [HuggingFace — CShorten/ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers/tree/main) and place `ML-Arxiv-Papers.csv` inside the `data/` folder.

2. Process the data:
   ```bash
   cd model_training
   python process_data.py
   ```

3. Fine-tune DistilBERT:
   ```bash
   python text_classifier.py --train
   ```
   Checkpoints are saved at the end of every epoch. If you cancel mid-training, all previously completed epochs are safely persisted.

4. Update the model path in `backend/explainability.py` — change `model_path="distilbert-base-uncased"` to point to the `final/` training output folder.

---

## Explainability Methods

| Method    | What it Shows                           | Color Coding                                  |
|-----------|-----------------------------------------|-----------------------------------------------|
| LIME      | Per-word contribution to prediction     | Green = supports, Red = opposes               |
| SHAP      | Shapley-value feature importance        | Blue = high impact, Gray = low impact         |
| Attention | Transformer self-attention weights      | Purple intensity = attention strength         |

---

## API Reference

### `POST /predict`

Classify an abstract into ArXiv categories.

```json
// Request
{ "abstract": "We propose a novel transformer architecture for..." }

// Response
{ "predictions": [{ "label": "cs.LG", "confidence": 0.92 }] }
```

### `POST /explain`

Get interpretability results for a given abstract.

```json
// Request
{ "abstract": "...", "method": "lime" }

// Response
{ "explanation": { "tokens": ["..."], "scores": [0.12, -0.05] } }
```

Supported methods: `lime`, `shap`, `attention`

---

## Contributing

1. Fork the repository
2. Create your branch — `git checkout -b feature/your-feature`
3. Commit changes — `git commit -m 'Add your feature'`
4. Push — `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

This project is open source and available under the [MIT License](LICENSE).
