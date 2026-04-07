<div align="center">

# 🧠 Explainable ArXiv Paper Classifier

### Multi-Label Research Paper Classification with Interpretable AI

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)

---

*Classify research paper abstracts and visually understand **why** the model made its predictions — powered by LIME, SHAP, and BERT Attention mapping.*

</div>

---

<!-- ============================================ -->
<!--        👇 ADD YOUR PHOTO / AVATAR BELOW 👇     -->
<!-- ============================================ -->

<div align="center">

## 👤 About the Author

<!-- 
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │   Replace the placeholder below with your photo:            │
  │                                                             │
  │   Option 1 — Local image (add image to repo root):          │
  │     ![Your Name](./your-photo.png)                          │
  │                                                             │
  │   Option 2 — External URL:                                  │
  │     ![Your Name](https://your-image-url.com/photo.jpg)      │
  │                                                             │
  │   Option 3 — GitHub avatar (auto from your username):       │
  │     <img src="https://github.com/<username>.png"            │
  │          width="150" style="border-radius:50%"/>            │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
-->

<img src="https://via.placeholder.com/150?text=Your+Photo" width="150" height="150" style="border-radius: 50%;" alt="Author Photo"/>

**Your Name**

<!-- Add a short bio, links, or socials here -->
<!-- [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin)](https://linkedin.com/in/yourprofile) -->
<!-- [![Portfolio](https://img.shields.io/badge/Portfolio-black?style=flat&logo=vercel)](https://yourportfolio.com) -->

</div>

<!-- ============================================ -->
<!--        👆 ADD YOUR PHOTO / AVATAR ABOVE 👆     -->
<!-- ============================================ -->

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🏷️ **Multi-Label Classification** | Classifies abstracts into multiple ArXiv categories simultaneously |
| 🟢 **LIME Explanations** | Highlights words that positively or negatively influenced each prediction |
| 🔵 **SHAP Analysis** | Shapley-value-based feature attribution for fine-grained interpretability |
| 🟣 **Attention Mapping** | Visualizes native DistilBERT attention weights across input tokens |
| ⚡ **Real-Time Inference** | FastAPI backend with instant prediction & explanation endpoints |
| 🎨 **Glassmorphism UI** | Modern, minimal React interface with a clean white-themed design |

---

## 🏗️ Architecture

```
Arxiv-reseach-100k/
│
├── frontend/                    # React + Vite application
│   ├── src/
│   │   ├── App.jsx              # Main application shell
│   │   ├── components/
│   │   │   └── ExplainabilityViewer.jsx   # LIME / SHAP / Attention visualizer
│   │   ├── index.css            # Global styles & glassmorphism design system
│   │   └── main.jsx             # React entry point
│   └── index.html
│
├── backend/                     # FastAPI server
│   ├── main.py                  # API routes: /predict, /explain
│   └── explainability.py        # LIME, SHAP, Attention wrapper logic
│
├── model_training/              # ML pipeline
│   ├── process_data.py          # Data cleaning & multi-hot encoding
│   └── text_classifier.py       # DistilBERT fine-tuning script
│
├── data/                        # Raw & processed datasets
└── bert-multi-label-model/      # Saved model artifacts
```

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** ≥ 18 & **npm** (frontend)
- **Python** ≥ 3.8 (backend & training)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/Arxiv-reseach-100k.git
cd Arxiv-reseach-100k
```

### 2️⃣ Backend Setup

```bash
# Create & activate virtual environment
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

> **Note:** First launch may take a moment while HuggingFace downloads and caches `distilbert-base-uncased`.

### 3️⃣ Frontend Setup

```bash
# In a new terminal
cd frontend
npm install
npm run dev
```

The app will be live at **http://localhost:5173** 🎉

---

## 🧪 Model Training (Optional)

Train your own fine-tuned DistilBERT model instead of using the default base model:

1. **Download the dataset** from [HuggingFace — CShorten/ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers/tree/main) and place `ML-Arxiv-Papers.csv` inside the `data/` folder.

2. **Process the data:**
   ```bash
   cd model_training
   python process_data.py
   ```

3. **Fine-tune DistilBERT:**
   ```bash
   python text_classifier.py --train
   ```
   > 💡 **Checkpoints** are saved at the end of every epoch. If you cancel mid-training (`Ctrl+C`), all previously completed epochs are safely persisted.

4. **Point the backend to your model** — In `backend/explainability.py`, update the `model_path` from `"distilbert-base-uncased"` to the path of your `final/` training output folder.

---

## 🎨 Explainability Methods

<div align="center">

| Method | What it Shows | Color Coding |
|--------|--------------|--------------|
| **LIME** | Per-word contribution to the prediction | 🟢 Green = supports · 🔴 Red = opposes |
| **SHAP** | Shapley-value feature importance | 🔵 Blue = high impact · Gray = low impact |
| **Attention** | Transformer self-attention weights | 🟣 Purple intensity = attention strength |

</div>

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18, Vite, Vanilla CSS (Glassmorphism) |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **ML Model** | DistilBERT (HuggingFace Transformers) |
| **Explainability** | LIME, SHAP, Native Attention |
| **Data** | Scikit-learn, NumPy, Datasets |

</div>

---

## 📄 API Reference

### `POST /predict`

Classify an abstract into ArXiv categories.

```json
// Request
{ "abstract": "We propose a novel transformer architecture for..." }

// Response
{ "predictions": [{ "label": "cs.LG", "confidence": 0.92 }, ...] }
```

### `POST /explain`

Get interpretability results for a given abstract.

```json
// Request
{ "abstract": "...", "method": "lime" }  // method: "lime" | "shap" | "attention"

// Response
{ "explanation": { "tokens": [...], "scores": [...] } }
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repo
2. Create your branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

*Built with ❤️ and a passion for Explainable AI*

</div>
