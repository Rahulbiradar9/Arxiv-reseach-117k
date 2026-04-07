import sys
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from explainability import ExplainableBertPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Explainable AI - Multi-Label Classifier")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model at startup
predictor = None

@app.on_event("startup")
async def startup_event():
    global predictor
    logger.info("Initializing ExplainableBertPredictor...")
    try:
        # Load the fine-tuned model from the local checkpoint
        model_path = os.path.join(os.path.dirname(__file__), "..", "bert-multi-label-model", "final")
        predictor = ExplainableBertPredictor(model_path)
        logger.info("Predictor Initialized Successfully.")
    except Exception as e:
        logger.error(f"Failed to load the model: {e}")

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_endpoint(req: TextRequest):
    if not predictor:
        raise HTTPException(status_code=500, detail="Model is still loading or failed to load")
    
    logger.info("Running prediction...")
    try:
        # Standard predict call to get probabilities mapping
        # `predict_proba` returns an array of shape (current_batch, num_labels)
        probs = predictor.predict_proba([req.text])[0]
        
        # We process predictions based on basic 0.4 threshold
        results = []
        for idx, prob in enumerate(probs):
            if prob >= 0.4:
                results.append({
                    "label": predictor.label_names[idx],
                    "confidence": round(float(prob), 4)
                })
                
        # Sort descending by confidence
        results = sorted(results, key=lambda x: x["confidence"], reverse=True)
        return {"predictions": results}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def explain_endpoint(req: TextRequest):
    if not predictor:
        raise HTTPException(status_code=500, detail="Model is still loading or failed to load")
        
    logger.info("Generating explanations... (This may take a moment)")
    try:
        # Utilizing the built-in Explain_All handler from the predictor 
        # Using threshold 0.1 for the untuned base-bert model to guarantee returning visualizations
        explanations = predictor.explain_all(req.text, threshold=0.4)
        return {"explanations": explanations}
    except Exception as e:
        logger.error(f"Explainability error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
