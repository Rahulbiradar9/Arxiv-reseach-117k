import torch
import numpy as np
import json
import warnings

# Suppress some common HuggingFace warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BertMultiLabelPredictor:
    def __init__(self, model_path="bert-base-uncased"):
        """
        Initializes the model and tokenizer from a given path or model name.
        """
        self.label_names = ["AI", "Networks", "Security", "Systems"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=4,
            problem_type="multi_label_classification"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, threshold=0.5):
        """
        Takes raw text, tokenizes, runs the model, and applies sigmoid.
        Returns the top predicted labels and their confidence scores.
        """
        # 1. Tokenize input
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=256, 
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 2. Run model
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # 3. Apply sigmoid
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # 4. Map probabilities to labels
        all_probs = {self.label_names[idx]: round(float(prob), 4) for idx, prob in enumerate(probs)}
        
        # 5. Determine top predictions above threshold
        top_predictions = {
            k: v for k, v in sorted(all_probs.items(), key=lambda item: item[1], reverse=True) 
            if v >= threshold
        }
        
        return {
            "top_predictions": top_predictions,
            "all_probabilities": all_probs
        }

def train():
    from datasets import load_dataset
    from transformers import TrainingArguments, Trainer, EvalPrediction
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    DATA_FILE = "../data/processed_dataset.json"
    OUTPUT_DIR = "../bert-multi-label-model"
    
    print(f"Loading dataset {DATA_FILE}...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    
    # Perform a 90/10 train-validation split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Initialize tokenizer and model
    print("Initializing distilbert-base-uncased...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=4,
        problem_type="multi_label_classification"
    )

    def tokenize_function(examples):
        # Truncate to 256 for a good balance of speed and representation
        encoding = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
        encoding["labels"] = [[float(v) for v in vector] for vector in examples["multi_hot_vector"]]
        return encoding

    print("Tokenizing the datasets...")
    train_dataset = dataset["train"].map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    eval_dataset = dataset["test"].map(tokenize_function, batched=True, remove_columns=dataset["test"].column_names)
    
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    # Define metrics function
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = 1.0 / (1.0 + np.exp(-preds))
        y_pred = (preds >= 0.5).astype(int)
        y_true = p.label_ids
        
        return {
            "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="micro", zero_division=0)
        }

    # Training Configuration
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",  
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        fp16=True,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Initiating training...")
    trainer.train()

    # Save finalized model and tokenizer
    final_path = f"{OUTPUT_DIR}/final"
    print(f"Saving model to {final_path}...")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print("Training finished successfully.")

if __name__ == "__main__":
    import sys
    if "--train" in sys.argv:
        train()
    else:
        print("Instantiating predictive model (Base uncased, untuned):")
        predictor = BertMultiLabelPredictor("distilbert-base-uncased")
        sample = "We use a neural network approach and a transformer for natural language deep learning."
        
        print(f"\nSample Text: '{sample}'")
        print("\Predicted Output:")
        res = predictor.predict(sample)
        print(json.dumps(res, indent=2))
        
        print("\n(Use 'python text_classifier.py --train' to begin the training process)")
