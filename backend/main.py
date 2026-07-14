import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from backend.model_manager import ModelManager, get_training_state, training_state

app = FastAPI(title="Sentify AI - Sentiment Analysis API")

# Initialize Model Manager
manager = ModelManager()

# Pydantic Schemas
class TrainRequest(BaseModel):
    sample_size: int = 10000

class PredictRequest(BaseModel):
    text: str

@app.get("/api/status")
def get_status():
    """Returns the current background training status and metrics if available"""
    state = get_training_state()
    # Check if models are currently loaded and ready
    state["ready"] = (manager.preprocessor is not None and 
                      manager.nb_model is not None and 
                      manager.lr_model is not None and 
                      manager.lstm_model is not None)
    return state

@app.post("/api/train")
def train_models(request: TrainRequest):
    """Triggers the model training pipeline in a background thread"""
    state = get_training_state()
    if state["status"] in ["downloading", "preprocessing", "training_nb", "training_lr", "training_lstm"]:
        return {"status": "error", "message": "Training is already in progress."}
    
    # Cap sample size to a reasonable range
    sample_size = min(max(request.sample_size, 1000), 50000)
    
    # Start training
    manager.train_all(sample_size=sample_size)
    return {"status": "started", "message": f"Training initiated with {sample_size} samples."}

@app.post("/api/predict")
def predict_sentiment(request: PredictRequest):
    """Predicts the sentiment of the input text using all three models"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    # Check if models are ready
    if manager.preprocessor is None:
        raise HTTPException(
            status_code=503, 
            detail="Models are not trained or loaded. Please trigger training first."
        )
    
    try:
        predictions = manager.predict(request.text)
        return {
            "text": request.text,
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve Frontend static files
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    print(f"Warning: Frontend directory '{FRONTEND_DIR}' not found. Static files won't be served.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
