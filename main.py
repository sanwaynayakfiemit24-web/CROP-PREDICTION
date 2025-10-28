from fastapi import FastAPI
from pydantic import BaseModel
import joblib


app = FastAPI(
    title="🌱 Crop Recommendation API",
    description="AI-powered crop recommendation system",
    version="1.0.0"
)

# Define data models
class FarmConditions(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    moisture:float
    rainfall: float


class RecommendationResponse(BaseModel):
    success: bool
    recommendations: str
    message: str

# Global variables for loaded models
model = None
crop_encoder = None
feature_names = None

def load_models():
    """Load model and encoders"""
    global model,  crop_encoder, feature_names
    
    try:
        model = joblib.load('model_exports/crop_model.joblib')
        crop_encoder = joblib.load('model_exports/crop_encoder.joblib')
        feature_names = joblib.load('model_exports/feature_names.joblib')
        print("✅ All models loaded successfully!")
        return True
    except Exception as e:
        print(f"⚠️  Model loading failed: {e}")
        print("💡 Please export your model first using the notebook code")
        return False

# Load models when API starts
models_loaded = load_models()

# Your custom function
def get_high_confidence_recommendations(input_features, threshold=0.03):

    
    output = crop_encoder.inverse_transform(model.predict([input_features]))
    
    return output[0]

# API endpoints
@app.get("/")
def read_root():
    return {
        "message": "🌱 Welcome to Crop Recommendation API",
        "model_loaded": models_loaded,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "recommend": "/recommend"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": models_loaded,
        "crop_classes": crop_encoder.classes_.tolist() if crop_encoder else None
    }

@app.post("/recommend", response_model=RecommendationResponse)
def recommend_crop(conditions: FarmConditions):
    """Main recommendation endpoint"""
    
    if not models_loaded:
        return RecommendationResponse(
            success=False,
            recommendations=[],
            message="Model not loaded. Please export your model first.",
        )
    

        
    # Prepare input features (in correct order)
    input_features = [
        conditions.N,
        conditions.P, 
        conditions.K,
        conditions.temperature,
        conditions.humidity,
        conditions.ph,
        conditions.moisture,
        conditions.rainfall,
    ]
    
    # Get recommendations using YOUR custom function
    recommendation = get_high_confidence_recommendations(input_features)
    
    if recommendation:
        message = f"Found {recommendation}"
    else:
        message = "No crops "
    
    return RecommendationResponse(
        success=True,
        recommendations=recommendation,
        message=message,

    )
        
