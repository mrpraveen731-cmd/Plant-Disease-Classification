
from pathlib import Path
import sys
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


# ==========================================
# 1. PROJECT PATH CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================
# 2. IMPORT TRAINED MODEL PREDICTION
# ==========================================

from src.predict import predict_image


# ==========================================
# 3. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Plant Disease Classification API",
    description="Plant disease prediction using a trained ResNet18 deep learning model.",
    version="1.0.0"
)


# ==========================================
# 4. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Plant Disease Classification API is running!",
        "model": "ResNet18",
        "status": "success"
    }


# ==========================================
# 5. HEALTH CHECK ENDPOINT
# ==========================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "project": "Plant Disease Classification",
        "model": "ResNet18"
    }


# ==========================================
# 6. PLANT DISEASE PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check file type
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported."
        )

    try:

        # Read uploaded image
        image_bytes = await file.read()

        # Check file size (10 MB maximum)
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image size must be below 10 MB."
            )

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        # Validate image
        image = Image.open(BytesIO(image_bytes))
        image.verify()

        # Reopen image for prediction
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Predict plant disease
        result = predict_image(image)

        return {
            "status": "success",
            "filename": file.filename,
            "predicted_disease": result["predicted_disease"],
            "confidence": result["confidence"]
        }

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file."
        )

    finally:
        await file.close()
