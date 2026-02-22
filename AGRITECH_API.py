# ===============================================
# AgriTech Crop Yield Prediction API
# ===============================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # <-- added
from pydantic import BaseModel
from AGRITECH_python_script import predict_crop_yield  # import your function
import uvicorn

# -----------------------------
# Define input data model (matches backend)
# -----------------------------
class CropInput(BaseModel):
    Item: str
    Year: int
    average_rain_fall_mm_per_year: float
    pesticides_tonnes: float
    avg_temp: float

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(title="AgriTech Crop Yield Predictor API", version="1.0.0")

# -----------------------------
# Add CORS middleware
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# -----------------------------
# Define prediction endpoint
# -----------------------------
@app.post("/predict")
def get_prediction(data: CropInput):
    """
    Receives crop input data and returns predicted yield.
    """
    # Convert Pydantic model to dict
    input_dict = data.dict()

    # Call backend prediction function
    predicted_yield = predict_crop_yield(input_dict)

    return {"predicted_yield_hg_per_ha": round(predicted_yield, 2)}

# -----------------------------
# Run the API (dev mode)
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("AGRITECH_API:app", host="0.0.0.0", port=8000, reload=True)