from fastapi import FastAPI, UploadFile, File
import shutil
import os

from src.preprocess import process_file

from src.house_predict import predict_price

app = FastAPI()

UPLOAD_DIR = "data/raw"
# ensure folder exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Medical AI API is running 🚀"}
    
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 👉 NEW: process the file
    processed_path = process_file(file_path)

    return {
        "message": f"File '{file.filename}' uploaded successfully",
        "processed_file": processed_path
        }
    
    

@app.get("/predict-house")
def predict_house(
    area: int,
    bedrooms: int,
    bathrooms: int,
    age: int,
    distance_to_city: int,
    parking: int
):
    result = predict_price(
        area,
        bedrooms,
        bathrooms,
        age,
        distance_to_city,
        parking
    )

    return {
        "predicted_price": float(result)
    }