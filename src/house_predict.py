import joblib
import pandas as pd

model = joblib.load("house_model.pkl")

def predict_price(
    area,
    bedrooms,
    bathrooms,
    age,
    distance_to_city,
    parking
):
    input_data = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age": age,
        "distance_to_city": distance_to_city,
        "parking": parking
    }])

    prediction = model.predict(input_data)

    return prediction[0]