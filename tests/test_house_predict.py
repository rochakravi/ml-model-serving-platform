from src.house_predict import predict_price

def test_prediction():
    result = predict_price(
        2400,
        4,
        3,
        4,
        7,
        2
    )

    print("Predicted Price:", result)

test_prediction()