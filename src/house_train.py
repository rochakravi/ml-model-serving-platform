# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import joblib

# data = pd.read_csv("data/houses.csv")

# X = data[["area", "bedrooms"]]
# y = data["price"]

# model = LinearRegression()

# model.fit(X, y)

# joblib.dump(model, "house_model.pkl")

# print("House model trained successfully")
## 2nd version
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import joblib

# data = pd.read_csv("data/houses.csv")

# X = data[
#     [
#         "area",
#         "bedrooms",
#         "bathrooms",
#         "age",
#         "distance_to_city",
#         "parking"
#     ]
# ]

# y = data["price"]

# model = LinearRegression()

# model.fit(X, y)

# joblib.dump(model, "house_model.pkl")

# print("House model trained successfully")

import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# load dataset
data = pd.read_csv("data/houses.csv")

# features
X = data[
    [
        "area",
        "bedrooms",
        "bathrooms",
        "age",
        "distance_to_city",
        "parking"
    ]
]

# target
y = data["price"]

# split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# model
model = LinearRegression()

# train
model.fit(X_train, y_train)

# predict on test data
predictions = model.predict(X_test)

# evaluate
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# save model
joblib.dump(model, "house_model.pkl")

print("House model trained and saved successfully")