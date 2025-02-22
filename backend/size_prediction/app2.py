from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import BayesianRidge
from xgboost import XGBClassifier

app = FastAPI()

# ✅ Define input model
class SizeRequest(BaseModel):
    height: float = Field(..., example=170.0, description="Height in cm")
    gender: str = Field(..., example="male", description="Gender (male/female)")

# 🔹 **Synthetic Training Data** (For ML Model)
np.random.seed(42)
heights = np.random.randint(140, 190, 500)  # Height in cm
genders = np.random.choice(["male", "female"], 500)  # Random gender distribution

# ✅ **Size distribution mapping (statistically adjusted)**
size_mapping_male = {140: "XS", 150: "S", 160: "M", 170: "L", 180: "XL"}
size_mapping_female = {140: "XS", 150: "S", 160: "M", 170: "L", 180: "XL"}

df = pd.DataFrame({"height": heights, "gender": genders})
df["size"] = df.apply(lambda row: size_mapping_male.get(row["height"], "M") if row["gender"] == "male" else size_mapping_female.get(row["height"], "M"), axis=1)

# ✅ **Gaussian Mixture Model (GMM) for Body Type Categorization**
gmm = GaussianMixture(n_components=3, random_state=42)
df["body_type"] = gmm.fit_predict(df[["height"]])

# ✅ **Polynomial Regression for Height-Size Mapping**
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(df[["height"]])

bayesian_model = BayesianRidge()
bayesian_model.fit(X_poly, df["body_type"])  # Predicts body type based on height

# ✅ **XGBoost Classifier for Final Size Prediction**
size_labels = {"XS": 0, "S": 1, "M": 2, "L": 3, "XL": 4, "XXL": 5}
df["size_label"] = df["size"].map(size_labels)

xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(df[["height", "body_type"]], df["size_label"])

# 🔹 **Function to Predict Size**
def get_predicted_size(height: float, gender: str):
    # Predict body type using Polynomial Regression
    height_poly = poly.transform([[height]])
    predicted_body_type = int(round(bayesian_model.predict(height_poly)[0]))

    # Predict final size using XGBoost
    predicted_size_label = xgb_model.predict([[height, predicted_body_type]])[0]
    size_inverse_mapping = {v: k for k, v in size_labels.items()}
    predicted_size = size_inverse_mapping[predicted_size_label]

    return predicted_size

# 🔹 **API Route**
@app.post("/get-size", summary="Get ML-Powered Size Recommendation (Height + Gender Only)", tags=["Size Recommendation"])
def size_recommendation(data: SizeRequest):
    recommended_size = get_predicted_size(data.height, data.gender)

    return {
        "height": data.height,
        "gender": data.gender,
        "recommended_size": recommended_size
    }
