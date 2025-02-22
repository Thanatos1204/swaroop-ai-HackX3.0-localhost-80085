from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

app = FastAPI()

# ✅ Define input model
class SizeRequest(BaseModel):
    height: float = Field(..., example=170.0, description="Height in cm")
    weight: float = Field(..., example=70.0, description="Weight in kg")
    age: int = Field(..., example=25, description="Age in years")
    gender: str = Field(..., example="male", description="Gender (male/female)")

# 🔹 **Synthetic Training Data** (For Body Type Classification)
np.random.seed(42)
heights = np.random.randint(140, 190, 500)  # Height in cm
weights = np.random.randint(40, 120, 500)  # Weight in kg

df = pd.DataFrame({'height': heights, 'weight': weights})

# ✅ **K-Means Clustering to Detect Body Types**
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['body_type'] = kmeans.fit_predict(df[['height', 'weight']])

# ✅ **PCA for Feature Reduction**
pca = PCA(n_components=1)
df['pca_feature'] = pca.fit_transform(df[['height', 'weight']])

# ✅ **XGBoost Model to Predict Weight Range per Height**
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(df[['height', 'pca_feature']], df['weight'])

# 🔹 **Indian Height-Weight Size Chart (Adjusted via ML)**
size_chart = {
    "XS": (40, 55),
    "S": (56, 65),
    "M": (66, 75),
    "L": (76, 85),
    "XL": (86, 95),
    "XXL": (96, 110),
}

# ✅ **Function to Predict Size**
def get_predicted_size(height: float, weight: float, gender: str):
    # Predict the "expected weight range" using ML
    pca_feature = pca.transform([[height, weight]])[0][0]
    predicted_weight = xgb_model.predict([[height, pca_feature]])[0]

    # Find closest cluster (body type)
    cluster = kmeans.predict([[height, weight]])[0]

    # Adjust weight ranges dynamically
    if cluster == 0:  # Lean
        weight_adjustment = -5
    elif cluster == 1:  # Average
        weight_adjustment = 0
    else:  # Broad/Muscular
        weight_adjustment = +5

    adjusted_weight = weight + weight_adjustment

    # Find closest size category
    for size, (min_w, max_w) in size_chart.items():
        if min_w <= adjusted_weight <= max_w:
            return size

    return "XXL" if adjusted_weight > 110 else "XS"

# 🔹 **API Route**
@app.post("/get-size", summary="Get ML-Powered Size Recommendation", tags=["Size Recommendation"])
def size_recommendation(data: SizeRequest):
    recommended_size = get_predicted_size(data.height, data.weight, data.gender)

    return {
        "height": data.height,
        "weight": data.weight,
        "age": data.age,
        "gender": data.gender,
        "recommended_size": recommended_size
    }
