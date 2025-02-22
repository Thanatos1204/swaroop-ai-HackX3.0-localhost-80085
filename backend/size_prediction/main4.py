from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBRegressor, XGBClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsClassifier
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define input models for general size recommendation
class SizeRequest(BaseModel):
    height: float = Field(..., example=170.0, description="Height in cm")
    weight: float = Field(..., example=70.0, description="Weight in kg")
    age: int = Field(..., example=25, description="Age in years")
    gender: str = Field(..., example="male", description="Gender (male/female)")

class SizeRequestHeightGender(BaseModel):
    height: float = Field(..., example=170.0, description="Height in cm")
    gender: str = Field(..., example="male", description="Gender (male/female)")

class PantFitRequest(BaseModel):
    waist_size: float = Field(..., example=80, description="Waist size in cm")
    height: float = Field(..., example=180, description="Height in cm")
    weight: float = Field(..., example=75, description="Weight in kg")

class FitRequest(BaseModel):
    waist_size: float = Field(..., example=80, description="Waist size in cm")
    shoulder_length: float = Field(..., example=48, description="Shoulder length in cm")
    height: float = Field(..., example=180, description="Height in cm")

class ClothingRequest(BaseModel):
    height: float = Field(..., example=170, description="Height in cm")
    waist_size: float = Field(..., example=80, description="Waist size in cm")
    shoulder_length: float = Field(..., example=45, description="Shoulder length in cm")
    chest: float = Field(..., example=95, description="Chest circumference in cm")
    weight: float = Field(..., example=70, description="Weight in kg")
    hips: float = Field(..., example=98, description="Hips circumference in cm")
    gender: str = Field(..., example="male", description="Gender (male/female)")

class ClothingRecommendation(BaseModel):
    top_recommended_fit: str
    top_recommended_size: str
    pant_recommended_fit: str
    pant_recommended_size: str

class FitRequestAdjust(BaseModel):
    size: str  # Current size
    preferred_fit: str  # User's chosen fit
    recommended_fit: str  # Fit given by system

# 🔹 Synthetic Training Data (For Body Type Classification) - KEPT (used by other endpoints)
np.random.seed(42)
heights_data = np.random.randint(140, 190, 500)
weights_data = np.random.randint(40, 120, 500)

df = pd.DataFrame({'height': heights_data, 'weight': weights_data})

# K-Means Clustering to Detect Body Types - KEPT (used by other endpoints)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['body_type'] = kmeans.fit_predict(df[['height', 'weight']])

# PCA for Feature Reduction - KEPT (used by other endpoints)
pca = PCA(n_components=1)
df['pca_feature'] = pca.fit_transform(df[['height', 'weight']])

# XGBoost Model to Predict Weight Range per Height - KEPT (used by other endpoints)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(df[['height', 'pca_feature']], df['weight'])

# Indian Height-Weight Size Chart (Adjusted via ML) - KEPT (used by other endpoints)
size_chart = {
    "XS": (40, 55),
    "S": (56, 65),
    "M": (66, 75),
    "L": (76, 85),
    "XL": (86, 95),
    "XXL": (96, 110),
}

# Training data for clothing recommendation - KEPT (used by other endpoints)
training_data = np.array([
    # Waist, Shoulder, Chest, Height, Hips, BMI
    [70, 44, 90, 170, 92, 24],  # Slim, S
    [75, 46, 95, 175, 96, 24],  # Regular, M
    [80, 48, 100, 180, 100, 25], # Regular, M
    [85, 50, 105, 185, 104, 25], # Relaxed, L
    [90, 52, 110, 190, 108, 25], # Baggy, XL
    [65, 42, 85, 160, 88, 25],  # Slim, XS
    [78, 47, 98, 172, 102, 26], # Regular, L
    [82, 49, 102, 178, 106, 26], # Relaxed, L
    [88, 51, 108, 183, 110, 27], # Baggy, XL
    [92, 53, 112, 195, 114, 24], # Baggy, XXL
    [72, 45, 92, 168, 94, 25],  # Slim, S
    [76, 46, 96, 174, 98, 25],  # Regular, M
    [84, 49, 104, 179, 104, 26], # Regular, L
    [89, 52, 109, 188, 108, 25], # Baggy, XL
    [95, 55, 115, 200, 112, 24], # Baggy, XXL
    [68, 43, 88, 165, 90, 25],  # Slim, S
    [74, 45, 94, 170, 95, 25],  # Regular, M
    [79, 47, 99, 176, 99, 25],  # Regular, M
    [86, 50, 106, 182, 103, 26], # Relaxed, L
    [91, 53, 111, 192, 107, 25], # Baggy, XL
    [67, 42, 87, 162, 89, 25],  # Slim, XS
    [73, 44, 93, 169, 93, 25],  # Slim, S
    [81, 48, 101, 177, 101, 26], # Regular, L
    [87, 51, 107, 184, 105, 26], # Relaxed, XL
    [94, 54, 114, 198, 110, 24], # Baggy, XXL
    [66, 41, 86, 158, 87, 26],  # Slim, XS
    [77, 46, 97, 173, 100, 26], # Regular, M
    [83, 49, 103, 181, 104, 25], # Relaxed, L
    [90, 52, 110, 187, 108, 26], # Baggy, XL
    [96, 56, 116, 205, 112, 24], # Baggy, XXL
    [71, 44, 91, 171, 93, 24],  # Regular, S
    [74, 45, 94, 172, 95, 25],  # Regular, M
    [82, 48, 102, 178, 102, 26], # Regular, L
    [93, 54, 113, 196, 113, 28], # Baggy, XXL
    [97, 57, 117, 207, 115, 29]  # Baggy, XXL
])

# Fit labels for clothing recommendation - KEPT (used by other endpoints)
fit_labels = [
    "Slim", "Regular", "Regular", "Relaxed", "Baggy",
    "Slim", "Regular", "Regular", "Baggy", "Baggy",
    "Slim", "Regular", "Regular", "Baggy", "Baggy",
    "Slim", "Regular", "Regular", "Relaxed", "Baggy",
    "Slim", "Regular", "Regular", "Relaxed", "Baggy",
    "Slim", "Regular", "Relaxed", "Baggy", "Baggy",
    "Regular", "Regular", "Regular",
    "Baggy", "Baggy"
]

# Size labels for clothing recommendation - KEPT (used by other endpoints)
size_labels = [
    "S", "M", "M", "L", "XL",
    "XS", "L", "L", "XL", "XXL",
    "S", "M", "L", "XL", "XXL",
    "S", "M", "M", "L", "XL",
    "XS", "S", "L", "XL", "XXL",
    "XS", "M", "L", "XL", "XXL",
    "S", "M", "L",
    "XXL", "XXL"
]

# Size chart for size adjustments - KEPT (used by other endpoints)
size_chart_adjust = ["XS", "S", "M", "L", "XL", "XXL"]

# Fit adjustment logic (Bidirectional) - KEPT (used by other endpoints)
fit_adjustments = {
    "Relaxed": {"Slim": 1, "Regular": 0, "Oversized": -1},
    "Slim": {"Relaxed": -1, "Regular": 0, "Skinny": 1},
    "Regular": {"Relaxed": 0, "Slim": 0, "Oversized": 1},
    "Oversized": {"Regular": -1, "Slim": -2},
    "Skinny": {"Regular": 1, "Slim": 0}
}

# Reverse fit adjustments - KEPT (used by other endpoints)
reverse_fit_adjustments = {
    "Slim": {"Relaxed": -1, "Regular": 0, "Oversized": 2},
    "Relaxed": {"Slim": 1, "Regular": 0, "Skinny": -1},
    "Regular": {"Slim": 0, "Relaxed": 0, "Oversized": -1},
    "Oversized": {"Regular": 1, "Slim": 2},
    "Skinny": {"Regular": -1, "Slim": 0}
}

# Initialize models - KEPT (used by other endpoints)
combined_data = np.concatenate((training_data, np.array(fit_labels)[:, None], np.array(size_labels)[:, None]), axis=1)
df_clothing = pd.DataFrame(combined_data, columns=['waist_size', 'shoulder_length', 'chest', 'height', 'hips', 'bmi', 'fit_type', 'size'])

# Convert to numeric where applicable - KEPT (used by other endpoints)
for col in ['waist_size', 'shoulder_length', 'chest', 'height', 'hips', 'bmi']:
    df_clothing[col] = pd.to_numeric(df_clothing[col])

# Scaling - KEPT (used by other endpoints)
scaler = StandardScaler()
numerical_cols = ['waist_size', 'shoulder_length', 'chest', 'height', 'hips', 'bmi']
df_clothing[numerical_cols] = scaler.fit_transform(df_clothing[numerical_cols])

# Fit KNN - KEPT (used by other endpoints)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(df_clothing[numerical_cols], df_clothing[['fit_type', 'size']])

# Training data for pants and t-shirt - KEPT (used by other endpoints)
training_data_pant = np.array([
    [70, 18], [75, 20], [80, 22], [85, 25], [90, 28],
    [65, 17], [78, 21], [82, 24], [88, 27], [92, 30],
    [72, 19], [76, 21], [84, 23], [89, 29], [95, 32],
    [68, 16], [74, 20], [79, 23], [86, 26], [91, 31],
    [67, 18], [73, 19], [81, 22], [87, 26], [94, 33],
    [66, 17], [77, 21], [83, 25], [90, 28], [96, 34]
])

# Fit labels for pants and t-shirt - KEPT (used by other endpoints)
fit_labels_pant = [
    "Slim", "Slim", "Regular", "Relaxed", "Baggy",
    "Slim", "Slim", "Regular", "Relaxed", "Baggy",
    "Slim", "Slim", "Regular", "Baggy", "Baggy",
    "Slim", "Slim", "Regular", "Relaxed", "Baggy",
    "Slim", "Slim", "Regular", "Relaxed", "Baggy",
    "Slim", "Regular", "Relaxed", "Baggy", "Baggy"
]

# Training data for t-shirt - KEPT (used by other endpoints)
training_data_tshirt = np.array([
    [70, 44, 170], [75, 46, 175], [80, 48, 180], [85, 50, 185], [90, 52, 190],
    [65, 42, 160], [78, 47, 172], [82, 49, 178], [88, 51, 183], [92, 53, 195],
    [72, 45, 168], [76, 46, 174], [84, 49, 179], [89, 52, 188], [95, 55, 200],
    [68, 43, 165], [74, 45, 170], [79, 47, 176], [86, 50, 182], [91, 53, 192],
    [67, 42, 162], [73, 44, 169], [81, 48, 177], [87, 51, 184], [94, 54, 198],
    [66, 41, 158], [77, 46, 173], [83, 49, 181], [90, 52, 187], [96, 56, 205]
])

# Fit labels for t-shirt - KEPT (used by other endpoints)
fit_labels_tshirt = [
    "Slim", "Regular", "Regular", "Relaxed", "Oversized",
    "Slim", "Regular", "Relaxed", "Relaxed", "Oversized",
    "Slim", "Regular", "Regular", "Oversized", "Oversized",
    "Slim", "Regular", "Regular", "Relaxed", "Oversized",
    "Slim", "Slim", "Regular", "Relaxed", "Oversized",
    "Slim", "Regular", "Relaxed", "Oversized", "Oversized"
]

# Initialize KNN models - KEPT (used by other endpoints)
knn_pant = KNeighborsClassifier(n_neighbors=3)
knn_pant.fit(training_data_pant, fit_labels_pant)

knn_tshirt = KNeighborsClassifier(n_neighbors=3)
knn_tshirt.fit(training_data_tshirt, fit_labels_tshirt)

# Initialize PolynomialFeatures and BayesianRidge - KEPT (used by other endpoints)
poly = PolynomialFeatures(degree=2)  # Choose an appropriate degree
heights = np.array([[150], [160], [170], [180], [190]])  # Example heights
poly.fit(heights)  # Fit the polynomial features

# Example usage (replace with your actual training data) - KEPT (used by other endpoints)
X_poly = poly.transform(heights)  # Transform heights to polynomial features
y = np.array([1, 2, 1, 3, 2])  # Example target values (body type?)

bayesian_model = BayesianRidge()
bayesian_model.fit(X_poly, y)

def get_predicted_size(height: float, weight: float, gender: str):
    pca_feature = pca.transform([[height, weight]])[0][0]
    predicted_weight = xgb_model.predict([[height, pca_feature]])[0]

    cluster = kmeans.predict([[height, weight]])[0]

    if cluster == 0:
        weight_adjustment = -5
    elif cluster == 1:
        weight_adjustment = 0
    else:
        weight_adjustment = +5

    adjusted_weight = weight + weight_adjustment

    for size, (min_w, max_w) in size_chart.items():
        if min_w <= adjusted_weight <= max_w:
            return size

    return "XXL" if adjusted_weight > 110 else "XS"

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

def get_predicted_size_height_gender(height: float, gender: str):
    height_poly = poly.transform([[height]])
    predicted_body_type = int(round(bayesian_model.predict(height_poly)[0]))

    # Assuming size_labels is a dictionary mapping size names to numerical labels
    # and xgb_model_height_gender is trained to predict these numerical labels
    predicted_size_label = xgb_model_height_gender.predict([[height, predicted_body_type]])[0]

    # Create an inverse mapping from numerical labels to size names
    size_inverse_mapping = {v: k for k, v in size_labels.items()}

    # Use the inverse mapping to get the predicted size name
    predicted_size = size_inverse_mapping[predicted_size_label]

    return predicted_size

@app.post("/get-size-height-gender", summary="Get ML-Powered Size Recommendation (Height + Gender Only)", tags=["Size Recommendation"])
def size_recommendation_height_gender(data: SizeRequestHeightGender):
    recommended_size = get_predicted_size_height_gender(data.height, data.gender)

    return {
        "height": data.height,
        "gender": data.gender,
        "recommended_size": recommended_size
    }

@app.post("/get-pant-fit", summary="Get Recommended Pant Fit", tags=["Pant Fit Recommendation"])
def get_pant_fit(data: PantFitRequest):
    bmi = data.weight / ((data.height / 100) ** 2)
    bmi_category = round(bmi)

    input_features = np.array([[data.waist_size, bmi_category]])
    predicted_fit = knn_pant.predict(input_features)[0]

    return {
        "recommended_pant_fit": predicted_fit
    }

@app.post("/get-fit", summary="Get Recommended Fit Type", tags=["Fit Recommendation"])
def get_fit(data: FitRequest):
    input_features = np.array([[data.waist_size, data.shoulder_length, data.height]])
    predicted_fit = knn_tshirt.predict(input_features)[0]

    return {
        "recommended_fit": predicted_fit
    }

@app.post("/recommend_clothing", response_model=ClothingRecommendation, summary="Recommend Clothing Fit and Size", tags=["Clothing Recommendation"])
def recommend_clothing(data: ClothingRequest):
    try:
        # Feature Engineering (BMI)
        bmi = data.weight / ((data.height / 100) ** 2)

        # Scale the input data
        input_features = np.array([data.waist_size, data.shoulder_length, data.chest, data.height, data.hips, bmi])
        input_scaled = scaler.transform(input_features.reshape(1, -1))

        # Predict using KNN
        predicted_fit_size = knn.predict(input_scaled)[0]
        top_recommended_fit = predicted_fit_size[0]  # Top fit
        top_recommended_size = predicted_fit_size[1] # Top size

        pant_recommended_fit = predicted_fit_size[0]  # Pant fit
        pant_recommended_size = predicted_fit_size[1] # Pant Size

        return ClothingRecommendation(
            top_recommended_fit=top_recommended_fit,
            top_recommended_size=top_recommended_size,
            pant_recommended_fit=pant_recommended_fit,
            pant_recommended_size=pant_recommended_size
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Replace original adjust_size function with the corrected version
def adjust_size(size: str, preferred_fit: str, recommended_fit: str):
    """Adjusts size based on fit preference with bidirectional logic"""
    if size not in size_chart_adjust:
        raise HTTPException(status_code=400, detail="Invalid size input")

    # Get current index in size chart
    current_index = size_chart_adjust.index(size)

    # Determine size shift based on *preferred_fit*, not recommended_fit
    if preferred_fit in fit_adjustments and recommended_fit in fit_adjustments[preferred_fit]:
        size_shift = fit_adjustments[preferred_fit][recommended_fit]
    elif recommended_fit in reverse_fit_adjustments and preferred_fit in reverse_fit_adjustments[recommended_fit]:
        size_shift = reverse_fit_adjustments[recommended_fit][preferred_fit]
    else:
        raise HTTPException(status_code=400, detail="Invalid fit conversion")

    # Apply size shift and ensure it stays within valid range
    new_index = max(0, min(len(size_chart_adjust) - 1, current_index + size_shift))

    return {"preferred_fit": preferred_fit, "adjusted_size": size_chart_adjust[new_index]}

@app.post("/choosefit/")
def choose_fit(request: FitRequestAdjust):
    """Returns adjusted size based on user's fit preference"""
    return adjust_size(request.size, request.preferred_fit, request.recommended_fit)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main2:app", host="127.0.0.1", port=8001, reload=True)
