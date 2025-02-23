from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

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

# 🔹 Synthetic Training Data (For Body Type Classification)
np.random.seed(42)
heights_data = np.random.randint(140, 190, 500)
weights_data = np.random.randint(40, 120, 500)

df = pd.DataFrame({'height': heights_data, 'weight': weights_data})

# K-Means Clustering to Detect Body Types
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['body_type'] = kmeans.fit_predict(df[['height', 'weight']])

# PCA for Feature Reduction
pca = PCA(n_components=1)
df['pca_feature'] = pca.fit_transform(df[['height', 'weight']])

# XGBoost Model to Predict Weight Range per Height
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
df_for_xgb = df[['height', 'pca_feature']]
xgb_model.fit(df_for_xgb, df['weight'])

# Indian Height-Weight Size Chart
size_chart = {
    "XS": (40, 55),
    "S": (56, 65),
    "M": (66, 75),
    "L": (76, 85),
    "XL": (86, 95),
    "XXL": (96, 110),
}

# Training data for clothing recommendation
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

# Fit labels for clothing recommendation
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

# Size labels for clothing recommendation
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

# Truncate labels to match training data length
fit_labels = fit_labels[:len(training_data)]
size_labels = size_labels[:len(training_data)]

# Size chart for size adjustments
size_chart_adjust = ["XS", "S", "M", "L", "XL", "XXL"]

# Fit adjustment logic (Bidirectional)
fit_adjustments = {
    "Relaxed": {"Slim": 1, "Regular": 0, "Oversized": -1},
    "Slim": {"Relaxed": -1, "Regular": 0, "Skinny": 1},
    "Regular": {"Relaxed": 0, "Slim": 0, "Oversized": 1},
    "Oversized": {"Regular": -1, "Slim": -2},
    "Skinny": {"Regular": 1, "Slim": 0}
}

# Reverse fit adjustments
reverse_fit_adjustments = {
    "Slim": {"Relaxed": -1, "Regular": 0, "Oversized": 2},
    "Relaxed": {"Slim": 1, "Regular": 0, "Skinny": -1},
    "Regular": {"Slim": 0, "Relaxed": 0, "Oversized": -1},
    "Oversized": {"Regular": 1, "Slim": 2},
    "Skinny": {"Regular": -1, "Slim": 0}
}

# Initialize models
combined_data = np.concatenate((training_data, np.array(fit_labels)[:, None], np.array(size_labels)[:, None]), axis=1)
df_clothing = pd.DataFrame(combined_data, columns=['waist_size', 'shoulder_length', 'chest', 'height', 'hips', 'bmi', 'fit_type', 'size'])

# Convert to numeric where applicable
for col in ['waist_size', 'shoulder_length', 'chest', 'height', 'hips', 'bmi']:
    df_clothing[col] = pd.to_numeric(df_clothing[col])

# Scaling
scaler = StandardScaler()
numerical_cols = ['waist_size', 'shoulder_length', 'chest', 'height', 'hips', 'bmi']
df_clothing[numerical_cols] = scaler.fit_transform(df_clothing[numerical_cols])

# Fit KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Correctly fit KNN with both 'fit_type' and 'size'
knn.fit(df_clothing[numerical_cols], df_clothing[['fit_type', 'size']].values.tolist())

# Initialize PolynomialFeatures and BayesianRidge
poly = PolynomialFeatures(degree=2)  # Choose an appropriate degree
heights = np.array([[150], [160], [170], [180], [190]])  # Example heights
poly.fit(heights)  # Fit the polynomial features

# Example usage (replace with your actual training data)
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

@app.route("/get-size", methods=['POST'])
def size_recommendation():
    try:
        data = request.get_json()
        size_request = SizeRequest(**data)
        recommended_size = get_predicted_size(size_request.height, size_request.weight, size_request.gender)

        return jsonify({
            "height": size_request.height,
            "weight": size_request.weight,
            "age": size_request.age,
            "gender": size_request.gender,
            "recommended_size": recommended_size
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Updated `get_predicted_size_height_gender` Function
size_mapping_male = {
    range(140,150): "XS",
    range(150,160): "S",
    range(160,170): "M",
    range(170,180): "L",
    range(180,210): "XL"  #Setting default height
}

size_mapping_female = {
    range(140,150): "XS",
    range(150,160): "S",
    range(160,170): "M",
    range(170,180): "L",
    range(180,210): "XL" #Setting default height
}

def get_predicted_size_height_gender(height: float, gender: str):
    """Predicts size based on height and gender using the size mappings."""

    if gender == "male":
        for height_range, size in size_mapping_male.items():
            if int(height) in height_range:
                return size
    elif gender == "female":
        for height_range, size in size_mapping_female.items():
            if int(height) in height_range:
                return size
    return "M"  # Default if no mapping is found

@app.route("/get-size", methods=['POST'])
def get_size():
    try:
        data = request.get_json()
        height = data.get("height")
        gender = data.get("gender")

        if not height or not isinstance(height, (int, float)):
            return jsonify({"error": "Invalid height"}), 400

        if height < 155:
            size = "XS"
        elif 155 <= height <= 165:
            size = "S"
        elif 166 <= height <= 175:
            size = "M"
        elif 176 <= height <= 185:
            size = "L"
        else:
            size = "XL"

        return jsonify({
            "height": height,
            "gender": gender,
            "recommended_size": size
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get-pant-fit", methods=['POST'])
def get_pant_fit():
    try:
        data = request.get_json()
        pant_fit_request = PantFitRequest(**data)
        bmi = pant_fit_request.weight / ((pant_fit_request.height / 100) ** 2)
        bmi_category = round(bmi)

        input_features = np.array([[pant_fit_request.waist_size, bmi_category]])
        predicted_fit = knn_pant.predict(input_features)[0]

        return jsonify({
            "recommended_pant_fit": predicted_fit
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get-fit", methods=['POST'])
def get_fit():
    try:
        data = request.get_json()
        fit_request = FitRequest(**data)
        input_features = np.array([[fit_request.waist_size, fit_request.shoulder_length, fit_request.height]])
        predicted_fit = knn_tshirt.predict(input_features)[0]

        return jsonify({
            "recommended_fit": predicted_fit
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/recommend_clothing", methods=['POST'])
def recommend_clothing():
    try:
        data = request.get_json()
        clothing_request = ClothingRequest(**data)

        # Feature Engineering (BMI)
        bmi = clothing_request.weight / ((clothing_request.height / 100) ** 2)

        # Scale the input data
        input_features = np.array([clothing_request.waist_size, clothing_request.shoulder_length, clothing_request.chest, clothing_request.height, clothing_request.hips, bmi])
        input_scaled = scaler.transform(input_features.reshape(1, -1))

        # Predict using KNN
        # Reshape input_scaled to (1, -1) if it's not already 2D
        if input_scaled.ndim == 1:
            input_scaled = input_scaled.reshape(1, -1)

        predicted_fit_size = knn.predict(input_scaled)

        # Check if predicted_fit_size is not empty before accessing elements
        if predicted_fit_size.size > 0:
            # Flatten the prediction result
            predicted_fit_size = predicted_fit_size.flatten()
            top_recommended_fit = predicted_fit_size[0]  # Top fit
            top_recommended_size = predicted_fit_size[1]  # Top size
        else:
            # Handle the case where the prediction result is empty
            top_recommended_fit = "N/A"
            top_recommended_size = "N/A"


        pant_recommended_fit = top_recommended_fit  # Pant fit
        pant_recommended_size = top_recommended_size # Pant Size

        return jsonify({
            "top_recommended_fit": top_recommended_fit,
            "top_recommended_size": top_recommended_size,
            "pant_recommended_fit": pant_recommended_fit,
            "pant_recommended_size": pant_recommended_size
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Replace original adjust_size function with the corrected version
def adjust_size(size: str, preferred_fit: str, recommended_fit: str):
    """Adjusts size based on fit preference with bidirectional logic"""
    if size not in size_chart_adjust:
        return jsonify({"error": "Invalid size input"}), 400

    # Get current index in size chart
    current_index = size_chart_adjust.index(size)

    # Determine size shift based on *preferred_fit*, not recommended_fit
    if preferred_fit in fit_adjustments and recommended_fit in fit_adjustments[preferred_fit]:
        size_shift = fit_adjustments[preferred_fit][recommended_fit]
    elif preferred_fit in reverse_fit_adjustments and recommended_fit in reverse_fit_adjustments[preferred_fit]:
        size_shift = reverse_fit_adjustments[preferred_fit][recommended_fit]
    else:
        size_shift = 0  # No adjustment if fit types are incompatible

    # Calculate new index, ensuring it stays within bounds
    new_index = max(0, min(current_index + size_shift, len(size_chart_adjust) - 1))

    return size_chart_adjust[new_index]

@app.route("/adjust_size", methods=['POST'])
def adjust_size_route():
    try:
        data = request.get_json()
        size = data.get('size')
        preferred_fit = data.get('preferred_fit')
        recommended_fit = data.get('recommended_fit')

        adjusted_size = adjust_size(size, preferred_fit, recommended_fit)
        return jsonify({"adjusted_size": adjusted_size})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8001)
