from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# ✅ Data Models
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

# ✅ Training Data (Combined and Adjusted for Indian Users)
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

    # Indian Specific adjustments - more 'Regular' fits
    [71, 44, 91, 171, 93, 24],  # Regular, S
    [74, 45, 94, 172, 95, 25],  # Regular, M
    [82, 48, 102, 178, 102, 26], # Regular, L

    # More "Baggy" for very high waist or high BMI
    [93, 54, 113, 196, 113, 28], # Baggy, XXL
    [97, 57, 117, 207, 115, 29]  # Baggy, XXL
])

# Fit labels corresponding to training data
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

# Size labels corresponding to training data - XS, S, M, L, XL, XXL
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

# Combine features
combined_data = np.concatenate((training_data, np.array(fit_labels)[:, None], np.array(size_labels)[:, None]), axis=1)
df = pd.DataFrame(combined_data, columns=['waist_size', 'shoulder_length', 'chest', 'height', 'hips', 'bmi', 'fit_type', 'size'])

# Convert to numeric where applicable
for col in ['waist_size', 'shoulder_length', 'chest', 'height', 'hips', 'bmi']:
    df[col] = pd.to_numeric(df[col])

# Scaling
scaler = StandardScaler()
numerical_cols = ['waist_size', 'shoulder_length', 'chest', 'height', 'hips', 'bmi']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Fit KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(df[numerical_cols], df[['fit_type', 'size']])

# ✅ API Endpoint
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
