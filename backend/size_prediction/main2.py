from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import BayesianRidge
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

app = FastAPI()

# ✅ Define input model for general size recommendation (height, weight, age, gender)
class SizeRequest(BaseModel):
    height: float = Field(..., example=170.0, description="Height in cm")
    weight: float = Field(..., example=70.0, description="Weight in kg")
    age: int = Field(..., example=25, description="Age in years")
    gender: str = Field(..., example="male", description="Gender (male/female)")

# ✅ Define input model for height and gender size recommendation
class SizeRequestHeightGender(BaseModel):
    height: float = Field(..., example=170.0, description="Height in cm")
    gender: str = Field(..., example="male", description="Gender (male/female)")

# ✅ Define input model for pant fit recommendation
class PantFitRequest(BaseModel):
    waist_size: float = Field(..., example=80, description="Waist size in cm")
    height: float = Field(..., example=180, description="Height in cm")
    weight: float = Field(..., example=75, description="Weight in kg")

# ✅ Define input model for T-shirt fit recommendation
class FitRequest(BaseModel):
    waist_size: float = Field(..., example=80, description="Waist size in cm")
    shoulder_length: float = Field(..., example=48, description="Shoulder length in cm")
    height: float = Field(..., example=180, description="Height in cm")

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

#-----------------------------------------------------------------------------------------------------------------------
#Second App
np.random.seed(42)
heights = np.random.randint(140, 190, 500)  
genders = np.random.choice(["male", "female"], 500) 

size_mapping_male = {140: "XS", 150: "S", 160: "M", 170: "L", 180: "XL"}
size_mapping_female = {140: "XS", 150: "S", 160: "M", 170: "L", 180: "XL"}

df = pd.DataFrame({"height": heights, "gender": genders})
df["size"] = df.apply(lambda row: size_mapping_male.get(row["height"], "M") if row["gender"] == "male" else size_mapping_female.get(row["height"], "M"), axis=1)


gmm = GaussianMixture(n_components=3, random_state=42)
df["body_type"] = gmm.fit_predict(df[["height"]])


poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(df[["height"]])

bayesian_model = BayesianRidge()
bayesian_model.fit(X_poly, df["body_type"])  

size_labels = {"XS": 0, "S": 1, "M": 2, "L": 3, "XL": 4, "XXL": 5}
df["size_label"] = df["size"].map(size_labels)

xgb_model_height_gender = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model_height_gender.fit(df[["height", "body_type"]], df["size_label"])


def get_predicted_size_height_gender(height: float, gender: str):

    height_poly = poly.transform([[height]])
    predicted_body_type = int(round(bayesian_model.predict(height_poly)[0]))

    predicted_size_label = xgb_model_height_gender.predict([[height, predicted_body_type]])[0]
    size_inverse_mapping = {v: k for k, v in size_labels.items()}
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

#-----------------------------------------------------------------------------------------------------------------------
#Third App

training_data = np.array([
    [70, 18], [75, 20], [80, 22], [85, 25], [90, 28],
    [65, 17], [78, 21], [82, 24], [88, 27], [92, 30],
    [72, 19], [76, 21], [84, 23], [89, 29], [95, 32],
    [68, 16], [74, 20], [79, 23], [86, 26], [91, 31],
    [67, 18], [73, 19], [81, 22], [87, 26], [94, 33],
    [66, 17], [77, 21], [83, 25], [90, 28], [96, 34]
])

fit_labels = [
    "Slim", "Slim", "Regular", "Relaxed", "Baggy",
    "Slim", "Slim", "Regular", "Relaxed", "Baggy",
    "Slim", "Slim", "Regular", "Baggy", "Baggy",
    "Slim", "Slim", "Regular", "Relaxed", "Baggy",
    "Slim", "Slim", "Regular", "Relaxed", "Baggy",
    "Slim", "Regular", "Relaxed", "Baggy", "Baggy"
]

knn_pant = KNeighborsClassifier(n_neighbors=3)
knn_pant.fit(training_data, fit_labels)

@app.post("/get-pant-fit", summary="Get Recommended Pant Fit", tags=["Pant Fit Recommendation"])
def get_pant_fit(data: PantFitRequest):
  
    bmi = data.weight / ((data.height / 100) ** 2)
    
    bmi_category = round(bmi)


    input_features = np.array([[data.waist_size, bmi_category]])
    predicted_fit = knn_pant.predict(input_features)[0]

    return {
        "recommended_pant_fit": predicted_fit
    }

#-----------------------------------------------------------------------------------------------------------------------
#Fourth App

training_data_tshirt = np.array([
    [70, 44, 170], [75, 46, 175], [80, 48, 180], [85, 50, 185], [90, 52, 190],
    [65, 42, 160], [78, 47, 172], [82, 49, 178], [88, 51, 183], [92, 53, 195],
    [72, 45, 168], [76, 46, 174], [84, 49, 179], [89, 52, 188], [95, 55, 200],
    [68, 43, 165], [74, 45, 170], [79, 47, 176], [86, 50, 182], [91, 53, 192],
    [67, 42, 162], [73, 44, 169], [81, 48, 177], [87, 51, 184], [94, 54, 198],
    [66, 41, 158], [77, 46, 173], [83, 49, 181], [90, 52, 187], [96, 56, 205]
])

fit_labels_tshirt = [
    "Slim", "Regular", "Regular", "Relaxed", "Oversized",
    "Slim", "Regular", "Relaxed", "Relaxed", "Oversized",
    "Slim", "Regular", "Regular", "Oversized", "Oversized",
    "Slim", "Regular", "Regular", "Relaxed", "Oversized",
    "Slim", "Slim", "Regular", "Relaxed", "Oversized",
    "Slim", "Regular", "Relaxed", "Oversized", "Oversized"
]

knn_tshirt = KNeighborsClassifier(n_neighbors=3)
knn_tshirt.fit(training_data_tshirt, fit_labels_tshirt)

@app.post("/get-fit", summary="Get Recommended Fit Type", tags=["Fit Recommendation"])
def get_fit(data: FitRequest):
    input_features = np.array([[data.waist_size, data.shoulder_length, data.height]])
    predicted_fit = knn_tshirt.predict(input_features)[0]

    return {
        "recommended_fit": predicted_fit
    }


