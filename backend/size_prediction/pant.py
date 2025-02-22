from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

app = FastAPI()


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

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(training_data, fit_labels)


class PantFitRequest(BaseModel):
    waist_size: float = Field(..., example=80, description="Waist size in cm")
    height: float = Field(..., example=180, description="Height in cm")
    weight: float = Field(..., example=75, description="Weight in kg")


@app.post("/get-pant-fit", summary="Get Recommended Pant Fit", tags=["Pant Fit Recommendation"])
def get_pant_fit(data: PantFitRequest):
  
    bmi = data.weight / ((data.height / 100) ** 2)
    
    bmi_category = round(bmi)


    input_features = np.array([[data.waist_size, bmi_category]])
    predicted_fit = knn.predict(input_features)[0]

    return {
        "recommended_pant_fit": predicted_fit
    }
