from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

app = FastAPI()


training_data = np.array([
    [70, 44, 170], [75, 46, 175], [80, 48, 180], [85, 50, 185], [90, 52, 190],
    [65, 42, 160], [78, 47, 172], [82, 49, 178], [88, 51, 183], [92, 53, 195],
    [72, 45, 168], [76, 46, 174], [84, 49, 179], [89, 52, 188], [95, 55, 200],
    [68, 43, 165], [74, 45, 170], [79, 47, 176], [86, 50, 182], [91, 53, 192],
    [67, 42, 162], [73, 44, 169], [81, 48, 177], [87, 51, 184], [94, 54, 198],
    [66, 41, 158], [77, 46, 173], [83, 49, 181], [90, 52, 187], [96, 56, 205]
])

fit_labels = [
    "Slim", "Regular", "Regular", "Relaxed", "Oversized",
    "Slim", "Regular", "Relaxed", "Relaxed", "Oversized",
    "Slim", "Regular", "Regular", "Oversized", "Oversized",
    "Slim", "Regular", "Regular", "Relaxed", "Oversized",
    "Slim", "Slim", "Regular", "Relaxed", "Oversized",
    "Slim", "Regular", "Relaxed", "Oversized", "Oversized"
]


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(training_data, fit_labels)


class FitRequest(BaseModel):
    waist_size: float = Field(..., example=80, description="Waist size in cm")
    shoulder_length: float = Field(..., example=48, description="Shoulder length in cm")
    height: float = Field(..., example=180, description="Height in cm")


@app.post("/get-fit", summary="Get Recommended Fit Type", tags=["Fit Recommendation"])
def get_fit(data: FitRequest):
    input_features = np.array([[data.waist_size, data.shoulder_length, data.height]])
    predicted_fit = knn.predict(input_features)[0]

    return {
        "recommended_fit": predicted_fit
    }
