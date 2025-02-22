from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Define request model
class FitRequest(BaseModel):
    size: str  # Current size
    preferred_fit: str  # User's chosen fit
    recommended_fit: str  # Fit given by system

# Size mapping logic
size_chart = ["XS", "S", "M", "L", "XL", "XXL"]

# Fit adjustment logic (Bidirectional)
fit_adjustments = {
    "Relaxed": {"Slim": 1, "Regular": 0, "Oversized": -1},
    "Slim": {"Relaxed": -1, "Regular": 0, "Skinny": 1},
    "Regular": {"Relaxed": 0, "Slim": 0, "Oversized": 1},
    "Oversized": {"Regular": -1, "Slim": -2},
    "Skinny": {"Regular": 1, "Slim": 0}
}

# Reverse fit adjustments (for bidirectional lookup)
reverse_fit_adjustments = {
    "Slim": {"Relaxed": -1, "Regular": 0, "Oversized": 2},
    "Relaxed": {"Slim": 1, "Regular": 0, "Skinny": -1},
    "Regular": {"Slim": 0, "Relaxed": 0, "Oversized": -1},
    "Oversized": {"Regular": 1, "Slim": 2},
    "Skinny": {"Regular": -1, "Slim": 0}
}

def adjust_size(size: str, preferred_fit: str, recommended_fit: str):
    """Adjusts size based on fit preference with bidirectional logic"""
    if size not in size_chart:
        raise HTTPException(status_code=400, detail="Invalid size input")

    # Get current index in size chart
    current_index = size_chart.index(size)

    # Determine size shift
    if recommended_fit in fit_adjustments and preferred_fit in fit_adjustments[recommended_fit]:
        size_shift = fit_adjustments[recommended_fit][preferred_fit]
    elif preferred_fit in reverse_fit_adjustments and recommended_fit in reverse_fit_adjustments[preferred_fit]:
        size_shift = reverse_fit_adjustments[preferred_fit][recommended_fit]
    else:
        raise HTTPException(status_code=400, detail="Invalid fit conversion")

    # Apply size shift and ensure it stays within valid range
    new_index = max(0, min(len(size_chart) - 1, current_index + size_shift))

    return {"preferred_fit": preferred_fit, "adjusted_size": size_chart[new_index]}

@app.post("/choosefit/")
def choose_fit(request: FitRequest):
    """Returns adjusted size based on user's fit preference"""
    return adjust_size(request.size, request.preferred_fit, request.recommended_fit)

# Run Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("choosefit:app", host="127.0.0.1", port=8001, reload=True)
