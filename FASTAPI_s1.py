from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import pandas as pd
import joblib

# Define amplify_prediction function
def amplify_prediction(region, month, base_prediction):
    if region == "north" and month in [12, 1, 2]:  # Winter in North
        return base_prediction * 1.3  # Amplify by 30%
    elif region == "south" and month in [6, 7, 8]:  # Summer in South
        return base_prediction * 1.5  # Amplify by 50%
    elif region == "central" and month in [12, 1, 2, 6, 7, 8]:  # Both seasons in Central
        return base_prediction * 1.5  # Amplify by 50%
    else:
        return base_prediction

# Define predict_next_consumption function
def predict_next_consumption(model, scaler, region, hour, day, month, consumption):
    # Scale the input features using the provided scaler
    input_data = pd.DataFrame({
        'hour': [hour],
        'day': [day],
        'month': [month],
        'consumption': [consumption]
    })
    scaled_data = scaler.transform(input_data)

    # Make base prediction using the LightGBM model
    base_prediction = model.predict(scaled_data)[0]

    # Apply amplification logic based on region and seasonality
    amplified_prediction = amplify_prediction(region, month, base_prediction)

    return amplified_prediction

# Load the saved model and scaler (only these two are saved)
try:
    model, scaler = joblib.load('consumption_prediction_model_lgb.joblib')
except FileNotFoundError:
    raise RuntimeError("The model file 'consumption_prediction_model_lgb.joblib' was not found. Please train and save the model first.")

# Create FastAPI instance
app = FastAPI()

@app.get("/api/predict_next_consumption")
async def predict_next_consumption_api(
    region: str = Query(..., description="Region (e.g., 'north', 'south', 'central')"),
    hour: int = Query(..., ge=0, le=23, description="Current hour (0-23)"),
    day: int = Query(..., ge=1, le=31, description="Current day of the month (1-31)"),
    month: int = Query(..., ge=1, le=12, description="Current month (1-12)"),
    consumption: float = Query(..., description="Current energy consumption value")
):
    """
    Predict the next hour's energy consumption based on query parameters.
    
    Parameters:
        - region: Region (e.g., "north", "south", "central")
        - hour: Current hour (0-23)
        - day: Current day of the month (1-31)
        - month: Current month (1-12)
        - consumption: Current energy consumption value
    
    Returns:
        - Predicted next hour energy consumption (amplified based on region and seasonality)
    """
    
    # Validate region input
    valid_regions = ["north", "south", "central"]
    if region.lower() not in valid_regions:
        raise HTTPException(status_code=400, detail=f"Invalid region '{region}'. Valid regions are {valid_regions}.")
    
    try:
        # Use predict_next_consumption function to predict next-hour consumption dynamically.
        prediction = predict_next_consumption(
            model=model,
            scaler=scaler,
            region=region.lower(),
            hour=hour,
            day=day,
            month=month,
            consumption=consumption,
        )
        
        return {
            "region": region,
            "hour": hour,
            "day": day,
            "month": month,
            "current_consumption": consumption,
            "predicted_next_hour_consumption": round(prediction, 2),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)