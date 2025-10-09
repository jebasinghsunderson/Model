from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import json 
import requests
from pydantic import BaseModel
import pickle

class CropInput(BaseModel):
    Fertilizer: float
    Pesticide: float
   # Annual_Rainfall: float
    Area: float
    #Crop: str
    Season: str
    State: str
    Crop_Year: int
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    Moisture: float
    Soil_Type: str

app = FastAPI()

# Enable CORS so frontend (different port) can access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],    
)
#Production model
# --- Load artifacts ---
model = joblib.load("model.pkl")         # your trained model
scaler_X = joblib.load("scaler_x.pkl")   # scaler for features
#scaler_y = joblib.load("scaler_y.pkl")   # scaler for target
ohe = joblib.load("ohe_features.pkl")             # fitted OneHotEncoder
model_columns = joblib.load("model_columns.pkl")

# Columns
numeric_cols = ['Pesticide', 'Fertilizer', 'Area', 'Annual_Rainfall']
categorical_cols =['State', 'Crop', 'Season']

# Crop suggestion model

crop_model = joblib.load("crop_model.pkl")
crop_scaler = joblib.load("crop_scaler.pkl")
crop_le = joblib.load("crop_label_encoder.pkl")

with open("crop_features.json", "r") as f:
    crop_feature_list = json.load(f)

# Fertilizer prediction model

with open("fertilizer_model.pkl", "rb") as f:
    fertilizer_model = pickle.load(f)

with open("fertilizer_scaler.pkl", "rb") as f:
    fertilizer_scaler = pickle.load(f)

soil_le = pickle.load(open("Soil_Type_encoder.pkl", "rb"))
crop_le = pickle.load(open("Crop_Type_encoder.pkl", "rb"))
target_le = pickle.load(open("fertilizer_target_encoder.pkl", "rb"))


# Alternatively, manually save the list from training
@app.post("/predict")
async def predict(data: CropInput):
    try:
        # -----------------------------
        # 🌾 STEP 1: Crop prediction
        # -----------------------------
        input_df = pd.DataFrame([data.dict()])[crop_feature_list]
        input_scaled = crop_scaler.transform(input_df)
        pred_label = crop_model.predict(input_scaled)
        pred_crop = crop_le.inverse_transform(pred_label)

        # -----------------------------
        # 🌧️ STEP 2: Production prediction
        # -----------------------------
        df = pd.DataFrame([data.dict()])
        df["Crop"] = pred_crop[0]
        df["Annual_Rainfall"] = data.rainfall

        # Clean column names
        df.columns = df.columns.str.strip()

        # Log-transform numeric features
        log_cols = [col + "_log" for col in numeric_cols]
        for col in numeric_cols:
            df[col + "_log"] = np.log1p(df[col])

        area = np.expm1(df["Area_log"])

        # Scale numeric
        df_scaled = scaler_X.transform(df[log_cols])
        df_scaled = pd.DataFrame(
            df_scaled,
            columns=[col.replace("_log", "_scaled") for col in log_cols],
            index=df.index
        )

        df = df.drop(columns=log_cols + numeric_cols)
        df = pd.concat([df, df_scaled], axis=1)

        # Encode categorical
        for col in categorical_cols:
            df[col] = df[col].astype(str).str.strip()

        ohe_array = ohe.transform(df[categorical_cols])
        ohe_cols = ohe.get_feature_names_out(categorical_cols)
        ohe_df = pd.DataFrame(ohe_array, columns=ohe_cols, index=df.index)

        df = df.drop(columns=categorical_cols)
        df = pd.concat([df, ohe_df], axis=1)

        df.columns = df.columns.str.strip()
        model_cols = [c.strip() for c in model.feature_names_in_]
        df = df.reindex(columns=model_cols, fill_value=0)
        df = df.astype(np.float64)

        # Predict production
        y_pred_log = model.predict(df)
        y_pred_original = np.expm1(y_pred_log)

        # -----------------------------
        # 🧪 STEP 3: Fertilizer prediction
        # -----------------------------
        # Encode categorical first
        soil_encoded = int(soil_le.transform([data.Soil_Type.strip()])[0])
        crop_encoded = int(crop_le.transform([pred_crop[0].strip()])[0])

        # Combine numeric + encoded categorical
        num_values = np.array([
            data.temperature,
            data.humidity,
            data.Moisture,
            data.N,
            data.P,
            data.K
        ]).reshape(1, -1)

        num_scaled = fertilizer_scaler.transform(num_values)
        X_input = np.hstack([num_scaled, [[soil_encoded, crop_encoded]]])

        fertilizer_pred_encoded = fertilizer_model.predict(X_input)[0]
        pred_fertilizer = target_le.inverse_transform([fertilizer_pred_encoded])[0]

        # -----------------------------
        # 🎯 Final output
        # -----------------------------
        return {
            "predicted_crop": pred_crop[0],
            "predicted_fertilizer": pred_fertilizer,
            "prediction_scaled": f"{float(y_pred_original[0]):.2f}",
            "Yield": f"{float((y_pred_original[0]) / area):.2f}"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}