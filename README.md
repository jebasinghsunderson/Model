# Agricultural ML Model

An integrated machine learning system for agricultural decision support combining three predictive models to optimize crop production planning and resource management.

## Overview

This repository contains three machine learning models combined into a single FastAPI service:

1. **Crop Prediction Model** - Recommends the most suitable crop based on environmental and soil conditions
2. **Yield Prediction Model** - Predicts agricultural production based on various input parameters
3. **Fertilizer Recommendation Model** - Suggests appropriate fertilizer type based on soil and crop characteristics

## Features

- Multi-model prediction pipeline serving crop selection, yield forecasting, and fertilizer recommendations
- FastAPI-based REST API for seamless integration
- CORS enabled for cross-domain frontend integration
- Trained models with pre-fitted scalers and encoders for production-ready predictions
- Handles categorical encoding and feature scaling automatically
- Logarithmic transformation for improved yield predictions

## Project Structure

```
├── main.py                           # FastAPI application with prediction endpoints
├── requirements.txt                  # Python dependencies
├── crop_features.json               # Feature list for crop prediction model
│
├── Models (Pre-trained)
│   ├── crop_model.pkl              # Crop recommendation model
│   ├── model.pkl                   # Yield/production prediction model
│   └── fertilizer_model.pkl        # Fertilizer recommendation model
│
├── Scalers
│   ├── crop_scaler.pkl             # Feature scaler for crop model
│   ├── scaler_x.pkl                # Feature scaler for yield model
│   ├── scaler_y.pkl                # Target scaler for yield model
│   ├── minmax_scaler.pkl           # Additional scaling utility
│   └── fertilizer_scaler.pkl       # Feature scaler for fertilizer model
│
├── Encoders
│   ├── crop_label_encoder.pkl      # Crop label encoder
│   ├── Crop_Type_encoder.pkl       # Crop type categorical encoder
│   ├── Soil_Type_encoder.pkl       # Soil type categorical encoder
│   ├── fertilizer_target_encoder.pkl # Fertilizer type encoder
│   ├── ohe.pkl                     # One-hot encoder for categorical features
│   └── ohe_features.pkl            # OHE feature list
│
└── model_columns.pkl               # Required feature columns for yield model
```

## Dependencies

```
fastapi
uvicorn[standard]
pandas
numpy
joblib
scikit-learn
requests
pydantic
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Input Parameters

The API accepts the following input parameters for predictions:

### Numeric Inputs
- `Fertilizer` (float) - Amount of fertilizer used
- `Pesticide` (float) - Amount of pesticide used
- `Area` (float) - Agricultural area in hectares
- `Crop_Year` (int) - Year of cultivation
- `N` (float) - Nitrogen content in soil
- `P` (float) - Phosphorus content in soil
- `K` (float) - Potassium content in soil
- `temperature` (float) - Average temperature in Celsius
- `humidity` (float) - Average humidity percentage
- `ph` (float) - Soil pH value
- `rainfall` (float) - Annual rainfall in mm
- `Moisture` (float) - Soil moisture percentage

### Categorical Inputs
- `Soil_Type` (str) - Type of soil
- `Season` (str) - Season of cultivation
- `State` (str) - Geographic state/region

## API Usage

### Starting the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Prediction Endpoint

**POST** `/predict`

Request body example:

```json
{
  "Fertilizer": 120.0,
  "Pesticide": 50.0,
  "Area": 2.5,
  "Season": "Kharif",
  "State": "Maharashtra",
  "Crop_Year": 2024,
  "N": 45.0,
  "P": 30.0,
  "K": 40.0,
  "temperature": 28.5,
  "humidity": 65.0,
  "ph": 6.8,
  "rainfall": 800.0,
  "Moisture": 55.0,
  "Soil_Type": "Loamy"
}
```

Response example:

```json
{
  "predicted_crop": "Rice",
  "predicted_fertilizer": "NPK 10-26-26",
  "prediction_scaled": "4500.75",
  "Yield": "1800.30"
}
```

## Model Pipeline

### Step 1: Crop Prediction
- Input features: N, P, K, temperature, humidity, ph, rainfall
- Output: Recommended crop type
- Uses the crop prediction model with feature scaling

### Step 2: Production/Yield Prediction
- Uses the predicted crop from Step 1
- Applies log transformation to numeric features
- Scales features using MinMaxScaler
- One-hot encodes categorical variables (State, Crop, Season)
- Predicts production yield with inverse log transformation
- Calculates yield per hectare

### Step 3: Fertilizer Recommendation
- Uses soil type and predicted crop
- Combines with NPK values, temperature, humidity, and moisture
- Scales numerical inputs
- Recommends appropriate fertilizer type

## Technical Details

### Data Preprocessing
- **Numeric Features**: Log transformation followed by MinMax scaling
- **Categorical Features**: One-hot encoding for State, Crop, and Season
- **Missing Values**: Annual rainfall derived from input rainfall parameter

### Model Characteristics
- All models are pre-trained and serialized using joblib/pickle
- Feature consistency enforced through model_columns.pkl
- Column names automatically stripped of whitespace
- Handles edge cases with reindexing and zero-filling for missing features

## Requirements for Production

- Ensure all .pkl files are in the same directory as main.py
- crop_features.json must be present
- Update CORS allowed origins in production (`allow_origins=["*"]` should be restricted)
- Consider adding authentication/authorization layer
- Implement input validation and error handling for edge cases

## Error Handling

The API returns an error response if prediction fails:

```json
{
  "error": "Error description"
}
```

Common issues:
- Missing or invalid input parameters
- Corrupted model files
- Unsupported soil type or state values

## Notes

- The `Annual_Rainfall` parameter is automatically populated from the `rainfall` input
- Crop year is stored but not directly used in predictions
- Area information is critical for yield per hectare calculations
- All predictions assume standard agricultural practices

## License

This project is for educational and agricultural research purposes.

## Author

Jebasingh Sunderson
