from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

app = FastAPI(
    title="Industrial Predictive Maintenance API",
    description="AI-based machine failure detection and fault diagnosis",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load XGBoost models
failure_detection_model = xgb.XGBClassifier()
failure_detection_model.load_model("failure_detection_model.json")

failure_diagnosis_model = xgb.XGBClassifier()
failure_diagnosis_model.load_model("failure_diagnosis_model.json")

# Load existing files
# Load preprocessors and supporting files
failure_preprocessor = joblib.load("failure_preprocessor.pkl")
diagnosis_preprocessor = joblib.load("diagnosis_preprocessor.pkl")

label_encoder = joblib.load("failure_label_encoder.pkl")
normal_stats = joblib.load("normal_stats.pkl")


sensor_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


class MachineData(BaseModel):
    machine_type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


def calculate_deviation_score(machine_data):

    deviations = []

    for feature in sensor_features:

        value = machine_data[feature]

        median = normal_stats["median"][feature]
        std = normal_stats["std"][feature]

        if std == 0:
            deviation = 0
        else:
            deviation = abs(value - median) / std

        deviations.append(deviation)

    average_deviation = np.mean(deviations)

    deviation_score = min(100, average_deviation * 25)

    return deviation_score


def inspection_recommendation(
    failure_probability,
    deviation_score
):

    if failure_probability >= 70 or deviation_score >= 60:

        return "Immediate Human Inspection Required"

    elif failure_probability >= 40 or deviation_score >= 30:

        return "Inspection Recommended"

    else:

        return "Normal Monitoring"


@app.get("/")
def home():

    return {
        "message": "Industrial Predictive Maintenance API is running"
    }


@app.post("/predict")
def predict_machine(data: MachineData):

    machine_data = pd.DataFrame([{
        "Type": data.machine_type,
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear
    }])

    # Apply the same preprocessing used during training
    model_input = failure_preprocessor.transform(machine_data)

    # Failure prediction
    probability = (
        failure_detection_model
        .predict_proba(model_input)[0][1]
        * 100
    )

    if probability >= 50:
        machine_status = "Abnormal"
    else:
        machine_status = "Normal"

    # Deviation
    deviation_score = calculate_deviation_score(
        machine_data.iloc[0]
    )

    # Failure diagnosis
    if machine_status == "Abnormal":

        diagnosis_input = diagnosis_preprocessor.transform(
            machine_data
        )

        prediction = failure_diagnosis_model.predict(
            diagnosis_input
        )[0]

        prediction = int(prediction)

        failure_type = label_encoder.inverse_transform(
            [prediction]
        )[0]

    else:
        failure_type = "No Failure"

    # Inspection recommendation
    recommendation = inspection_recommendation(
        probability,
        deviation_score
    )

    return {
    "machine_status": machine_status,
    "failure_probability": float(round(probability, 2)),
    "deviation_score": float(round(deviation_score, 2)),
    "failure_type": str(failure_type),
    "inspection_recommendation": str(recommendation)
}