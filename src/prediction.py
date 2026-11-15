import joblib
import pandas as pd
import os

# Feature columns in the exact training order
FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type"
]

# Project directory
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Load trained models
failure_model = joblib.load(
    os.path.join(BASE_DIR, "models", "best_model.pkl")
)

anomaly_model = joblib.load(
    os.path.join(BASE_DIR, "models", "isolation_forest.pkl")
)

# Load global feature importance
importance_path = os.path.join(
    BASE_DIR, "results", "feature_importance.csv"
)

feature_importance = pd.read_csv(importance_path)


def generate_recommendations(
    sensor_data,
    failure_probability,
    is_anomaly
):
    recommendations = []

    if failure_probability >= 0.50:
        recommendations.append(
            "High predicted failure probability. "
            "Arrange an immediate machine inspection."
        )

    elif failure_probability >= 0.30:
        recommendations.append(
            "Elevated predicted failure probability. "
            "Schedule a maintenance inspection."
        )

    else:
        recommendations.append(
            "Predicted failure probability is currently low. "
            "Continue routine monitoring."
        )

    if sensor_data["Torque [Nm]"] > 60:
        recommendations.append(
            "Torque is relatively high. "
            "Inspect the motor and mechanical load."
        )

    if sensor_data["Rotational speed [rpm]"] > 2500:
        recommendations.append(
            "Rotational speed is high. "
            "Check operating conditions and vibration."
        )

    if sensor_data["Tool wear [min]"] > 200:
        recommendations.append(
            "Tool wear is high. "
            "Inspect the tool and consider replacement."
        )

    temperature_difference = (
        sensor_data["Process temperature [K]"]
        - sensor_data["Air temperature [K]"]
    )

    if temperature_difference > 12:
        recommendations.append(
            "The temperature difference is elevated. "
            "Inspect the cooling system."
        )

    if is_anomaly:
        recommendations.append(
            "Unusual operating pattern detected. "
            "Inspect the machine sensors and operating conditions."
        )

    return recommendations


def predict_machine(sensor_data):

    missing_features = [
        feature for feature in FEATURE_COLUMNS
        if feature not in sensor_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing sensor inputs: {missing_features}"
        )

    if sensor_data["Type"] not in ["L", "M", "H"]:
        raise ValueError(
            "Machine Type must be L, M, or H."
        )

    input_df = pd.DataFrame(
        [[sensor_data[feature] for feature in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    # Predict failure probability
    probabilities = failure_model.predict_proba(input_df)[0]

    # Locate the positive class rather than assuming column 1.
    positive_class_index = list(failure_model.classes_).index(1)
    failure_probability = probabilities[positive_class_index]

    failure_prediction = int(failure_probability >= 0.50)

    # Predict anomaly
    anomaly_result = anomaly_model.predict(input_df)[0]
    is_anomaly = bool(anomaly_result == -1)

    # Risk classification
    if failure_probability >= 0.60:
        risk_level = "High"
    elif failure_probability >= 0.30:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    recommendations = generate_recommendations(
        sensor_data,
        failure_probability,
        is_anomaly
    )

    return {
        "failure_probability": round(
            float(failure_probability) * 100, 2
        ),
        "failure_prediction": failure_prediction,
        "failure_status": (
            "Failure Risk Detected"
            if failure_prediction == 1
            else "No Failure Predicted"
        ),
        "risk_level": risk_level,
        "anomaly_status": (
            "Anomaly Detected"
            if is_anomaly
            else "Normal Operating Pattern"
        ),
        "is_anomaly": is_anomaly,
        "recommendations": recommendations,
        "feature_importance": feature_importance
    }


if __name__ == "__main__":

    sample_machine = {
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
        "Type": "M"
    }

    result = predict_machine(sample_machine)

    print("\n========== MACHINE HEALTH PREDICTION ==========")

    print(
        "\nFailure Probability:",
        result["failure_probability"], "%"
    )

    print("Failure Status:", result["failure_status"])
    print("Risk Level:", result["risk_level"])
    print("Anomaly Status:", result["anomaly_status"])

    print("\nMaintenance Recommendations:")

    for recommendation in result["recommendations"]:
        print("-", recommendation)

    print("\n========== PREDICTION COMPLETED ==========")