from database.database import get_component
from ml.predict import predict_single
from ml.anomaly_detection import analyze_component
from ml.drift_prediction import predict_component_drift


def predict_component(component_id):
    """
    Run all AnomeX ML modules for a single component.

    Returns:
        Combined failure-risk, anomaly, and drift assessment.
    """

    component_df = get_component(component_id)

    if component_df.empty:
        return None

    component = component_df.iloc[0]

    # Module 1: Supervised failure-risk prediction
    failure_prediction = predict_single(component)

    # Module 2: Dynamic anomaly detection
    anomaly_prediction = analyze_component(component_id)

    # Module 3: Time-series drift prediction
    drift_prediction = predict_component_drift(component_id)

    return {
        "component_id": component_id,
        "failure_risk": failure_prediction,
        "anomaly": anomaly_prediction,
        "drift": drift_prediction,
    }