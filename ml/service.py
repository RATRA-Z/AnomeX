import pandas as pd

from database.database import get_component
from ml.predict import predict_single


def predict_component(component_id):
    """
    Get a component from SQLite and run the AnomeX ML prediction.
    """

    component_df = get_component(component_id)

    if component_df.empty:
        return None

    component = component_df.iloc[0]

    prediction = predict_single(component)

    return prediction