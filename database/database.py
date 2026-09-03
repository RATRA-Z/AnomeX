import sqlite3
import pandas as pd
import os

from ml.config import DATASET_CSV, DATABASE_FILE


# -----------------------------------
# Connect to database
# -----------------------------------

def get_connection():
    os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
    return sqlite3.connect(DATABASE_FILE)


# -----------------------------------
# Initialize database from CSV
# -----------------------------------

def initialize_database():
    df = pd.read_csv(DATASET_CSV)

    connection = get_connection()

    df.to_sql(
        "components",
        connection,
        if_exists="replace",
        index=False
    )

    connection.close()

    return len(df)


# -----------------------------------
# Get all components
# -----------------------------------

def get_all_components():
    connection = get_connection()

    df = pd.read_sql_query(
        "SELECT * FROM components",
        connection
    )

    connection.close()

    return df


# -----------------------------------
# Get one component
# -----------------------------------

def get_component(component_id):
    connection = get_connection()

    df = pd.read_sql_query(
        "SELECT * FROM components WHERE component_id = ?",
        connection,
        params=(component_id,)
    )

    connection.close()

    return df


# -----------------------------------
# Get components by lot
# -----------------------------------

def get_components_by_lot(lot_id):
    connection = get_connection()

    df = pd.read_sql_query(
        "SELECT * FROM components WHERE lot_id = ?",
        connection,
        params=(lot_id,)
    )

    connection.close()

    return df