# ml/system_analysis.py

import pandas as pd
import numpy as np
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 1. COMPONENT RISK SCORE
# ============================================================

def calculate_component_risk(df):
    """
    Creates a 0-100 risk score for every component.

    If anomaly_score / failure_probability already exist from
    your ML models, they are used.

    Otherwise the function falls back to drift-related features
    from the current AnomeX dataset.
    """

    data = df.copy()

    # --------------------------------------------------------
    # Anomaly contribution
    # --------------------------------------------------------

    if "anomaly_score" in data.columns:
        anomaly = data["anomaly_score"].fillna(0).astype(float)

        # Normalize to 0-1 if necessary
        if anomaly.max() > 1:
            anomaly = anomaly / anomaly.max()

    else:
        # Temporary fallback until anomaly_detection.py is ready
        drift = data["drift_rate"].abs().fillna(0)

        max_drift = drift.quantile(0.99)

        if max_drift == 0:
            anomaly = pd.Series(0, index=data.index)
        else:
            anomaly = (drift / max_drift).clip(0, 1)

    # --------------------------------------------------------
    # Failure prediction contribution
    # --------------------------------------------------------

    if "failure_probability" in data.columns:
        failure_prob = (
            data["failure_probability"]
            .fillna(0)
            .astype(float)
            .clip(0, 1)
        )

    elif "failure_label" in data.columns:
        failure_prob = (
            data["failure_label"]
            .fillna(0)
            .astype(float)
            .clip(0, 1)
        )

    else:
        failure_prob = pd.Series(0, index=data.index)

    # --------------------------------------------------------
    # Drift contribution
    # --------------------------------------------------------

    percentage_change = (
        data["percentage_change"]
        .abs()
        .fillna(0)
    )

    max_change = percentage_change.quantile(0.99)

    if max_change == 0:
        drift_risk = pd.Series(0, index=data.index)
    else:
        drift_risk = (
            percentage_change / max_change
        ).clip(0, 1)

    # --------------------------------------------------------
    # Trajectory contribution
    # --------------------------------------------------------

    if "unusual_trajectory" in data.columns:

        trajectory_risk = (
            data["unusual_trajectory"]
            .astype(str)
            .str.lower()
            .map({
                "true": 1,
                "false": 0
            })
            .fillna(0)
        )

    else:
        trajectory_risk = pd.Series(0, index=data.index)

    # --------------------------------------------------------
    # Final risk score
    #
    # Give higher importance to failure/anomaly because
    # false negatives are important in this problem.
    # --------------------------------------------------------

    risk = (
        0.35 * anomaly +
        0.40 * failure_prob +
        0.20 * drift_risk +
        0.05 * trajectory_risk
    )

    data["risk_score"] = (risk * 100).clip(0, 100)

    # Risk categories
    data["risk_level"] = pd.cut(
        data["risk_score"],
        bins=[-1, 35, 65, 100],
        labels=["LOW", "MEDIUM", "HIGH"]
    )

    return data


# ============================================================
# 2. SYSTEM HEALTH SCORE
# ============================================================

def calculate_system_health(df):
    """
    Converts all component risks into one System Health score.

    100 = excellent
    0   = extremely unhealthy
    """

    data = calculate_component_risk(df)

    average_risk = data["risk_score"].mean()

    high_risk_ratio = (
        (data["risk_level"] == "HIGH").mean()
    )

    failure_ratio = (
        data["failure_label"].mean()
        if "failure_label" in data.columns
        else 0
    )

    # Penalise not only average risk but also concentration
    # of high-risk components.
    system_risk = (
        0.60 * average_risk +
        0.25 * high_risk_ratio * 100 +
        0.15 * failure_ratio * 100
    )

    system_health = 100 - system_risk

    return round(
        float(np.clip(system_health, 0, 100)),
        2
    )


# ============================================================
# 3. LOT / BATCH LEVEL ANALYSIS
# ============================================================

def analyze_lots(df):
    """
    Finds lots that contain unusual concentrations of
    degrading/high-risk components.
    """

    data = calculate_component_risk(df)

    lot_analysis = (
        data
        .groupby("lot_id", observed=True)
        .agg(
            total_components=("component_id", "count"),

            average_risk=("risk_score", "mean"),

            max_risk=("risk_score", "max"),

            average_drift=("drift_rate", "mean"),

            average_percentage_change=(
                "percentage_change",
                "mean"
            ),

            failures=("failure_label", "sum")
            if "failure_label" in data.columns
            else ("component_id", "count")
        )
        .reset_index()
    )

    # Count high-risk components
    high_counts = (
        data[data["risk_level"] == "HIGH"]
        .groupby("lot_id", observed=True)
        .size()
        .rename("high_risk_components")
    )

    lot_analysis = lot_analysis.merge(
        high_counts,
        on="lot_id",
        how="left"
    )

    lot_analysis["high_risk_components"] = (
        lot_analysis["high_risk_components"]
        .fillna(0)
        .astype(int)
    )

    lot_analysis["high_risk_percentage"] = (
        lot_analysis["high_risk_components"]
        / lot_analysis["total_components"]
        * 100
    )

    # Lot health
    lot_analysis["lot_health"] = (
        100 -
        (
            0.7 * lot_analysis["average_risk"] +
            0.3 * lot_analysis["high_risk_percentage"]
        )
    ).clip(0, 100)

    lot_analysis["lot_status"] = pd.cut(
        lot_analysis["lot_health"],
        bins=[-1, 50, 75, 100],
        labels=["CRITICAL", "WATCH", "HEALTHY"]
    )

    return lot_analysis.sort_values(
        "average_risk",
        ascending=False
    )


# ============================================================
# 4. COMPONENT RELATIONSHIP GRAPH
# ============================================================

def build_relationship_graph(
    df,
    max_components=500,
    neighbours=5,
    similarity_threshold=0.82
):
    """
    Builds a graph where:

        Node = Component
        Edge = Similar degradation behaviour

    We DON'T compare all 10,000 components against each other
    because that would be unnecessarily expensive.

    Instead we analyse the highest-risk components and use
    nearest neighbours.
    """

    data = calculate_component_risk(df)

    # Focus on the most interesting components
    data = (
        data
        .sort_values("risk_score", ascending=False)
        .head(max_components)
        .copy()
    )

    features = [
        "value_0h",
        "value_24h",
        "value_96h",
        "value_168h",
        "change_0h_24h",
        "change_24h_96h",
        "change_96h_168h",
        "drift_rate",
        "percentage_change"
    ]

    # Only use columns that actually exist
    features = [
        col for col in features
        if col in data.columns
    ]

    X = (
        data[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Standardise measurements
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find nearest behavioural neighbours
    n_neighbors = min(
        neighbours + 1,
        len(data)
    )

    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean"
    )

    model.fit(X_scaled)

    distances, indices = model.kneighbors(X_scaled)

    graph = nx.Graph()

    # --------------------------------------------------------
    # Add nodes
    # --------------------------------------------------------

    for _, row in data.iterrows():

        graph.add_node(
            row["component_id"],

            risk_score=round(
                float(row["risk_score"]),
                2
            ),

            risk_level=str(row["risk_level"]),

            lot_id=row.get(
                "lot_id",
                "UNKNOWN"
            ),

            component_type=row.get(
                "component_type",
                "UNKNOWN"
            ),

            manufacturer=row.get(
                "manufacturer",
                "UNKNOWN"
            )
        )

    # --------------------------------------------------------
    # Add similarity connections
    # --------------------------------------------------------

    for i in range(len(data)):

        component_a = data.iloc[i]["component_id"]

        for j in range(1, n_neighbors):

            neighbour_index = indices[i][j]

            component_b = (
                data.iloc[neighbour_index]
                ["component_id"]
            )

            distance = distances[i][j]

            # Convert distance into intuitive similarity
            similarity = 1 / (1 + distance)

            # Give small bonus when metadata also matches
            same_lot = (
                data.iloc[i]["lot_id"]
                ==
                data.iloc[neighbour_index]["lot_id"]
            )

            same_type = (
                data.iloc[i]["component_type"]
                ==
                data.iloc[neighbour_index]["component_type"]
            )

            same_manufacturer = (
                data.iloc[i]["manufacturer"]
                ==
                data.iloc[neighbour_index]["manufacturer"]
            )

            metadata_bonus = (
                0.06 * same_lot +
                0.03 * same_type +
                0.02 * same_manufacturer
            )

            similarity = min(
                1,
                similarity + metadata_bonus
            )

            if similarity >= similarity_threshold:

                graph.add_edge(
                    component_a,
                    component_b,

                    similarity=round(
                        float(similarity),
                        3
                    ),

                    same_lot=bool(same_lot),

                    same_type=bool(same_type),

                    same_manufacturer=bool(
                        same_manufacturer
                    )
                )

    return graph, data


# ============================================================
# 5. FIND DEGRADATION CLUSTERS
# ============================================================

def detect_degradation_clusters(graph):
    """
    Finds groups of components showing related behaviour.
    """

    if graph.number_of_nodes() == 0:
        return []

    if graph.number_of_edges() == 0:
        return []

    communities = list(
        nx.community.asyn_lpa_communities(
            graph,
            weight="similarity",
            seed=42
        )
    )

    clusters = []

    for cluster_id, community in enumerate(
        communities,
        start=1
    ):

        nodes = list(community)

        risks = [
            graph.nodes[node]["risk_score"]
            for node in nodes
        ]

        lots = [
            graph.nodes[node]["lot_id"]
            for node in nodes
        ]

        manufacturers = [
            graph.nodes[node]["manufacturer"]
            for node in nodes
        ]

        cluster = {
            "cluster_id": cluster_id,

            "size": len(nodes),

            "components": nodes,

            "average_risk": round(
                float(np.mean(risks)),
                2
            ),

            "max_risk": round(
                float(np.max(risks)),
                2
            ),

            "dominant_lot": (
                pd.Series(lots)
                .mode()
                .iloc[0]
            ),

            "dominant_manufacturer": (
                pd.Series(manufacturers)
                .mode()
                .iloc[0]
            )
        }

        clusters.append(cluster)

    clusters.sort(
        key=lambda x: x["average_risk"],
        reverse=True
    )

    return clusters


# ============================================================
# 6. COMPLETE SYSTEM ANALYSIS
# ============================================================

def run_system_analysis(df):
    """
    Main function.

    Run this after your component-level ML models.
    """

    data = calculate_component_risk(df)

    health = calculate_system_health(data)

    lots = analyze_lots(data)

    graph, graph_data = build_relationship_graph(data)

    clusters = detect_degradation_clusters(graph)

    high_risk = data[
        data["risk_level"] == "HIGH"
    ]

    medium_risk = data[
        data["risk_level"] == "MEDIUM"
    ]

    summary = {

        "system_health": health,

        "total_components": len(data),

        "high_risk_components": len(high_risk),

        "medium_risk_components": len(medium_risk),

        "high_risk_percentage": round(
            len(high_risk) / len(data) * 100,
            2
        ),

        "number_of_relationships":
            graph.number_of_edges(),

        "number_of_clusters":
            len(clusters),

        "highest_risk_lot":
            lots.iloc[0]["lot_id"]
            if len(lots) > 0
            else None
    }

    return {
        "components": data,
        "summary": summary,
        "lots": lots,
        "graph": graph,
        "graph_components": graph_data,
        "clusters": clusters
    }


# ============================================================
# 7. WHAT-IF: REMOVE HIGH-RISK COMPONENTS
# ============================================================

def simulate_component_removal(
    df,
    component_ids
):
    """
    What happens to system health if certain components
    are rejected/removed?
    """

    original_health = calculate_system_health(df)

    simulated_df = df[
        ~df["component_id"].isin(component_ids)
    ].copy()

    new_health = calculate_system_health(
        simulated_df
    )

    return {

        "scenario":
            "Remove selected components",

        "removed_components":
            component_ids,

        "original_health":
            original_health,

        "simulated_health":
            new_health,

        "health_change":
            round(
                new_health - original_health,
                2
            )
    }


# ============================================================
# 8. WHAT-IF: FUTURE DEGRADATION
# ============================================================

def simulate_degradation(
    df,
    component_ids,
    degradation_percentage=10
):
    """
    Scenario simulation:
    assumes selected components continue degrading by
    a chosen percentage.

    IMPORTANT:
    This is a model-based scenario, NOT a physical digital twin.
    """

    original_health = calculate_system_health(df)

    simulated = df.copy()

    mask = simulated[
        "component_id"
    ].isin(component_ids)

    factor = (
        1 +
        degradation_percentage / 100
    )

    measurement_columns = [
        "value_24h",
        "value_96h",
        "value_168h"
    ]

    for column in measurement_columns:

        if column in simulated.columns:

            simulated.loc[
                mask,
                column
            ] *= factor

    # Recalculate degradation features
    if all(
        col in simulated.columns
        for col in ["value_0h", "value_168h"]
    ):

        simulated[
            "overall_change_0h_168h"
        ] = (
            simulated["value_168h"]
            -
            simulated["value_0h"]
        )

        simulated[
            "percentage_change"
        ] = (
            simulated[
                "overall_change_0h_168h"
            ]
            /
            simulated["value_0h"]
            * 100
        )

        simulated["drift_rate"] = (
            simulated[
                "overall_change_0h_168h"
            ] / 168
        )

    new_health = calculate_system_health(
        simulated
    )

    return {

        "scenario":
            f"{degradation_percentage}% additional degradation",

        "affected_components":
            len(component_ids),

        "original_health":
            original_health,

        "simulated_health":
            new_health,

        "health_change":
            round(
                new_health - original_health,
                2
            )
    }