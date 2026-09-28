import time

from datetime import date, datetime, time as dt_time, timezone

import numpy as np

import httpx

import pandas as pd

import streamlit as st

import plotly.graph_objects as go

from pathlib import Path
import joblib





# ==========================================

# CONFIGURATION

# ==========================================



API_URL = "http://127.0.0.1:8000"

PROJECT_DIR = Path(__file__).resolve().parents[1]
FINAL_2026_JOBLIB = PROJECT_DIR / "models" / "final_2026_complete_evaluation.joblib"
FINAL_2026_CSV = PROJECT_DIR / "results" / "final_2026_predictions.csv"





st.set_page_config(

    page_title="Energy Demand Forecasting",

    page_icon="⚡",

    layout="wide",

)



dashboard_start = (

    time.perf_counter()

)





# ==========================================

# FONCTION API

# ==========================================



def call_api(

    endpoint,

    params=None,

    timeout=5.0,

):



    start = time.perf_counter()



    try:



        response = httpx.get(

            API_URL + endpoint,

            params=params,

            timeout=timeout,

        )



        elapsed_ms = (

            time.perf_counter()

            - start

        ) * 1000



        response.raise_for_status()



        return {

            "success": True,

            "data": response.json(),

            "latency_ms": elapsed_ms,

            "error": None,

        }



    except Exception as exc:



        elapsed_ms = (

            time.perf_counter()

            - start

        ) * 1000



        return {

            "success": False,

            "data": None,

            "latency_ms": elapsed_ms,

            "error": str(exc),

        }





# ==========================================

# CACHED API CALL

# ==========================================



@st.cache_data(

    ttl=60,

    show_spinner=False,

)

def cached_api_data(

    endpoint,

    params_items=(),

    timeout=5.0,

):



    params = dict(

        params_items

    )



    response = httpx.get(

        API_URL + endpoint,

        params=params,

        timeout=timeout,

    )



    response.raise_for_status()



    return response.json()





def call_api_cached(

    endpoint,

    params=None,

    timeout=5.0,

):



    start = time.perf_counter()



    try:



        params_items = tuple(

            sorted(

                (params or {}).items()

            )

        )



        data = cached_api_data(

            endpoint,

            params_items,

            timeout,

        )



        elapsed_ms = (

            time.perf_counter()

            - start

        ) * 1000



        return {

            "success": True,

            "data": data,

            "latency_ms": elapsed_ms,

            "error": None,

        }



    except Exception as exc:



        elapsed_ms = (

            time.perf_counter()

            - start

        ) * 1000



        return {

            "success": False,

            "data": None,

            "latency_ms": elapsed_ms,

            "error": str(exc),

        }





# ==========================================

# TITRE

# ==========================================



# ============================================================
# FINAL TEST 2026 — READ ONLY
# ============================================================

@st.cache_data(show_spinner=False)
def load_final_2026_dataframe():

    # Source privil?gi?e pour GitHub :
    # pr?dictions finales fig?es dans un CSV.
    if FINAL_2026_CSV.exists():

        df = pd.read_csv(
            FINAL_2026_CSV,
            parse_dates=["timestamp_utc"],
        )

        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"],
            utc=True,
        )

        required_columns = {
            "timestamp_utc",
            "actual_mw",
            "pred_ridge",
            "pred_stage1",
            "pred_patch",
            "prediction_mw",
            "event_mask",
            "error_mw",
            "error_percent",
        }

        missing = (
            required_columns
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                "Colonnes manquantes dans le CSV : "
                + ", ".join(sorted(missing))
            )

        return (
            df
            .sort_values("timestamp_utc")
            .reset_index(drop=True)
        )

    # Fallback local :
    # utilise le joblib si le CSV n'existe pas.
    if not FINAL_2026_JOBLIB.exists():

        raise FileNotFoundError(
            "Aucune source 2026 trouv?e. "
            f"CSV : {FINAL_2026_CSV} ; "
            f"joblib : {FINAL_2026_JOBLIB}"
        )

    obj = joblib.load(
        FINAL_2026_JOBLIB
    )

    index = pd.to_datetime(
        obj["index"],
        utc=True,
    )

    actual = np.asarray(
        obj["y_true"],
        dtype=float,
    )

    prediction = np.asarray(
        obj["prediction_final"],
        dtype=float,
    )

    error = actual - prediction

    error_percent = np.where(
        actual != 0,
        100.0 * error / actual,
        np.nan,
    )

    df = pd.DataFrame({
        "timestamp_utc": index,
        "actual_mw": actual,
        "pred_ridge": np.asarray(
            obj["pred_ridge"],
            dtype=float,
        ),
        "pred_stage1": np.asarray(
            obj["pred_stage1"],
            dtype=float,
        ),
        "pred_patch": np.asarray(
            obj["pred_patch"],
            dtype=float,
        ),
        "prediction_mw": prediction,
        "event_mask": np.asarray(
            obj["event_mask"],
            dtype=bool,
        ),
        "error_mw": error,
        "error_percent": error_percent,
    })

    return (
        df
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )


def compute_metrics_2026(actual, prediction):

    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    mask = (
        np.isfinite(actual)
        & np.isfinite(prediction)
    )

    actual = actual[mask]
    prediction = prediction[mask]

    error = actual - prediction

    rmse = float(
        np.sqrt(
            np.mean(error ** 2)
        )
    )

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    nonzero = actual != 0

    mape = float(
        np.mean(
            np.abs(
                error[nonzero]
                / actual[nonzero]
            )
        )
        * 100
    )

    ss_res = float(
        np.sum(error ** 2)
    )

    ss_tot = float(
        np.sum(
            (
                actual
                - np.mean(actual)
            ) ** 2
        )
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    bias = float(
        np.mean(error)
    )

    mean_actual = float(
        np.mean(actual)
    )

    nrmse = (
        100.0 * rmse / mean_actual
        if mean_actual != 0
        else np.nan
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "bias": bias,
        "nrmse": nrmse,
    }


def render_final_2026_dashboard():

    try:
        full_df = load_final_2026_dataframe()

    except Exception as exc:
        st.error(
            "Impossible de charger le test final 2026."
        )
        st.code(str(exc))
        return

    st.subheader(
        "🔒 Test final indépendant 2026"
    )

    st.warning(
        "Mode lecture seule : les données 2026 ont été gardées "
        "hors du développement du modèle. "
        "Aucun réentraînement ni réglage n'est effectué ici."
    )

    min_date = full_df["timestamp_utc"].min().date()
    max_date = full_df["timestamp_utc"].max().date()

    default_end = min(
        min_date + pd.Timedelta(days=7),
        max_date,
    )

    with st.sidebar:

        st.divider()

        st.subheader(
            "Période 2026"
        )

        start_date_2026 = st.date_input(
            "Date de début — 2026",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="final_2026_start",
        )

        end_date_2026 = st.date_input(
            "Date de fin — 2026",
            value=default_end,
            min_value=min_date,
            max_value=max_date,
            key="final_2026_end",
        )

    if end_date_2026 < start_date_2026:
        st.error(
            "La date de fin doit être postérieure "
            "ou égale à la date de début."
        )
        return

    start_ts = pd.Timestamp(
        start_date_2026,
        tz="UTC",
    )

    end_ts = (
        pd.Timestamp(
            end_date_2026,
            tz="UTC",
        )
        + pd.Timedelta(days=1)
    )

    selected_df = (
        full_df[
            (
                full_df["timestamp_utc"]
                >= start_ts
            )
            & (
                full_df["timestamp_utc"]
                < end_ts
            )
        ]
        .copy()
    )

    if selected_df.empty:
        st.warning(
            "Aucune donnée disponible sur cette période."
        )
        return

    # ============================================================
    # HISTORICAL COMPARISON — 2025 vs 2026
    # ============================================================

    with st.sidebar:

        st.divider()

        st.subheader(
            "Comparaison historique"
        )

        compare_history = st.checkbox(
            "Comparer 2026 à une période 2025",
            value=True,
            key="final_2026_compare_history",
        )

        historical_start_date = None
        historical_end_date = None

        if compare_history:

            same_calendar_dates = st.checkbox(
                "Même période un an plus tôt",
                value=True,
                key="final_2026_same_calendar_dates",
            )

            if same_calendar_dates:

                historical_start_date = (
                    start_date_2026.replace(
                        year=2025
                    )
                )

                historical_end_date = (
                    end_date_2026.replace(
                        year=2025
                    )
                )

                st.caption(
                    "Référence : "
                    f"{historical_start_date:%d/%m/%Y} → "
                    f"{historical_end_date:%d/%m/%Y}"
                )

            else:

                historical_start_date = st.date_input(
                    "Début comparaison — 2025",
                    value=date(
                        2025,
                        1,
                        15,
                    ),
                    min_value=date(
                        2025,
                        1,
                        1,
                    ),
                    max_value=date(
                        2025,
                        12,
                        31,
                    ),
                    key="final_2026_history_start",
                )

                historical_end_date = st.date_input(
                    "Fin comparaison — 2025",
                    value=date(
                        2025,
                        1,
                        22,
                    ),
                    min_value=date(
                        2025,
                        1,
                        1,
                    ),
                    max_value=date(
                        2025,
                        12,
                        31,
                    ),
                    key="final_2026_history_end",
                )

    historical_df = pd.DataFrame()
    historical_metrics = None
    historical_response = None

    if compare_history:

        historical_days = (
            historical_end_date
            - historical_start_date
        ).days + 1

        if historical_end_date < historical_start_date:

            st.warning(
                "Comparaison 2025 désactivée : "
                "la date de fin est antérieure "
                "à la date de début."
            )

        elif historical_days > 31:

            st.warning(
                "Comparaison 2025 désactivée : "
                "la période ne peut pas dépasser "
                "31 jours."
            )

        else:

            historical_start_ts = datetime.combine(
                historical_start_date,
                dt_time.min,
                tzinfo=timezone.utc,
            )

            historical_end_ts = (
                datetime.combine(
                    historical_end_date,
                    dt_time.min,
                    tzinfo=timezone.utc,
                )
                + pd.Timedelta(days=1)
            )

            historical_response = call_api_cached(
                "/predictions",
                params={
                    "start": (
                        historical_start_ts
                        .isoformat()
                        .replace(
                            "+00:00",
                            "Z",
                        )
                    ),
                    "end": (
                        historical_end_ts
                        .isoformat()
                        .replace(
                            "+00:00",
                            "Z",
                        )
                    ),
                    "limit": 5000,
                },
            )

            if historical_response["success"]:

                historical_df = pd.DataFrame(
                    historical_response[
                        "data"
                    ][
                        "predictions"
                    ]
                )

                if not historical_df.empty:

                    historical_df[
                        "timestamp_utc"
                    ] = pd.to_datetime(
                        historical_df[
                            "timestamp_utc"
                        ],
                        utc=True,
                    )

                    historical_metrics = (
                        compute_metrics_2026(
                            historical_df[
                                "actual_mw"
                            ],
                            historical_df[
                                "prediction_mw"
                            ],
                        )
                    )

            else:

                st.warning(
                    "La comparaison avec 2025 "
                    "nécessite FastAPI/PostgreSQL."
                )

                st.caption(
                    "Erreur API : "
                    f"{historical_response['error']}"
                )

    global_metrics = compute_metrics_2026(
        full_df["actual_mw"],
        full_df["prediction_mw"],
    )

    st.caption(
        "Période complète du test : "
        f"{min_date:%d/%m/%Y} → "
        f"{max_date:%d/%m/%Y} "
        f"({len(full_df)} points)"
    )

    st.subheader(
        "Performance globale — 2026"
    )

    g1, g2, g3, g4 = st.columns(4)

    with g1:
        st.metric(
            "RMSE",
            f"{global_metrics['rmse']:.2f} MW",
        )

    with g2:
        st.metric(
            "MAE",
            f"{global_metrics['mae']:.2f} MW",
        )

    with g3:
        st.metric(
            "MAPE",
            f"{global_metrics['mape']:.3f} %",
        )

    with g4:
        st.metric(
            "R²",
            f"{global_metrics['r2']:.6f}",
        )

    st.metric(
        "NRMSE",
        f"{global_metrics['nrmse']:.3f} %",
    )

    progression = []

    for label, column in [
        ("Ridge", "pred_ridge"),
        ("Ridge + MLP1", "pred_stage1"),
        ("+ Patch Transformer", "pred_patch"),
        ("Final + Event Expert", "prediction_mw"),
    ]:

        metrics = compute_metrics_2026(
            full_df["actual_mw"],
            full_df[column],
        )

        progression.append({
            "Modèle": label,
            "RMSE (MW)": metrics["rmse"],
            "MAE (MW)": metrics["mae"],
            "MAPE (%)": metrics["mape"],
            "R²": metrics["r2"],
        })

    progression_df = pd.DataFrame(
        progression
    )

    ridge_rmse = float(
        progression_df.loc[
            progression_df["Modèle"] == "Ridge",
            "RMSE (MW)",
        ].iloc[0]
    )

    final_rmse = float(
        progression_df.loc[
            progression_df["Modèle"]
            == "Final + Event Expert",
            "RMSE (MW)",
        ].iloc[0]
    )

    gain_vs_ridge = (
        100.0
        * (ridge_rmse - final_rmse)
        / ridge_rmse
    )

    st.subheader(
        "Progression des modèles"
    )

    st.metric(
        "Gain RMSE final vs Ridge",
        f"{gain_vs_ridge:.2f} %",
    )

    st.dataframe(
        progression_df,
        width="stretch",
        hide_index=True,
    )

    fig_progression = go.Figure()

    fig_progression.add_trace(
        go.Bar(
            x=progression_df["Modèle"],
            y=progression_df["RMSE (MW)"],
            name="RMSE",
        )
    )

    fig_progression.update_layout(
        xaxis_title="Modèle",
        yaxis_title="RMSE (MW)",
        height=420,
    )

    st.plotly_chart(
        fig_progression,
        width="stretch",
    )

    period_metrics = compute_metrics_2026(
        selected_df["actual_mw"],
        selected_df["prediction_mw"],
    )

    st.divider()

    st.subheader(
        "Performance sur la période sélectionnée"
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "RMSE période",
            f"{period_metrics['rmse']:.2f} MW",
        )

    with m2:
        st.metric(
            "MAE période",
            f"{period_metrics['mae']:.2f} MW",
        )

    with m3:
        st.metric(
            "MAPE période",
            f"{period_metrics['mape']:.3f} %",
        )

    with m4:
        st.metric(
            "Biais moyen",
            f"{period_metrics['bias']:.2f} MW",
        )

    st.subheader(
        "Consommation réelle vs prédiction"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=selected_df["timestamp_utc"],
            y=selected_df["actual_mw"],
            mode="lines",
            name="Consommation réelle",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=selected_df["timestamp_utc"],
            y=selected_df["prediction_mw"],
            mode="lines",
            name="Prédiction finale",
            customdata=selected_df[
                ["error_mw", "error_percent"]
            ],
            hovertemplate=(
                "Prédiction : %{y:.0f} MW"
                "<br>Erreur : %{customdata[0]:.1f} MW"
                "<br>Erreur : %{customdata[1]:.3f} %"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        xaxis_title="Temps",
        yaxis_title="Consommation (MW)",
        hovermode="x unified",
        height=520,
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.subheader(
        "Erreur au cours du temps"
    )

    fig_error = go.Figure()

    fig_error.add_trace(
        go.Scatter(
            x=selected_df["timestamp_utc"],
            y=selected_df["error_mw"],
            mode="lines",
            name="Erreur",
        )
    )

    fig_error.add_hline(
        y=0,
        line_dash="dash",
    )

    fig_error.update_layout(
        xaxis_title="Temps",
        yaxis_title="Erreur (MW)",
        height=350,
    )

    st.plotly_chart(
        fig_error,
        width="stretch",
    )

    st.subheader(
        "Distribution des erreurs"
    )

    fig_hist = go.Figure()

    fig_hist.add_trace(
        go.Histogram(
            x=selected_df["error_mw"],
            nbinsx=30,
            name="Erreur",
        )
    )

    fig_hist.update_layout(
        xaxis_title="Erreur (MW)",
        yaxis_title="Fréquence",
        height=400,
    )

    st.plotly_chart(
        fig_hist,
        width="stretch",
    )

    abs_errors = np.abs(
        selected_df["error_mw"]
    )

    q50, q90, q95, q99 = np.quantile(
        abs_errors,
        [0.50, 0.90, 0.95, 0.99],
    )

    st.subheader(
        "Quantiles de l'erreur absolue"
    )

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        st.metric("50 %", f"{q50:.1f} MW")
    with q2:
        st.metric("90 %", f"{q90:.1f} MW")
    with q3:
        st.metric("95 %", f"{q95:.1f} MW")
    with q4:
        st.metric("99 %", f"{q99:.1f} MW")

    largest_errors = (
        selected_df
        .assign(
            abs_error_mw=lambda df:
            np.abs(df["error_mw"])
        )
        .sort_values(
            "abs_error_mw",
            ascending=False,
        )
        .head(10)
    )

    st.subheader(
        "10 plus grosses erreurs"
    )

    st.dataframe(
        largest_errors[
            [
                "timestamp_utc",
                "actual_mw",
                "prediction_mw",
                "error_mw",
                "error_percent",
                "abs_error_mw",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    event_count = int(
        selected_df["event_mask"].sum()
    )

    event_share = (
        100.0
        * event_count
        / len(selected_df)
    )

    st.metric(
        "Points Event Expert",
        f"{event_count} ({event_share:.2f} %)",
    )

    # ============================================================
    # DISPLAY HISTORICAL COMPARISON
    # ============================================================

    if (
        compare_history
        and historical_metrics is not None
        and not historical_df.empty
    ):

        st.divider()

        st.subheader(
            "Comparaison 2025 ↔ 2026"
        )

        st.caption(
            "2025 est utilisé uniquement comme "
            "référence historique. "
            "Le test final 2026 reste en lecture seule."
        )

        relative_rmse_change = (
            100.0
            * (
                period_metrics["rmse"]
                - historical_metrics["rmse"]
            )
            / historical_metrics["rmse"]
        )

        relative_mae_change = (
            100.0
            * (
                period_metrics["mae"]
                - historical_metrics["mae"]
            )
            / historical_metrics["mae"]
        )

        relative_mape_change = (
            100.0
            * (
                period_metrics["mape"]
                - historical_metrics["mape"]
            )
            / historical_metrics["mape"]
        )

        hc1, hc2, hc3, hc4 = st.columns(4)

        with hc1:

            st.metric(
                "RMSE — 2025",
                f"{historical_metrics['rmse']:.2f} MW",
            )

        with hc2:

            st.metric(
                "RMSE — 2026",
                f"{period_metrics['rmse']:.2f} MW",
                delta=(
                    f"{relative_rmse_change:+.1f} % "
                    "vs 2025"
                ),
                delta_color="inverse",
            )

        with hc3:

            st.metric(
                "MAE — 2025",
                f"{historical_metrics['mae']:.2f} MW",
            )

        with hc4:

            st.metric(
                "MAE — 2026",
                f"{period_metrics['mae']:.2f} MW",
                delta=(
                    f"{relative_mae_change:+.1f} % "
                    "vs 2025"
                ),
                delta_color="inverse",
            )

        comparison_metrics_df = pd.DataFrame({
            "Métrique": [
                "RMSE (MW)",
                "MAE (MW)",
                "MAPE (%)",
                "R²",
                "NRMSE (%)",
            ],
            "2025": [
                historical_metrics["rmse"],
                historical_metrics["mae"],
                historical_metrics["mape"],
                historical_metrics["r2"],
                historical_metrics["nrmse"],
            ],
            "2026": [
                period_metrics["rmse"],
                period_metrics["mae"],
                period_metrics["mape"],
                period_metrics["r2"],
                period_metrics["nrmse"],
            ],
            "Écart 2026 - 2025": [
                (
                    period_metrics["rmse"]
                    - historical_metrics["rmse"]
                ),
                (
                    period_metrics["mae"]
                    - historical_metrics["mae"]
                ),
                (
                    period_metrics["mape"]
                    - historical_metrics["mape"]
                ),
                (
                    period_metrics["r2"]
                    - historical_metrics["r2"]
                ),
                (
                    period_metrics["nrmse"]
                    - historical_metrics["nrmse"]
                ),
            ],
        })

        st.dataframe(
            comparison_metrics_df,
            width="stretch",
            hide_index=True,
        )

        st.metric(
            "Variation MAPE 2026 vs 2025",
            f"{relative_mape_change:+.1f} %",
            delta_color="inverse",
        )

        st.subheader(
            "Profils de consommation comparés"
        )

        fig_compare_consumption = go.Figure()

        fig_compare_consumption.add_trace(
            go.Scatter(
                x=np.arange(
                    len(historical_df)
                ),
                y=historical_df[
                    "actual_mw"
                ],
                mode="lines",
                name=(
                    "Réel 2025 — "
                    f"{historical_start_date:%d/%m}"
                    " → "
                    f"{historical_end_date:%d/%m}"
                ),
            )
        )

        fig_compare_consumption.add_trace(
            go.Scatter(
                x=np.arange(
                    len(selected_df)
                ),
                y=selected_df[
                    "actual_mw"
                ],
                mode="lines",
                name=(
                    "Réel 2026 — "
                    f"{start_date_2026:%d/%m}"
                    " → "
                    f"{end_date_2026:%d/%m}"
                ),
            )
        )

        fig_compare_consumption.update_layout(
            xaxis_title=(
                "Pas de temps depuis le début "
                "de chaque période"
            ),
            yaxis_title="Consommation (MW)",
            hovermode="x unified",
            height=450,
        )

        st.plotly_chart(
            fig_compare_consumption,
            width="stretch",
        )

        st.subheader(
            "Erreurs du modèle : 2025 vs 2026"
        )

        fig_compare_error = go.Figure()

        fig_compare_error.add_trace(
            go.Scatter(
                x=np.arange(
                    len(historical_df)
                ),
                y=historical_df[
                    "error_mw"
                ],
                mode="lines",
                name="Erreur 2025",
            )
        )

        fig_compare_error.add_trace(
            go.Scatter(
                x=np.arange(
                    len(selected_df)
                ),
                y=selected_df[
                    "error_mw"
                ],
                mode="lines",
                name="Erreur 2026",
            )
        )

        fig_compare_error.add_hline(
            y=0,
            line_dash="dash",
        )

        fig_compare_error.update_layout(
            xaxis_title=(
                "Pas de temps depuis le début "
                "de chaque période"
            ),
            yaxis_title="Erreur (MW)",
            hovermode="x unified",
            height=420,
        )

        st.plotly_chart(
            fig_compare_error,
            width="stretch",
        )

    st.info(
        "Le test 2026 est affiché à partir "
        "des prédictions finales figées. "
        "Aucun réentraînement n'est effectué."
    )


st.title(

    "⚡ Energy Demand Forecasting"

)



st.caption(

    "Probabilistic & Causal Machine Learning "

    "for Energy Demand"

)





# ==========================================

# SIDEBAR

# ==========================================



# ============================================================
# DATA SOURCE
# ============================================================

data_source = st.sidebar.radio(
    "Jeu de données",
    [
        "Développement / validation 2025",
        "Test final indépendant 2026 🔒",
    ],
    index=0,
)

if data_source == "Test final indépendant 2026 🔒":

    render_final_2026_dashboard()

    st.stop()


with st.sidebar:



    st.header(

        "Configuration"

    )



    st.write(

        "Bloc 10 — Dashboard"

    )



    st.divider()



    start_date = st.date_input(

        "Date de début",

        value=date(

            2025,

            1,

            14

        ),

        min_value=date(

            2025,

            1,

            1

        ),

        max_value=date(

            2025,

            12,

            31

        ),

    )



    end_date = st.date_input(

        "Date de fin",

        value=date(

            2025,

            1,

            15

        ),

        min_value=date(

            2025,

            1,

            1

        ),

        max_value=date(

            2025,

            12,

            31

        ),

    )



    st.divider()



    st.warning(

        "2026 reste réservé "

        "au test final."

    )



    st.divider()



    if st.button(

    "Rafraîchir les données"

):



        st.cache_data.clear()



        st.rerun()



    st.subheader(

        "Comparaison"

    )



    compare_enabled = st.checkbox(

        "Comparer avec une autre période",

        value=False,

    )



    if compare_enabled:



        compare_start_date = st.date_input(

            "Début période B",

            value=date(

                2025,

                7,

                14

            ),

            min_value=date(

                2025,

                1,

                1

            ),

            max_value=date(

                2025,

                12,

                31

            ),

        )



        compare_end_date = st.date_input(

            "Fin période B",

            value=date(

                2025,

                7,

                15

            ),

            min_value=date(

                2025,

                1,

                1

            ),

            max_value=date(

                2025,

                12,

                31

            ),

        )





# ==========================================

# DATE VALIDATION

# ==========================================



start_datetime = datetime.combine(

    start_date,

    dt_time.min,

    tzinfo=timezone.utc,

)



end_datetime = datetime.combine(

    end_date,

    dt_time.min,

    tzinfo=timezone.utc,

)





if end_datetime <= start_datetime:



    st.error(

        "La date de fin doit être "

        "postérieure à la date de début."

    )



    st.stop()





period_days = (

    end_datetime

    - start_datetime

).days





if period_days > 31:



    st.error(

        "La période maximale "

        "est de 31 jours."

    )



    st.stop()





 # ==========================================

# COMPARISON PERIOD VALIDATION

# ==========================================



if compare_enabled:



    compare_start_datetime = datetime.combine(

        compare_start_date,

        dt_time.min,

        tzinfo=timezone.utc,

    )



    compare_end_datetime = datetime.combine(

        compare_end_date,

        dt_time.min,

        tzinfo=timezone.utc,

    )





    if (

        compare_end_datetime

        <= compare_start_datetime

    ):



        st.error(

            "La fin de la période B "

            "doit être postérieure au début."

        )



        st.stop()





    compare_days = (

        compare_end_datetime

        - compare_start_datetime

    ).days





    if compare_days > 31:



        st.error(

            "La période B ne peut pas "

            "dépasser 31 jours."

        )



        st.stop()





# ==========================================

# ISO FORMAT FOR FASTAPI

# ==========================================



start_iso = (

    start_datetime

    .isoformat()

    .replace(

        "+00:00",

        "Z"

    )

)



end_iso = (

    end_datetime

    .isoformat()

    .replace(

        "+00:00",

        "Z"

    )

)



# ==========================================

# COMPARISON ISO FORMAT

# ==========================================



if compare_enabled:



    compare_start_iso = (

        compare_start_datetime

        .isoformat()

        .replace(

            "+00:00",

            "Z"

        )

    )



    compare_end_iso = (

        compare_end_datetime

        .isoformat()

        .replace(

            "+00:00",

            "Z"

        )

    )



# ==========================================

# API STATUS

# ==========================================



health = call_api(

    "/health"

)



database = call_api(

    "/database/health"

)



model = call_api(

    "/model/metrics"

)





# ==========================================

# FASTAPI MESSAGE

# ==========================================



if health["success"]:



    st.success(

        "FastAPI connecté"

    )



else:



    st.error(

        "Impossible de joindre FastAPI."

    )



    st.code(

        health["error"]

    )



    st.stop()





# ==========================================

# SYSTEM STATUS

# ==========================================



st.subheader(

    "État du système"

)





col1, col2, col3 = st.columns(3)





with col1:



    st.metric(

        "FastAPI",

        "ONLINE"

        if health["success"]

        else "OFFLINE"

    )





with col2:



    st.metric(

        "PostgreSQL",

        "ONLINE"

        if database["success"]

        else "OFFLINE"

    )





with col3:



    st.metric(

        "Modèle",

        "AVAILABLE"

        if model["success"]

        else "UNAVAILABLE"

    )





# ==========================================

# MODEL INFORMATION

# ==========================================



if model["success"]:



    model_data = model["data"]



    st.subheader(

        "Modèle de référence"

    )



    st.write(

        model_data["model_name"]

    )





    metric_col1, metric_col2, metric_col3, metric_col4 = (

        st.columns(4)

    )





    with metric_col1:



        st.metric(

            "RMSE",

            f"{model_data['rmse']:.2f} MW"

        )





    with metric_col2:



        st.metric(

            "MAE",

            f"{model_data['mae']:.2f} MW"

        )





    with metric_col3:



        st.metric(

            "MAPE",

            f"{model_data['mape']:.3f} %"

        )





    with metric_col4:



        st.metric(

            "R²",

            f"{model_data['r2']:.6f}"

        )





# ==========================================

# SELECTED PERIOD

# ==========================================



st.subheader(

    "Période sélectionnée"

)





period_col1, period_col2, period_col3 = (

    st.columns(3)

)





with period_col1:



    st.metric(

        "Début",

        start_date.strftime(

            "%d/%m/%Y"

        )

    )





with period_col2:



    st.metric(

        "Fin",

        end_date.strftime(

            "%d/%m/%Y"

        )

    )





with period_col3:



    st.metric(

        "Durée",

        f"{period_days} jour(s)"

    )





# ==========================================

# LOAD PREDICTIONS

# ==========================================



predictions_response = call_api_cached(

    "/predictions",

    params={

        "start": start_iso,

        "end": end_iso,

        "limit": 5000,

    }

)





if not predictions_response["success"]:



    st.error(

        "Impossible de charger "

        "les prédictions."

    )



    st.code(

        predictions_response["error"]

    )



    st.stop()





# ==========================================

# DATAFRAME

# ==========================================



prediction_data = (

    predictions_response["data"]

)





predictions_df = pd.DataFrame(

    prediction_data["predictions"]

)



if predictions_df.empty:



    st.warning(

        "Aucune prédiction disponible "

        "pour cette période."

    )



    st.stop()



predictions_df["timestamp_utc"] = (

    pd.to_datetime(

        predictions_df["timestamp_utc"],

        utc=True

    )

)





# ==========================================

# LOAD COMPARISON PREDICTIONS

# ==========================================



if compare_enabled:



    compare_response = call_api_cached(

        "/predictions",

        params={

            "start": compare_start_iso,

            "end": compare_end_iso,

            "limit": 5000,

        }

    )





    if not compare_response["success"]:



        st.error(

            "Impossible de charger "

            "la période B."

        )



        st.code(

            compare_response["error"]

        )



        st.stop()





    compare_data = (

        compare_response["data"]

    )





    compare_df = pd.DataFrame(

        compare_data["predictions"]

    )





    if compare_df.empty:



        st.warning(

            "Aucune donnée disponible "

            "pour la période B."

        )



        st.stop()





    compare_df[

        "timestamp_utc"

    ] = pd.to_datetime(

        compare_df[

            "timestamp_utc"

        ],

        utc=True,

    )









# ==========================================

# DATA PREVIEW

# ==========================================



st.subheader(

    "Données chargées"

)





data_col1, data_col2 = st.columns(2)





with data_col1:



    st.metric(

        "Nombre de points",

        len(predictions_df)

    )





with data_col2:



    st.metric(

        "Latence API",

        (

            f"{predictions_response['latency_ms']:.1f} ms"

        )

    )





st.dataframe(

    predictions_df.head(),

    width="stretch"

)



# ==========================================

# REAL VS PREDICTED CHART

# ==========================================



st.subheader(

    "Consommation réelle vs prédiction"

)





fig = go.Figure()





fig.add_trace(

    go.Scatter(

        x=predictions_df[

            "timestamp_utc"

        ],

        y=predictions_df[

            "actual_mw"

        ],

        mode="lines",

        name="Consommation réelle",

    )

)





fig.add_trace(

    go.Scatter(

        x=predictions_df[

            "timestamp_utc"

        ],

        y=predictions_df[

            "prediction_mw"

        ],

        mode="lines",

        name="Prédiction",

        customdata=predictions_df[

            [

                "error_mw",

                "error_percent",

            ]

        ],

        hovertemplate=(

            "Prédiction : %{y:.0f} MW"

            "<br>Erreur : %{customdata[0]:.1f} MW"

            "<br>Erreur : %{customdata[1]:.3f} %"

            "<extra></extra>"

        ),

    )

)





fig.update_layout(

    xaxis_title="Temps",

    yaxis_title="Consommation (MW)",

    hovermode="x unified",

    height=520,

)





st.plotly_chart(

    fig,

    width="stretch"

)



# ==========================================

# DYNAMIC METRICS

# ==========================================



errors = (

    predictions_df["actual_mw"]

    - predictions_df["prediction_mw"]

)





rmse_period = np.sqrt(

    np.mean(

        errors ** 2

    )

)





mae_period = np.mean(

    np.abs(

        errors

    )

)





mape_period = np.mean(

    np.abs(

        errors

        / predictions_df["actual_mw"]

    )

) * 100





bias_period = np.mean(

    errors

)





# ==========================================

# COMPARISON PERIOD METRICS

# ==========================================



if compare_enabled:



    compare_errors = (

        compare_df["actual_mw"]

        - compare_df["prediction_mw"]

    )





    compare_rmse = np.sqrt(

        np.mean(

            compare_errors ** 2

        )

    )





    compare_mae = np.mean(

        np.abs(

            compare_errors

        )

    )





    compare_mape = (

        np.mean(

            np.abs(

                compare_errors

                / compare_df[

                    "actual_mw"

                ]

            )

        )

        * 100

    )





    compare_bias = np.mean(

        compare_errors

    )





    compare_mean_consumption = (

        compare_df[

            "actual_mw"

        ].mean()

    )





    compare_peak = (

        compare_df[

            "actual_mw"

        ].max()

    )



st.subheader(

    "Performance sur la période sélectionnée"

)





m1, m2, m3, m4 = st.columns(4)





with m1:



    st.metric(

        "RMSE période",

        f"{rmse_period:.2f} MW"

    )





with m2:



    st.metric(

        "MAE période",

        f"{mae_period:.2f} MW"

    )





with m3:



    st.metric(

        "MAPE période",

        f"{mape_period:.3f} %"

    )





with m4:



    st.metric(

        "Biais moyen",

        f"{bias_period:.2f} MW"

    )



st.subheader(

    "Erreur au cours du temps"

)





fig_error = go.Figure()





fig_error.add_trace(

    go.Scatter(

        x=predictions_df[

            "timestamp_utc"

        ],

        y=predictions_df[

            "error_mw"

        ],

        mode="lines",

        name="Erreur",

    )

)



fig_error.add_hline(

    y=0,

    line_dash="dash"

)



fig_error.update_layout(

    title=(

        f"RMSE = {rmse_period:.1f} MW"

        f" | MAE = {mae_period:.1f} MW"

    ),

    xaxis_title="Temps",

    yaxis_title="Erreur (MW)",

    height=350,

)





st.plotly_chart(

    fig_error,

    width="stretch"

)



# ==========================================

# ERROR DISTRIBUTION

# ==========================================



st.subheader(

    "Distribution des erreurs"

)





fig_hist = go.Figure()





fig_hist.add_trace(

    go.Histogram(

        x=predictions_df[

            "error_mw"

        ],

        nbinsx=30,

        name="Erreur",

    )

)





fig_hist.update_layout(

    xaxis_title="Erreur (MW)",

    yaxis_title="Fréquence",

    height=400,

)





st.plotly_chart(

    fig_hist,

    width="stretch"

)





# ==========================================

# ERROR QUANTILES

# ==========================================



absolute_errors = np.abs(

    predictions_df[

        "error_mw"

    ]

)





q50 = np.quantile(

    absolute_errors,

    0.50

)



q90 = np.quantile(

    absolute_errors,

    0.90

)



q95 = np.quantile(

    absolute_errors,

    0.95

)



q99 = np.quantile(

    absolute_errors,

    0.99

)





st.subheader(

    "Quantiles de l'erreur absolue"

)





q1, q2, q3, q4 = (

    st.columns(4)

)





with q1:



    st.metric(

        "50 %",

        f"{q50:.1f} MW"

    )





with q2:



    st.metric(

        "90 %",

        f"{q90:.1f} MW"

    )





with q3:



    st.metric(

        "95 %",

        f"{q95:.1f} MW"

    )





with q4:



    st.metric(

        "99 %",

        f"{q99:.1f} MW"

    )





# ==========================================

# OVER / UNDER ESTIMATION

# ==========================================



n_under = int(

    (

        predictions_df[

            "error_mw"

        ] > 0

    ).sum()

)





n_over = int(

    (

        predictions_df[

            "error_mw"

        ] < 0

    ).sum()

)





n_exact = int(

    (

        predictions_df[

            "error_mw"

        ] == 0

    ).sum()

)





st.subheader(

    "Type d'erreur"

)





e1, e2, e3 = st.columns(3)





with e1:



    st.metric(

        "Sous-estimations",

        n_under

    )





with e2:



    st.metric(

        "Surestimations",

        n_over

    )





with e3:



    st.metric(

        "Erreurs nulles",

        n_exact

    )



# ==========================================

# LARGEST ERRORS

# ==========================================



largest_errors = (

    predictions_df

    .assign(

        abs_error_mw=lambda df:

            np.abs(

                df["error_mw"]

            )

    )

    .sort_values(

        "abs_error_mw",

        ascending=False

    )

    .head(10)

)



st.subheader(

    "10 plus grosses erreurs"

)





st.dataframe(

    largest_errors[

        [

            "timestamp_utc",

            "actual_mw",

            "prediction_mw",

            "error_mw",

            "error_percent",

            "abs_error_mw",

        ]

    ],

    width="stretch"

)



worst_row = (

    largest_errors.iloc[0]

)



st.subheader(

    "Pire erreur de la période"

)





w1, w2, w3, w4 = (

    st.columns(4)

)





with w1:



    st.metric(

        "Réel",

        f"{worst_row['actual_mw']:.0f} MW"

    )





with w2:



    st.metric(

        "Prédit",

        f"{worst_row['prediction_mw']:.0f} MW"

    )





with w3:



    st.metric(

        "Erreur",

        f"{worst_row['error_mw']:.1f} MW"

    )





with w4:



    st.metric(

        "Erreur relative",

        f"{worst_row['error_percent']:.3f} %"

    )



st.caption(

    "Timestamp UTC : "

    f"{worst_row['timestamp_utc']}"

)



# ==========================================

# LARGE ERROR DETECTION

# ==========================================



predictions_df[

    "large_error"

] = (

    np.abs(

        predictions_df[

            "error_mw"

        ]

    )

    > q95

)



large_error_count = int(

    predictions_df[

        "large_error"

    ].sum()

)





st.metric(

    "Erreurs > quantile 95 %",

    large_error_count

)



# ==========================================

# LOCAL R2

# ==========================================



actual = predictions_df[

    "actual_mw"

].to_numpy()



predicted = predictions_df[

    "prediction_mw"

].to_numpy()





ss_res = np.sum(

    (

        actual

        - predicted

    ) ** 2

)



ss_tot = np.sum(

    (

        actual

        - np.mean(actual)

    ) ** 2

)





if ss_tot > 0:



    r2_period = (

        1

        - ss_res / ss_tot

    )



else:



    r2_period = np.nan



st.metric(

    "R² période",

    (

        f"{r2_period:.6f}"

        if np.isfinite(r2_period)

        else "N/A"

    )

)



max_abs_error = np.max(

    np.abs(errors)

)



st.metric(

    "Erreur absolue max",

    f"{max_abs_error:.1f} MW"

)



rmse_global = float(

    model_data["rmse"]

)



mae_global = float(

    model_data["mae"]

)



mape_global = float(

    model_data["mape"]

)



r2_global = float(

    model_data["r2"]

)



rmse_delta = (

    rmse_period

    - rmse_global

)



mae_delta = (

    mae_period

    - mae_global

)



mape_delta = (

    mape_period

    - mape_global

)



r2_delta = (

    r2_period

    - r2_global

)



st.subheader(

    "Comparaison avec la référence 2025"

)





c1, c2, c3, c4 = st.columns(4)





with c1:



    st.metric(

        "RMSE",

        f"{rmse_period:.2f} MW",

        delta=(

            f"{rmse_delta:+.2f} MW"

        ),

        delta_color="inverse",

    )





with c2:



    st.metric(

        "MAE",

        f"{mae_period:.2f} MW",

        delta=(

            f"{mae_delta:+.2f} MW"

        ),

        delta_color="inverse",

    )





with c3:



    st.metric(

        "MAPE",

        f"{mape_period:.3f} %",

        delta=(

            f"{mape_delta:+.3f} pt"

        ),

        delta_color="inverse",

    )





with c4:



    st.metric(

        "R²",

        f"{r2_period:.6f}",

        delta=(

            f"{r2_delta:+.6f}"

        ),

    )



if rmse_period < rmse_global:



    st.success(

        "La période sélectionnée présente "

        "un RMSE inférieur à la moyenne 2025."

    )



else:



    st.warning(

        "La période sélectionnée présente "

        "un RMSE supérieur à la moyenne 2025."

    )



rmse_relative_change = (

    100

    * (

        rmse_period

        - rmse_global

    )

    / rmse_global

)



st.caption(

    "Écart relatif du RMSE : "

    f"{rmse_relative_change:+.2f} %"

)



comparison_df = pd.DataFrame({



    "Métrique": [

        "RMSE",

        "MAE",

        "MAPE",

        "R²",

    ],



    "Période sélectionnée": [

        rmse_period,

        mae_period,

        mape_period,

        r2_period,

    ],



    "Référence 2025": [

        rmse_global,

        mae_global,

        mape_global,

        r2_global,

    ],



    "Écart": [

        rmse_delta,

        mae_delta,

        mape_delta,

        r2_delta,

    ],

})





st.dataframe(

    comparison_df,

    width="stretch",

    hide_index=True,

)



mean_consumption = np.mean(

    predictions_df[

        "actual_mw"

    ]

)



nrmse_period = (

    rmse_period

    / mean_consumption

    * 100

)



st.metric(

    "NRMSE",

    f"{nrmse_period:.3f} %"

)



# ==========================================

# LOAD WEATHER / CALENDAR CONTEXT

# ==========================================



context_response = call_api_cached(

    "/context",

    params={

        "start": start_iso,

        "end": end_iso,

        "limit": 5000,

    }

)





if not context_response["success"]:



    st.warning(

        "Le contexte météo/calendrier "

        "n'est pas disponible."

    )



    context_df = pd.DataFrame()



else:



    context_df = pd.DataFrame(

        context_response[

            "data"

        ][

            "context"

        ]

    )



if not context_df.empty:



    context_df[

        "timestamp_utc"

    ] = pd.to_datetime(

        context_df[

            "timestamp_utc"

        ],

        utc=True

    )



if not context_df.empty:



    dashboard_df = (

        predictions_df.merge(

            context_df,

            on="timestamp_utc",

            how="left",

        )

    )



else:



    dashboard_df = (

        predictions_df.copy()

    )





# ==========================================

# LOAD COMPARISON CONTEXT

# ==========================================



if compare_enabled:



    compare_context_response = call_api_cached(

        "/context",

        params={

            "start": compare_start_iso,

            "end": compare_end_iso,

            "limit": 5000,

        }

    )





    if compare_context_response[

        "success"

    ]:



        compare_context_df = pd.DataFrame(

            compare_context_response[

                "data"

            ][

                "context"

            ]

        )





        if not compare_context_df.empty:



            compare_context_df[

                "timestamp_utc"

            ] = pd.to_datetime(

                compare_context_df[

                    "timestamp_utc"

                ],

                utc=True,

            )





            compare_dashboard_df = (

                compare_df.merge(

                    compare_context_df,

                    on="timestamp_utc",

                    how="left",

                )

            )





            compare_temp_mean = (

                compare_dashboard_df[

                    "temperature_france"

                ].mean()

            )



        else:



            compare_dashboard_df = (

                compare_df.copy()

            )



            compare_temp_mean = np.nan



    else:



        compare_dashboard_df = (

            compare_df.copy()

        )



        compare_temp_mean = np.nan







if (

    "temperature_france"

    in dashboard_df.columns

):



    st.subheader(

        "Température France"

    )





    fig_temp = go.Figure()





    fig_temp.add_trace(

        go.Scatter(

            x=dashboard_df[

                "timestamp_utc"

            ],

            y=dashboard_df[

                "temperature_france"

            ],

            mode="lines",

            name="Température France",

        )

    )





    fig_temp.update_layout(

        xaxis_title="Temps",

        yaxis_title="Température (°C)",

        height=350,

    )





    st.plotly_chart(

        fig_temp,

        width="stretch",

    )















if (

    "temperature_france"

    in dashboard_df.columns

):



    st.subheader(

        "Consommation et température"

    )





    fig_energy_temp = go.Figure()





    fig_energy_temp.add_trace(

        go.Scatter(

            x=dashboard_df[

                "timestamp_utc"

            ],

            y=dashboard_df[

                "actual_mw"

            ],

            name="Consommation",

            yaxis="y",

        )

    )





    fig_energy_temp.add_trace(

        go.Scatter(

            x=dashboard_df[

                "timestamp_utc"

            ],

            y=dashboard_df[

                "temperature_france"

            ],

            name="Température",

            yaxis="y2",

        )

    )





    fig_energy_temp.update_layout(



        xaxis={

            "title": "Temps"

        },



        yaxis={

            "title": "Consommation (MW)"

        },



        yaxis2={

            "title": "Température (°C)",

            "overlaying": "y",

            "side": "right",

        },



        hovermode="x unified",



        height=450,

    )





    st.plotly_chart(

        fig_energy_temp,

        width="stretch",

    )













if (

    "local_hour"

    in dashboard_df.columns

):



    hourly_error = (

        dashboard_df

        .groupby(

            "local_hour"

        )

        .agg(

            mae_mw=(

                "error_mw",

                lambda x:

                    np.mean(

                        np.abs(x)

                    )

            ),



            mean_error_mw=(

                "error_mw",

                "mean"

            ),



            n=(

                "error_mw",

                "size"

            ),

        )

        .reset_index()

    )













    st.subheader(

        "Erreur selon l'heure locale"

    )





    fig_hour = go.Figure()





    fig_hour.add_trace(

        go.Bar(

            x=hourly_error[

                "local_hour"

            ],

            y=hourly_error[

                "mae_mw"

            ],

            name="MAE",

        )

    )





    fig_hour.update_layout(

        xaxis_title="Heure locale",

        yaxis_title="MAE (MW)",

        height=400,

    )





    st.plotly_chart(

        fig_hour,

        width="stretch",

    )















if (

    "is_weekend"

    in dashboard_df.columns

):



    weekend_analysis = (

        dashboard_df

        .assign(

            abs_error=lambda df:

                np.abs(

                    df["error_mw"]

                )

        )

        .groupby(

            "is_weekend"

        )

        .agg(

            mae_mw=(

                "abs_error",

                "mean"

            ),



            mean_consumption_mw=(

                "actual_mw",

                "mean"

            ),



            n=(

                "actual_mw",

                "size"

            ),

        )

        .reset_index()

    )





    weekend_analysis[

        "Type"

    ] = weekend_analysis[

        "is_weekend"

    ].map({

        False: "Semaine",

        True: "Week-end",

    })





    st.subheader(

        "Semaine vs week-end"

    )





    st.dataframe(

        weekend_analysis[

            [

                "Type",

                "mae_mw",

                "mean_consumption_mw",

                "n",

            ]

        ],

        width="stretch",

        hide_index=True,

    )















if not context_df.empty:



    context_col1, context_col2, context_col3 = (

        st.columns(3)

    )





    with context_col1:



        st.metric(

            "Température moyenne",

            (

                f"{dashboard_df['temperature_france'].mean():.1f} °C"

            ),

        )





    with context_col2:



        st.metric(

            "Température min",

            (

                f"{dashboard_df['temperature_france'].min():.1f} °C"

            ),

        )





    with context_col3:



        weekend_share = (

            100

            * dashboard_df[

                "is_weekend"

            ].mean()

        )



        st.metric(

            "Points week-end",

            f"{weekend_share:.1f} %",

        )



# ==========================================

# PERIOD COMPARISON

# ==========================================



if compare_enabled:



    st.divider()



    st.subheader(

        "Comparaison des périodes"

    )





    temp_a = (

        dashboard_df[

            "temperature_france"

        ].mean()

        if "temperature_france"

        in dashboard_df.columns

        else np.nan

    )





    comparison_periods = pd.DataFrame({



        "Indicateur": [

            "Nombre de points",

            "Consommation moyenne (MW)",

            "Pic de consommation (MW)",

            "RMSE (MW)",

            "MAE (MW)",

            "MAPE (%)",

            "Biais moyen (MW)",

            "Température moyenne (°C)",

        ],



        "Période A": [

            len(predictions_df),



            predictions_df[

                "actual_mw"

            ].mean(),



            predictions_df[

                "actual_mw"

            ].max(),



            rmse_period,



            mae_period,



            mape_period,



            bias_period,



            temp_a,

        ],



        "Période B": [

            len(compare_df),



            compare_mean_consumption,



            compare_peak,



            compare_rmse,



            compare_mae,



            compare_mape,



            compare_bias,



            compare_temp_mean,

        ],

    })





    st.dataframe(

        comparison_periods,

        width="stretch",

        hide_index=True,

    )





# ==========================================

# COMPARISON CHART

# ==========================================



if compare_enabled:



    st.subheader(

        "Comparaison des profils "

        "de consommation"

    )





    fig_compare = go.Figure()





    fig_compare.add_trace(

        go.Scatter(

            x=np.arange(

                len(predictions_df)

            ),

            y=predictions_df[

                "actual_mw"

            ],

            mode="lines",

            name="Période A",

        )

    )





    fig_compare.add_trace(

        go.Scatter(

            x=np.arange(

                len(compare_df)

            ),

            y=compare_df[

                "actual_mw"

            ],

            mode="lines",

            name="Période B",

        )

    )





    fig_compare.update_layout(

        xaxis_title=(

            "Pas de temps de 30 minutes"

        ),

        yaxis_title=(

            "Consommation (MW)"

        ),

        hovermode="x unified",

        height=450,

    )





    st.plotly_chart(

        fig_compare,

        width="stretch",

    )







# ==========================================

# COMPARISON SUMMARY

# ==========================================



if compare_enabled:



    st.subheader(

        "Résumé de la comparaison"

    )





    a1, a2, a3 = st.columns(3)



    b1, b2, b3 = st.columns(3)





    with a1:



        st.metric(

            "RMSE — A",

            f"{rmse_period:.1f} MW"

        )





    with a2:



        st.metric(

            "Consommation moyenne — A",

            (

                f"{predictions_df['actual_mw'].mean():.0f} MW"

            )

        )





    with a3:



        st.metric(

            "Température moyenne — A",

            (

                f"{temp_a:.1f} °C"

                if np.isfinite(temp_a)

                else "N/A"

            )

        )





    with b1:



        st.metric(

            "RMSE — B",

            f"{compare_rmse:.1f} MW"

        )





    with b2:



        st.metric(

            "Consommation moyenne — B",

            (

                f"{compare_mean_consumption:.0f} MW"

            )

        )





    with b3:



        st.metric(

            "Température moyenne — B",

            (

                f"{compare_temp_mean:.1f} °C"

                if np.isfinite(

                    compare_temp_mean

                )

                else "N/A"

            )

        )





# ==========================================

# DASHBOARD PERFORMANCE

# ==========================================



dashboard_elapsed_ms = (

    time.perf_counter()

    - dashboard_start

) * 1000





st.divider()



st.caption(

    "Temps d'exécution serveur "

    f"du dashboard : "

    f"{dashboard_elapsed_ms:.1f} ms"

)







perf1, perf2, perf3 = (

    st.columns(3)

)





with perf1:



    st.metric(

        "Points période A",

        len(predictions_df)

    )





with perf2:



    st.metric(

        "API prédictions A",

        (

            f"{predictions_response['latency_ms']:.1f} ms"

        )

    )





with perf3:



    if compare_enabled:



        st.metric(

            "Points période B",

            len(compare_df)

        )



    else:



        st.metric(

            "Comparaison",

            "OFF"

        )