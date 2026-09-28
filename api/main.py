from datetime import datetime

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
)

import plotly.graph_objects as go

from fastapi.responses import HTMLResponse

from api.schemas import (
    HealthResponse,
    DatabaseHealthResponse,
    ModelMetricsResponse,
    PredictionResponse,
    PredictionListResponse,
    ContextPointResponse,
    ContextListResponse,
)

from src.database.postgres import (
    get_connection,
)



APP_VERSION = "0.2.0"

MODEL_NAME = (
    "Ridge + MLP1 + Patch Transformer "
    "+ Refined Event Expert"
)

app = FastAPI(
    title="Energy Forecasting API",
    description=(
        "API de prévision de la consommation "
        "électrique française."
    ),
    version=APP_VERSION,
)


# ============================================================
# HEALTH CHECK API
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health_check():

    return HealthResponse(
        status="ok",
        service="energy-forecasting-api",
        version=APP_VERSION,
    )


# ============================================================
# HEALTH CHECK DATABASE
# ============================================================

@app.get(
    "/database/health",
    response_model=DatabaseHealthResponse,
)
def database_health():

    try:

        with get_connection() as conn:

            with conn.cursor() as cur:

                cur.execute("""
                    SELECT
                        current_database();
                """)

                database = cur.fetchone()[0]

        return DatabaseHealthResponse(
            status="ok",
            database=database,
        )

    except Exception:

        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        )


# ============================================================
# MODEL METRICS
# ============================================================

@app.get(
    "/model/metrics",
    response_model=ModelMetricsResponse,
)
def model_metrics():

    try:

        with get_connection() as conn:

            with conn.cursor() as cur:

                cur.execute("""
                    SELECT
                        model_name,
                        evaluation_period,
                        mse,
                        rmse,
                        mae,
                        mape,
                        r2

                    FROM energy.model_metrics

                    WHERE
                        evaluation_period =
                        '2025-development'

                    ORDER BY
                        created_at DESC

                    LIMIT 1;
                """)

                row = cur.fetchone()

    except Exception:

        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        )

    if row is None:

        raise HTTPException(
            status_code=404,
            detail="Model metrics not found",
        )

    return ModelMetricsResponse(
        model_name=row[0],
        evaluation_period=row[1],
        mse=row[2],
        rmse=row[3],
        mae=row[4],
        mape=row[5],
        r2=row[6],
    )


@app.get(
    "/predictions",
    response_model=PredictionListResponse,
)
@app.get(
    "/predictions",
    response_model=PredictionListResponse,
)
def get_predictions(
    start: datetime = Query(...),
    end: datetime = Query(...),
    limit: int = Query(
        default=2000,
        ge=1,
        le=5000,
    ),
):

    if end <= start:
        raise HTTPException(
            status_code=400,
            detail="end must be greater than start",
        )

    if start.year >= 2026 or end.year >= 2026:
        raise HTTPException(
            status_code=403,
            detail=(
                "2026 is reserved for "
                "the final test."
            ),
        )

    if (end - start).days > 31:
        raise HTTPException(
            status_code=400,
            detail=(
                "Maximum requested period "
                "is 31 days."
            ),
        )

    try:

        with get_connection() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    """
                    SELECT
                        timestamp_utc,
                        model_name,
                        actual_mw,
                        prediction_mw,

                        actual_mw
                        - prediction_mw
                            AS error_mw,

                        CASE
                            WHEN actual_mw <> 0
                            THEN
                                100.0
                                * (
                                    actual_mw
                                    - prediction_mw
                                )
                                / actual_mw
                            ELSE 0.0
                        END
                            AS error_percent

                    FROM energy.predictions

                    WHERE
                        model_name = %s

                    AND
                        data_split = 'development'

                    AND
                        timestamp_utc >= %s

                    AND
                        timestamp_utc < %s

                    ORDER BY
                        timestamp_utc

                    LIMIT %s;
                    """,
                    (
                        MODEL_NAME,
                        start,
                        end,
                        limit,
                    ),
                )

                rows = cur.fetchall()

    except Exception as exc:

        # Important pendant le développement :
        # on garde la vraie erreur dans le terminal.
        print(
            "ERREUR /predictions :",
            type(exc).__name__,
            str(exc),
        )

        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        )

    predictions = [

        PredictionResponse(
            timestamp_utc=row[0],
            model_name=row[1],
            actual_mw=float(row[2]),
            prediction_mw=float(row[3]),
            error_mw=float(row[4]),
            error_percent=float(row[5]),
        )

        for row in rows
    ]

    return PredictionListResponse(
        count=len(predictions),
        predictions=predictions,
    )

@app.get(
    "/context",
    response_model=ContextListResponse,
)
def get_context(
    start: datetime = Query(...),
    end: datetime = Query(...),
    limit: int = Query(
        default=5000,
        ge=1,
        le=5000,
    ),
):

    if end <= start:

        raise HTTPException(
            status_code=400,
            detail=(
                "end must be greater "
                "than start"
            ),
        )


    if (
        start.year >= 2026
        or end.year >= 2026
    ):

        raise HTTPException(
            status_code=403,
            detail=(
                "2026 is reserved "
                "for the final test."
            ),
        )


    if (end - start).days > 31:

        raise HTTPException(
            status_code=400,
            detail=(
                "Maximum requested "
                "period is 31 days."
            ),
        )


    try:

        with get_connection() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    """
                    SELECT
                        c.timestamp_utc,

                        c.timestamp_utc
                        AT TIME ZONE
                        'Europe/Paris'
                            AS local_timestamp,

                        w.temperature
                            AS temperature_france,

                        EXTRACT(
                            HOUR FROM
                            c.timestamp_utc
                            AT TIME ZONE
                            'Europe/Paris'
                        )::int
                            AS local_hour,

                        EXTRACT(
                            ISODOW FROM
                            c.timestamp_utc
                            AT TIME ZONE
                            'Europe/Paris'
                        )::int
                            AS day_of_week,

                        EXTRACT(
                            MONTH FROM
                            c.timestamp_utc
                            AT TIME ZONE
                            'Europe/Paris'
                        )::int
                            AS month,

                        (
                            EXTRACT(
                                ISODOW FROM
                                c.timestamp_utc
                                AT TIME ZONE
                                'Europe/Paris'
                            )::int
                            IN (6, 7)
                        )
                            AS is_weekend,

                        (
                            EXTRACT(
                                MONTH FROM
                                c.timestamp_utc
                                AT TIME ZONE
                                'Europe/Paris'
                            )::int = 12

                            AND

                            EXTRACT(
                                DAY FROM
                                c.timestamp_utc
                                AT TIME ZONE
                                'Europe/Paris'
                            )::int >= 20
                        )
                            AS is_year_end_window

                    FROM energy.consumption AS c

                    LEFT JOIN LATERAL (

                        SELECT
                            temperature

                        FROM energy.weather

                        WHERE
                            city = 'France'

                        AND
                            timestamp_utc
                            <= c.timestamp_utc

                        ORDER BY
                            timestamp_utc DESC

                        LIMIT 1

                    ) AS w
                    ON TRUE

                    WHERE
                        c.timestamp_utc >= %s

                    AND
                        c.timestamp_utc < %s

                    ORDER BY
                        c.timestamp_utc

                    LIMIT %s;
                    """,
                    (
                        start,
                        end,
                        limit,
                    ),
                )

                rows = cur.fetchall()

    except Exception as exc:

        print(
            "ERREUR /context :",
            type(exc).__name__,
            str(exc),
        )

        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        )


    context = [

        ContextPointResponse(
            timestamp_utc=row[0],
            local_timestamp=row[1],
            temperature_france=(
                float(row[2])
                if row[2] is not None
                else None
            ),
            local_hour=int(row[3]),
            day_of_week=int(row[4]),
            month=int(row[5]),
            is_weekend=bool(row[6]),
            is_year_end_window=bool(row[7]),
        )

        for row in rows
    ]


    return ContextListResponse(
        count=len(context),
        context=context,
    )

@app.get(
    "/prediction",
    response_model=PredictionResponse,
)
def get_prediction(
    timestamp: datetime = Query(...),
):

    if timestamp.year >= 2026:

        raise HTTPException(
            status_code=403,
            detail=(
                "2026 is reserved for "
                "the final test."
            ),
        )

    try:

        with get_connection() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    """
                    SELECT
                        timestamp_utc,
                        model_name,
                        actual_mw,
                        prediction_mw,

                        actual_mw
                        - prediction_mw
                            AS error_mw,

                        CASE

                            WHEN actual_mw != 0

                            THEN
                                100.0
                                * (
                                    actual_mw
                                    - prediction_mw
                                )
                                / actual_mw

                            ELSE NULL

                        END AS error_percent

                    FROM energy.predictions

                    WHERE
                        timestamp_utc = %s

                    AND
                        model_name = %s

                    AND
                        data_split =
                        'development'

                    LIMIT 1;
                    """,

                    (
                        timestamp,
                        MODEL_NAME,
                    ),
                )

                row = cur.fetchone()

    except Exception:

        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        )

    if row is None:

        raise HTTPException(
            status_code=404,
            detail="Prediction not found",
        )

    return PredictionResponse(
        timestamp_utc=row[0],
        model_name=row[1],
        actual_mw=row[2],
        prediction_mw=row[3],
        error_mw=row[4],
        error_percent=row[5],
    )


@app.get(
    "/predictions/chart",
    response_class=HTMLResponse,
)
def predictions_chart(

    start: datetime = Query(...),
    end: datetime = Query(...),

):

    # ========================================================
    # VALIDATION
    # ========================================================

    if end <= start:

        raise HTTPException(
            status_code=400,
            detail="end must be greater than start",
        )

    if (
        start.year >= 2026
        or end.year >= 2026
    ):

        raise HTTPException(
            status_code=403,
            detail=(
                "2026 is reserved for "
                "the final test."
            ),
        )

    if (end - start).days > 31:

        raise HTTPException(
            status_code=400,
            detail=(
                "Maximum requested period "
                "is 31 days."
            ),
        )

    # ========================================================
    # POSTGRESQL
    # ========================================================

    try:

        with get_connection() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    """
                    SELECT
                        timestamp_utc,
                        actual_mw,
                        prediction_mw,

                        CASE
                            WHEN actual_mw != 0
                            THEN
                                100.0
                                * (
                                    actual_mw
                                    - prediction_mw
                                )
                                / actual_mw
                            ELSE NULL
                        END
                            AS error_percent

                    FROM energy.predictions

                    WHERE
                        model_name = %s

                    AND
                        data_split = 'development'

                    AND
                        timestamp_utc >= %s

                    AND
                        timestamp_utc < %s

                    ORDER BY
                        timestamp_utc;
                    """,

                    (
                        MODEL_NAME,
                        start,
                        end,
                    ),
                )

                rows = cur.fetchall()

    except Exception:

        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        )

    if not rows:

        raise HTTPException(
            status_code=404,
            detail="No predictions found",
        )

    # ========================================================
    # DONNEES
    # ========================================================

    timestamps = [
        row[0]
        for row in rows
    ]

    actual = [
        row[1]
        for row in rows
    ]

    predictions = [
        row[2]
        for row in rows
    ]

    errors_percent = [
        row[3]
        for row in rows
    ]

    # ========================================================
    # GRAPHIQUE
    # ========================================================

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=actual,
            mode="lines",
            name="Valeur réelle",

            hovertemplate=(
                "Date : %{x}<br>"
                "Réel : %{y:.0f} MW"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=predictions,
            mode="lines",
            name="Prédiction",

            customdata=errors_percent,

            hovertemplate=(
                "Date : %{x}<br>"
                "Prédiction : %{y:.0f} MW<br>"
                "Erreur : %{customdata:.3f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=(
            "Consommation réelle vs prédiction"
        ),

        xaxis_title=(
            "Temps — UTC"
        ),

        yaxis_title=(
            "Consommation (MW)"
        ),

        hovermode="x unified",

        template="plotly_white",
    )

    html = fig.to_html(
    full_html=True,
    include_plotlyjs="cdn",
)

    return HTMLResponse(
        content=html,
        status_code=200,
    )