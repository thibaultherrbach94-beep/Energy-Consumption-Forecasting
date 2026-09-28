from pydantic import BaseModel
from pydantic import BaseModel, ConfigDict
from datetime import datetime


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class DatabaseHealthResponse(BaseModel):
    status: str
    database: str


class ModelMetricsResponse(BaseModel):
    model_config = ConfigDict(
    protected_namespaces=()
)
    
    model_name: str
    evaluation_period: str

    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float


from datetime import datetime


class PredictionResponse(BaseModel):

    model_config = ConfigDict(
    protected_namespaces=()
)
    
    timestamp_utc: datetime
    model_name: str

    actual_mw: float
    prediction_mw: float

    error_mw: float
    error_percent: float


class PredictionListResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]

class ContextPointResponse(BaseModel):

    model_config = ConfigDict(
        protected_namespaces=()
    )

    timestamp_utc: datetime
    local_timestamp: datetime
    temperature_france: float | None
    local_hour: int
    day_of_week: int
    month: int
    is_weekend: bool
    is_year_end_window: bool


class ContextListResponse(BaseModel):

    count: int
    context: list[ContextPointResponse]