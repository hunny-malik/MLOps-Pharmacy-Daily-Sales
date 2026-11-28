import datetime as dt  # noqa: F401  (kept for future use)
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = Path(os.getenv("PARAMS_PATH", ROOT / "params.yaml"))


def _load_service_params() -> dict:
    """Read the `service:` block of params.yaml (falls back to defaults)."""
    try:
        with open(PARAMS_PATH, encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("service", {})
    except FileNotFoundError:
        return {}


_service = _load_service_params()
STOCK_LINES_MAX = int(_service.get("stock_lines_max", 5500))
TEMP_C_MIN = float(_service.get("temp_c_min", 2.5))


class PredictionRequest(BaseModel):
    """Input for POST /predict. Field names match src/mp108/pipeline.py."""

    model_config = ConfigDict(extra="forbid")  # unknown fields -> 422

    flu_cases_reported: int = Field(ge=0)
    temp_c: float = Field(ge=TEMP_C_MIN, description=f"Temperature in C, >= {TEMP_C_MIN}")
    nearby_clinics: int = Field(ge=0, lt=9999, description="9999 is the sentinel value, rejected")
    discount_pct: float = Field(ge=0, le=100)
    stock_lines: int = Field(ge=0, le=STOCK_LINES_MAX, description=f"<= {STOCK_LINES_MAX}")
    distance_to_hospital_km: float = Field(ge=0, description="Distance in kilometres (not metres)")
    shop_type: str = Field(min_length=1)
    locality: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    predicted_units_sold: float