"""
Solar Sense – Chengalpattu
Configuration & Constants Module
Course: 21CEO310T (Global Warming & Climate Change)
"""

from pathlib import Path
import json

# Project Metadata
PROJECT_TITLE = "Solar Sense – Chengalpattu"
PAGE_ICON = "☀️"
COURSE_CODE = "21CEO310T"
COURSE_NAME = "Global Warming & Climate Change"
LOCATION_NAME = "Chengalpattu, Tamil Nadu, India"
LATITUDE = 12.69
LONGITUDE = 79.98

# NASA POWER Daily API Configuration
NASA_POWER_API_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_RAW_PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",
]

# Exact feature list in order expected by trained scikit-learn models
FEATURES = [
    "T2M",
    "T_RANGE",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",
    "doy_sin",
    "doy_cos",
]

TARGET = "ALLSKY_SFC_SW_DWN"

# Directory & Artifact Paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CONFIG_JSON_PATH = ARTIFACTS_DIR / "config.json"
CLEAN_DATA_PATH = ARTIFACTS_DIR / "clean_data.csv"
METRICS_PATH = ARTIFACTS_DIR / "metrics.csv"
TEST_PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"
RF_PERM_IMPORTANCE_PATH = ARTIFACTS_DIR / "rf_permutation_importance.csv"
SHAP_SUMMARY_PATH = ARTIFACTS_DIR / "shap_summary.png"

MODEL_PATHS = {
    "Linear Regression": ARTIFACTS_DIR / "linear_regression.joblib",
    "SVR": ARTIFACTS_DIR / "svr.joblib",
    "Random Forest": ARTIFACTS_DIR / "random_forest.joblib",
}

# Load settings from config.json if available
def load_json_config() -> dict:
    if CONFIG_JSON_PATH.exists():
        try:
            with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

_cfg = load_json_config()

# Domain / Climate Defaults
DEFAULT_SYSTEM_KWP: float = float(_cfg.get("system_kwp", 5.0))
DEFAULT_PERFORMANCE_RATIO: float = float(_cfg.get("performance_ratio", 0.75))
DEFAULT_EMISSION_FACTOR: float = float(_cfg.get("emission_factor", 0.71))  # tCO2 per MWh (CEA India baseline)
BEST_MODEL_NAME: str = str(_cfg.get("best_model", "SVR"))
TRAIN_END_YEAR: int = int(_cfg.get("train_end_year", 2022))

# Environmental equivalence: 1 mature tree absorbs ~21.77 kg (0.02177 t) CO2/year
TREE_ANNUAL_CO2_TONNES: float = 0.02177

# Variable descriptions and units for tooltips and tables
VARIABLE_DESCRIPTIONS = {
    "ALLSKY_SFC_SW_DWN": {
        "name": "All Sky Surface Shortwave Downward Irradiance",
        "unit": "kWh/m²/day",
        "desc": "Total solar energy incident per square meter per day on a horizontal surface, accounting for clouds and atmospheric transmission.",
    },
    "T2M": {
        "name": "Temperature at 2 Meters",
        "unit": "°C",
        "desc": "Daily average air temperature at 2 meters above ground level.",
    },
    "T2M_MAX": {
        "name": "Maximum Temperature at 2 Meters",
        "unit": "°C",
        "desc": "Maximum daily air temperature at 2 meters.",
    },
    "T2M_MIN": {
        "name": "Minimum Temperature at 2 Meters",
        "unit": "°C",
        "desc": "Minimum daily air temperature at 2 meters.",
    },
    "T_RANGE": {
        "name": "Diurnal Temperature Range (T_MAX - T_MIN)",
        "unit": "°C",
        "desc": "Difference between daily maximum and minimum temperature. Clear sunny days produce larger diurnal ranges; cloudy/rainy days produce narrow ranges.",
    },
    "RH2M": {
        "name": "Relative Humidity at 2 Meters",
        "unit": "%",
        "desc": "Percentage of moisture saturation in the air. High humidity strongly correlates with cloud cover and atmospheric water vapor attenuation.",
    },
    "WS2M": {
        "name": "Wind Speed at 2 Meters",
        "unit": "m/s",
        "desc": "Daily average wind speed at 2 meters. Affects cloud advection and panel cooling efficiency.",
    },
    "PRECTOTCORR": {
        "name": "Corrected Total Precipitation",
        "unit": "mm/day",
        "desc": "Daily total rain/precipitation. Rainfall directly signifies dense cloud cover that severely attenuates ground-level solar irradiance.",
    },
    "doy_sin": {
        "name": "Day of Year Sine Cycle",
        "unit": "unitless [-1, 1]",
        "desc": "sin(2π · dayofyear / 365.25), encoding seasonal solar declination angles without artificial end-of-year discontinuities.",
    },
    "doy_cos": {
        "name": "Day of Year Cosine Cycle",
        "unit": "unitless [-1, 1]",
        "desc": "cos(2π · dayofyear / 365.25), pairing with doy_sin for a smooth continuous circular 2D seasonal coordinate.",
    },
}
