import pandas as pd

NUMERIC_FEATURES = [
    "age", "monthly_income", "years_at_company", "years_in_current_role",
    "years_since_promotion", "job_satisfaction", "work_life_balance",
    "overtime", "distance_from_home", "training_times_last_year"
]
CATEGORICAL_FEATURES = ["department", "job_role", "business_travel"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "attrition"

def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = df.copy()
    for col in NUMERIC_FEATURES:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result = result.dropna(subset=FEATURES + [TARGET]).copy()
    if result.empty:
        raise ValueError("No valid records remain after validation")
    if not result["overtime"].isin([0,1]).all():
        raise ValueError("overtime must be encoded as 0 or 1")
    if not result[TARGET].isin([0,1]).all():
        raise ValueError("attrition target must be encoded as 0 or 1")
    return result
