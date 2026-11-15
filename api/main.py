from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from talentiq.model import train, score_employee
from talentiq.validation import FEATURES

ROOT = Path(__file__).resolve().parents[1]
DATA = pd.read_csv(ROOT / "data" / "synthetic_employee_data.csv")
MODEL, METRICS = train(DATA)
app = FastAPI(title="TalentIQ API", version="1.0.0",
              description="Synthetic workforce analytics demo; not for employment decisions.")

class EmployeeInput(BaseModel):
    age: int = Field(ge=18, le=70)
    monthly_income: float = Field(gt=0, le=100000)
    years_at_company: float = Field(ge=0, le=45)
    years_in_current_role: float = Field(ge=0, le=30)
    years_since_promotion: float = Field(ge=0, le=30)
    job_satisfaction: int = Field(ge=1, le=4)
    work_life_balance: int = Field(ge=1, le=4)
    overtime: int = Field(ge=0, le=1)
    distance_from_home: float = Field(ge=1, le=100)
    training_times_last_year: int = Field(ge=0, le=20)
    department: str
    job_role: str
    business_travel: str

@app.get("/health")
def health():
    return {"status":"ok", "mode":"synthetic_demo", "employment_decision_use":False}

@app.get("/model-info")
def model_info():
    return {"model":"RandomForestClassifier", "features":FEATURES, "metrics":METRICS,
            "warning":"Synthetic-data metrics are not real-world validation."}

@app.post("/predict-attrition")
def predict_attrition(employee: EmployeeInput):
    try:
        return score_employee(MODEL, employee.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
