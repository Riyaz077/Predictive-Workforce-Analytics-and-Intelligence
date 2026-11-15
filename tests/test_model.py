from pathlib import Path
import pandas as pd
from talentiq.model import train, score_employee
ROOT = Path(__file__).resolve().parents[1]

def test_train_and_score():
    df = pd.read_csv(ROOT / "data" / "synthetic_employee_data.csv")
    model, metrics = train(df)
    employee = dict(age=32,monthly_income=6500,years_at_company=3,
        years_in_current_role=2,years_since_promotion=2,job_satisfaction=2,
        work_life_balance=2,overtime=1,distance_from_home=12,
        training_times_last_year=2,department="Research & Development",
        job_role="Research Scientist",business_travel="Travel_Rarely")
    result = score_employee(model, employee)
    assert result["illustrative_risk_band"] in {"Low","Medium","High"}
    assert 0 <= result["attrition_probability"] <= 1
    assert 0 <= metrics["f1"] <= 1
