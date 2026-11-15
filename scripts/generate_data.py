from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 3000
departments = ["Research & Development", "Sales", "Human Resources"]
roles = ["Research Scientist", "Laboratory Technician", "Sales Executive",
         "Sales Representative", "Manufacturing Director", "Healthcare Representative",
         "Manager", "Human Resources"]
travel = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]

age = rng.integers(20, 61, n)
income = np.clip(rng.lognormal(mean=8.55, sigma=.45, size=n), 1800, 22000).round(0)
tenure = rng.integers(0, 25, n)
current_role = np.minimum(tenure, rng.integers(0, 12, n))
promotion = np.minimum(tenure, rng.integers(0, 10, n))
satisfaction = rng.integers(1, 5, n)
wlb = rng.integers(1, 5, n)
overtime = rng.binomial(1, .28, n)
distance = rng.integers(1, 40, n)
training = rng.integers(0, 7, n)
dept = rng.choice(departments, n, p=[.65,.30,.05])
role = rng.choice(roles, n)
business_travel = rng.choice(travel, n, p=[.72,.18,.10])

# Artificial label-generation formula for demonstration only.
logit = (-2.0 + .75*overtime + .30*(satisfaction==1) + .25*(wlb==1)
         + .025*distance - .045*tenure + .35*(promotion==0)
         + .15*(income < 3500) + rng.normal(0,.55,n))
prob = 1/(1+np.exp(-logit))
attrition = rng.binomial(1, prob)

df = pd.DataFrame({
    "employee_demo_id":[f"TIQ{i+1:05d}" for i in range(n)],
    "age":age, "monthly_income":income.astype(int),
    "years_at_company":tenure, "years_in_current_role":current_role,
    "years_since_promotion":promotion, "job_satisfaction":satisfaction,
    "work_life_balance":wlb, "overtime":overtime,
    "distance_from_home":distance, "training_times_last_year":training,
    "department":dept, "job_role":role, "business_travel":business_travel,
    "attrition":attrition
})
out = Path(__file__).resolve().parents[1] / "data" / "synthetic_employee_data.csv"
df.to_csv(out, index=False)
print(f"Generated {len(df)} synthetic employee records: {out}")
