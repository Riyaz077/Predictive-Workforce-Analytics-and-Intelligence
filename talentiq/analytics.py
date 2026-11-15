import pandas as pd

def workforce_summary(df):
    total = int(len(df))
    attrition_rate = float(df["attrition"].mean()) if total else 0.0
    by_dept = (df.groupby("department", as_index=False)
                 .agg(employee_count=("employee_demo_id","count"),
                      synthetic_attrition_rate=("attrition","mean"))
                 .sort_values("synthetic_attrition_rate", ascending=False))
    by_role = (df.groupby("job_role", as_index=False)
                 .agg(employee_count=("employee_demo_id","count"),
                      synthetic_attrition_rate=("attrition","mean"))
                 .sort_values("synthetic_attrition_rate", ascending=False))
    return {
        "employee_count": total,
        "synthetic_attrition_rate": attrition_rate,
        "department_summary": by_dept,
        "role_summary": by_role
    }

def risk_distribution(df):
    return df["risk_band"].value_counts().rename_axis("risk_band").reset_index(name="employees")
