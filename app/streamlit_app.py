from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import plotly.express as px
import streamlit as st
from talentiq.model import train, score_employee
from talentiq.validation import FEATURES
from talentiq.analytics import workforce_summary

st.set_page_config(page_title="TalentIQ", page_icon="👥", layout="wide")
st.title("👥 TalentIQ")
st.caption("Employee Attrition Prediction and Workforce Intelligence System")
st.warning("Portfolio demo using synthetic employee data. Do not use predictions for real employment decisions.")

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "data" / "synthetic_employee_data.csv")
model, metrics = train(df)
with st.sidebar:
    st.header("Workforce filters")
    depts = st.multiselect("Department", sorted(df.department.unique()), default=sorted(df.department.unique()))
    roles = st.multiselect("Job role", sorted(df.job_role.unique()), default=sorted(df.job_role.unique()))
    overtime_filter = st.selectbox("Overtime", ["All", "Yes", "No"])
filtered = df[df.department.isin(depts) & df.job_role.isin(roles)].copy()
if overtime_filter == "Yes":
    filtered = filtered[filtered.overtime == 1]
elif overtime_filter == "No":
    filtered = filtered[filtered.overtime == 0]
if filtered.empty:
    st.info("No records match the selected filters.")
    st.stop()

X = filtered[FEATURES]
filtered["attrition_probability"] = model.predict_proba(X)[:,1]
filtered["risk_band"] = pd.cut(filtered.attrition_probability, bins=[-0.001,.35,.65,1.001],
                               labels=["Low","Medium","High"])
summary = workforce_summary(filtered)
a,b,c,d = st.columns(4)
a.metric("Employees in view", f"{len(filtered):,}")
b.metric("Synthetic attrition rate", f"{summary['synthetic_attrition_rate']*100:.1f}%")
c.metric("High demo risk", int((filtered.risk_band == "High").sum()))
d.metric("Model F1 (holdout)", f"{metrics['f1']:.2f}")

left,right = st.columns(2)
with left:
    st.subheader("Risk-band distribution")
    risk = filtered.risk_band.value_counts().rename_axis("risk_band").reset_index(name="employees")
    st.plotly_chart(px.pie(risk, names="risk_band", values="employees", hole=.4), use_container_width=True)
with right:
    st.subheader("Synthetic attrition by department")
    dept = filtered.groupby("department", as_index=False).attrition.mean()
    st.plotly_chart(px.bar(dept, x="department", y="attrition", title="Synthetic label rate"), use_container_width=True)

st.subheader("Workforce intelligence by job role")
role = filtered.groupby("job_role", as_index=False).agg(
    employees=("employee_demo_id","count"), synthetic_attrition_rate=("attrition","mean"))
st.plotly_chart(px.bar(role, x="job_role", y="synthetic_attrition_rate", color="employees"), use_container_width=True)

st.subheader("Demo employee risk scoring")
st.caption("Use fictional values only. The output is not a judgment about an employee.")
with st.form("employee_demo"):
    c1,c2,c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 70, 32)
        monthly_income = st.number_input("Monthly income (demo units)", 1000, 50000, 6500)
        years_at_company = st.number_input("Years at company", 0, 45, 3)
        years_in_current_role = st.number_input("Years in current role", 0, 30, 2)
    with c2:
        years_since_promotion = st.number_input("Years since promotion", 0, 30, 2)
        job_satisfaction = st.selectbox("Job satisfaction (1–4)", [1,2,3,4], index=1)
        work_life_balance = st.selectbox("Work-life balance (1–4)", [1,2,3,4], index=1)
        overtime = st.selectbox("Overtime", [0,1], format_func=lambda x: "No" if x == 0 else "Yes")
    with c3:
        distance_from_home = st.number_input("Distance from home (demo)", 1, 100, 12)
        training_times_last_year = st.number_input("Training sessions last year", 0, 20, 2)
        department = st.selectbox("Department", sorted(df.department.unique()))
        job_role = st.selectbox("Job role", sorted(df.job_role.unique()))
        business_travel = st.selectbox("Business travel", sorted(df.business_travel.unique()))
    submitted = st.form_submit_button("Estimate demo attrition score")
if submitted:
    employee = dict(age=age, monthly_income=monthly_income, years_at_company=years_at_company,
        years_in_current_role=years_in_current_role, years_since_promotion=years_since_promotion,
        job_satisfaction=job_satisfaction, work_life_balance=work_life_balance,
        overtime=overtime, distance_from_home=distance_from_home,
        training_times_last_year=training_times_last_year, department=department,
        job_role=job_role, business_travel=business_travel)
    result = score_employee(model, employee)
    st.metric("Illustrative attrition probability", f"{result['attrition_probability']*100:.1f}%")
    st.write("Demo risk band:", result["illustrative_risk_band"])
    st.caption(result["notice"])

st.subheader("Model evaluation (synthetic holdout)")
st.json(metrics)
st.subheader("Employee records")
view = filtered[["employee_demo_id","department","job_role","years_at_company","attrition_probability","risk_band"]]
st.dataframe(view, use_container_width=True, hide_index=True)
st.download_button("Download filtered workforce analytics CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="talentiq_workforce_analytics.csv", mime="text/csv")
