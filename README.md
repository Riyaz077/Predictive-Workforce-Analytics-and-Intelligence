# TalentIQ
### Employee Attrition Prediction and Workforce Intelligence System

TalentIQ is an educational HR analytics prototype that uses machine learning to estimate employee attrition risk from structured workforce data. It combines employee-level risk scoring, workforce trend analysis, model interpretation, and an interactive dashboard to demonstrate how predictive analytics can support proactive workforce planning.

> **Responsible-use notice:** The included dataset is fully synthetic. This project is a portfolio demonstration, not a validated HR assessment tool. Do not use its predictions to make hiring, firing, promotion, compensation, performance-management, or other employment decisions about real people. Attrition predictions can encode bias and must never replace employee conversations, fair HR processes, or human judgment.

## Features
- Synthetic employee dataset generator
- Data validation and preprocessing
- Random Forest attrition classifier
- Employee-level demo risk scoring and probability estimates
- Risk bands: Low, Medium, High (illustrative thresholds)
- Workforce analytics by department, role, tenure, and overtime
- Feature importance visualization
- Streamlit dashboard with filters and CSV export
- FastAPI scoring endpoints
- Unit tests and methodology documentation

## Quick start

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python scripts/generate_data.py
python scripts/train_model.py
streamlit run app/streamlit_app.py
```

## Run API
In another terminal:
```bash
uvicorn api.main:app --reload
```
Open `http://127.0.0.1:8000/docs`.

Example request:
```bash
curl -X POST "http://127.0.0.1:8000/predict-attrition" \
  -H "Content-Type: application/json" \
  -d '{"age":32,"monthly_income":6500,"years_at_company":3,"years_in_current_role":2,"years_since_promotion":2,"job_satisfaction":2,"work_life_balance":2,"overtime":1,"distance_from_home":12,"training_times_last_year":2,"department":"Research & Development","job_role":"Research Scientist","business_travel":"Travel_Rarely"}'
```

## Project structure
```text
TalentIQ/
├── api/                 # FastAPI service
├── app/                 # Streamlit dashboard
├── data/                # Synthetic employee dataset
├── docs/                # Methodology, responsible-use notes, API
├── notebooks/           # Exploratory analysis notebook
├── scripts/             # Data generation and model training
├── talentiq/            # Reusable ML and analytics modules
├── tests/               # Unit tests
├── requirements.txt
└── README.md
```

## Model notes
- The target `attrition` is synthetically generated using a transparent simulated formula.
- The model is trained on synthetic records; metrics do not indicate real-world HR accuracy.
- Risk bands are illustrative and should not be interpreted as validated probabilities or employee labels.
- Feature importance is a global model summary, not causal explanation.

## Responsible use
- Use only synthetic or properly authorized, privacy-protected workforce data.
- Do not use predictions for adverse employment decisions or individual employee surveillance.
- Avoid sensitive attributes and proxies; assess disparate impact and data quality before any research use.
- Treat outputs as aggregate planning signals only, and involve employees and HR professionals in interpretation.
- Any real-world deployment would require legal, privacy, security, fairness, and organizational review.

## Author

**Mohamed Riyaz S.**  
Integrated M.Tech, Computer Science and Engineering (Business Analytics)  
VIT Chennai

