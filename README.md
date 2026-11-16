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

## Quick Start

- Ensure Python 3.10 or later is installed.
- Create and activate a Python virtual environment.
- Install project dependencies using `requirements.txt`.
- Generate the synthetic employee dataset.
- Train the employee attrition prediction model.
- Launch the interactive Streamlit dashboard.
- Explore employee analytics, attrition patterns, and predictive insights.

## API Usage

- Start the FastAPI backend in a separate terminal.
- Access interactive API documentation at `http://127.0.0.1:8000/docs`.
- Submit employee details to the `/predict-attrition` POST endpoint.
- Analyze workforce attributes, job satisfaction, compensation, and employment history.
- Receive predicted employee attrition risk and model-generated insights in JSON format.

**Example input parameters:**

- Employee demographics and monthly income.
- Years at company and current role.
- Job satisfaction and work-life balance.
- Overtime, travel frequency, and distance from home.
- Department, job role, and training history.

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

