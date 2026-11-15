# API

Start:
```bash
uvicorn api.main:app --reload
```

- `GET /health` — service status and synthetic demo mode.
- `GET /model-info` — model features and illustrative synthetic holdout metrics.
- `POST /predict-attrition` — estimates a synthetic attrition probability for the supplied fictional employee profile.

Interactive docs: `/docs`.

Do not send identifiable or real employee data to this demo.
