from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from talentiq.model import train, save_model

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "data" / "synthetic_employee_data.csv")
model, metrics = train(df)
save_model(model, root / "models" / "talentiq_model.joblib")
print("Synthetic-data holdout metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.3f}")
print("These metrics are not evidence of real-world HR validity.")
