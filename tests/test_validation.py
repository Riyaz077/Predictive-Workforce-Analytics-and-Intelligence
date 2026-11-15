import pandas as pd
import pytest
from talentiq.validation import validate_frame, FEATURES

def test_validation_rejects_missing_columns():
    with pytest.raises(ValueError):
        validate_frame(pd.DataFrame({"age":[30]}))

def test_validation_accepts_valid_frame():
    row = {}
    for col in FEATURES:
        row[col] = ["Research & Development" if col in {"department"} else
                    "Research Scientist" if col == "job_role" else
                    "Travel_Rarely" if col == "business_travel" else 1]
    row["attrition"] = [0]
    assert len(validate_frame(pd.DataFrame(row))) == 1
