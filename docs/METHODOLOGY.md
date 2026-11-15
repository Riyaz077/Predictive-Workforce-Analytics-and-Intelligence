# Methodology

1. Generate a synthetic workforce dataset with demographic-like, job-context, satisfaction, tenure, and work-pattern fields.
2. Create a binary attrition label using a simulated probability formula. The label is artificial and not based on real employee outcomes.
3. Validate required fields and encode categorical features with OneHotEncoder.
4. Train a Random Forest classifier with a stratified holdout split.
5. Report accuracy, precision, recall, F1, and ROC-AUC on the synthetic holdout set.
6. Convert predicted probabilities into illustrative Low, Medium, and High bands.
7. Aggregate synthetic attrition labels by department and job role for dashboard exploration.
