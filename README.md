📌 Objective

The objective of this analysis is to evaluate whether the platform’s loan grading system (Grades A–G) accurately reflects borrower risk, remains predictive over time, and demonstrates internal consistency across borrower and credit characteristics.

We aim to determine:
	•	Is the grading system predictive?
	•	Is risk properly differentiated across grades?
	•	Is the system stable over time?
	•	Are there signs of drift or misalignment?

⸻

📂 Dataset Overview

The dataset contains:
	•	2,260,668 loan records
	•	143 total features
	•	Multiple aligned tables including:
	•	Loan characteristics
	•	Borrower profile
	•	Credit history
	•	Account-level attributes
	•	Temporal variables

Each row represents a single issued loan.

⸻

⚙️ Default Definition

A loan is classified as default if the loan status is:
	•	Charged Off
	•	Default
	•	Does not meet credit policy – Charged Off

Total Defaults: 262,447
Overall Default Rate: 11.61%

⸻

🔍 Section 1 — Core Risk Validation

📊 Default Rate by Grade

Default rates increase monotonically from Grade A to Grade G:
	•	Grade A: ~3%
	•	Grade G: ~37%

This confirms strong risk differentiation across grades.

📊 Default Rate by Loan Term

60-month loans exhibit higher default rates than 36-month loans, indicating increased exposure risk over longer durations.

📊 Grade × Term Interaction

A heatmap analysis confirms that lower grades consistently exhibit higher default rates across both loan terms.

⸻

👤 Section 2 — Borrower Alignment

💰 Income Analysis
	•	Higher grades correspond to higher average income.
	•	Lower income quintiles show significantly higher default rates.

This indicates alignment between borrower strength and assigned grade.

🧑‍💼 Employment Stability

Longer employment tenure is generally associated with lower default probability, reinforcing borrower quality segmentation.

⸻

🏦 Section 3 — Credit History Alignment

Credit risk indicators (e.g., delinquencies, public records, charge-offs) increase consistently from Grade A to Grade G.

This confirms that grading reflects historical credit behavior.

⸻

📉 Section 4 — Drift Analysis (Temporal Stability)

Default rates were analyzed across issue years.

Findings:
	•	Grade separation (standard deviation of default rates across grades) declined significantly over time.
	•	This suggests grade compression and reduced discriminatory power.

While the grading system remains predictive, its effectiveness has weakened over time.

⸻

🚨 Section 5 — Anomaly & Misalignment Detection

Identified patterns include:
	•	High-income borrowers defaulting unexpectedly.
	•	Overlapping risk between adjacent grades.
	•	Signs of potential grade compression in later years.

These segments warrant further investigation.

⸻

📈 Key Conclusions
	1.	The grading system remains strongly predictive.
	2.	Risk increases consistently from Grade A to G.
	3.	Borrower and credit characteristics broadly align with assigned grades.
	4.	However, temporal drift suggests reduced discriminatory power.
	5.	Model recalibration or retraining may be recommended.

⸻

🛠 Tools & Methodology
	•	Python (Pandas, NumPy)
	•	Plotly for visualization
	•	Statistical aggregation & segmentation
	•	Quantitative drift analysis

⸻

📌 Final Assessment

The grading system is fundamentally sound but shows signs of structural compression over time. Continued monitoring and potential recalibration are recommended to maintain predictive strength and pricing efficiency.
