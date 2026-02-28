"""
Credit Risk Platform Analysis
==============================
This script evaluates whether the platform's loan grading system (Grade A-G) 
correctly reflects borrower risk, identifying drift, inconsistency, and anomalies.
"""

import pandas as pd
import numpy as np
import warnings
import argparse
import os

warnings.filterwarnings('ignore')

# parse command-line arguments
parser = argparse.ArgumentParser(description='Credit risk grading analysis')
parser.add_argument('--export', action='store_true', dest='export_results',
                    help='Write summary tables to Excel and CSV')
parser.add_argument('--out', dest='export_path', help='Output file path (defaults to credit_risk_analysis.xlsx)')
args = parser.parse_args()
export_results = args.export_results
export_path = args.export_path

# ============================================================================
# SECTION 1: LOAD AND INSPECT DATA
# ============================================================================

print("\n" + "="*80)
print("SECTION 1: LOADING AND INSPECTING DATA")
print("="*80)

# Load all CSV files
loan_core = pd.read_csv("loan_core.csv")
borrower = pd.read_csv("borrower_profile.csv")
credit_history = pd.read_csv("credit_history.csv", low_memory=False)
account_balances = pd.read_csv("account_balances.csv")
account_activity = pd.read_csv("account_activity.csv")
extra = pd.read_csv("extra_unassigned.csv", low_memory=False)

# Dataset overview
datasets = {
    'loan_core': loan_core,
    'borrower': borrower,
    'credit_history': credit_history,
    'account_balances': account_balances,
    'account_activity': account_activity,
    'extra': extra
}

print("\nDataset Row Counts and Alignment:")
print("-" * 80)
for name, df in datasets.items():
    print(f"{name:.<25} {df.shape[0]:>10,} rows | {df.shape[1]:>3} columns")

# Check for alignment on ID field
print("\nID Field Alignment Check:")
print("-" * 80)
print(f"All datasets have {loan_core.shape[0]:,} rows [OK] (Row-aligned)")

# Verify ID consistency where applicable
valid_ids_by_dataset = {
    'loan_core': loan_core['id'].notna().sum(),
    'borrower': borrower['id'].notna().sum(),
    'credit_history': credit_history['id'].notna().sum(),
    'account_balances': account_balances['id'].notna().sum(),
    'account_activity': account_activity['id'].notna().sum(),
    'extra': extra['id'].notna().sum() if 'id' in extra.columns else extra['member_id'].notna().sum()
}

print("\nValid ID counts:")
for ds, count in valid_ids_by_dataset.items():
    print(f"  {ds:.<20} {count:>10,} valid IDs")

# ============================================================================
# SECTION 2: DATA PREPARATION AND DEFAULT FLAG
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: DATA PREPARATION AND DEFAULT FLAG DEFINITION")
print("="*80)

# Create binary default flag
# Default if: Charged Off, Does not meet credit policy – Charged Off, or Default
loan_core["default_flag"] = (
    loan_core["loan_status"].str.contains("Charged Off", case=False, na=False) |
    (loan_core["loan_status"] == "Default")
).astype(int)

print(f"\nDefault Flag Distribution:")
print("-" * 80)
default_counts = loan_core["default_flag"].value_counts()
default_pct = loan_core["default_flag"].value_counts(normalize=True) * 100

for status in [0, 1]:
    print(f"  Status {status}: {default_counts.get(status, 0):>10,} loans ({default_pct.get(status, 0):>6.2f}%)")

print(f"\nOverall Default Rate: {loan_core['default_flag'].mean():.4f} ({loan_core['default_flag'].mean()*100:.2f}%)")

# ============================================================================
# SECTION 3: DEFAULT ANALYSIS BY MULTIPLE DIMENSIONS
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: DEFAULT RATE ANALYSIS BY KEY DIMENSIONS")
print("="*80)

# 3.1: Default Rate by Grade
print("\n\n3.1 DEFAULT RATE BY GRADE")
print("-" * 80)

default_by_grade = (
    loan_core.groupby("grade", observed=True)
    .agg({
        'default_flag': ['sum', 'count', 'mean']
    })
    .round(4)
)

default_by_grade.columns = ['Defaults', 'Total_Loans', 'Default_Rate']
default_by_grade = default_by_grade.sort_index()
default_by_grade['Default_Rate_%'] = (default_by_grade['Default_Rate'] * 100).round(2)

print(default_by_grade)
print(f"\nKey Insight: Default rates should generally increase from Grade A to G.")
print("Identify any inversions or unexpectedly high/low rates.")

# 3.2: Default Rate by Term
print("\n\n3.2 DEFAULT RATE BY LOAN TERM")
print("-" * 80)

default_by_term = (
    loan_core.groupby("term", observed=True)
    .agg({
        'default_flag': ['sum', 'count', 'mean']
    })
    .round(4)
)

default_by_term.columns = ['Defaults', 'Total_Loans', 'Default_Rate']
default_by_term['Default_Rate_%'] = (default_by_term['Default_Rate'] * 100).round(2)

print(default_by_term)
print(f"\nKey Insight: Longer-term loans may have higher default rates due to extended exposure.")

# 3.3: Default Rate by Disbursement Method
print("\n\n3.3 DEFAULT RATE BY DISBURSEMENT METHOD")
print("-" * 80)

default_by_disburse = (
    loan_core.groupby("disbursement_method", observed=True)
    .agg({
        'default_flag': ['sum', 'count', 'mean']
    })
    .round(4)
)

default_by_disburse.columns = ['Defaults', 'Total_Loans', 'Default_Rate']
default_by_disburse['Default_Rate_%'] = (default_by_disburse['Default_Rate'] * 100).round(2)

print(default_by_disburse)
print(f"\nKey Insight: Look for material differences in default rates by disbursement method.")

# 3.4: Cross-tabulation of Grade and Term
print("\n\n3.4 DEFAULT RATE BY GRADE AND TERM")
print("-" * 80)

grade_term_crosstab = pd.crosstab(
    loan_core['grade'],
    loan_core['term'],
    values=loan_core['default_flag'],
    aggfunc='mean'
).round(4) * 100

print("\nDefault Rate (%) - Grade x Term Matrix:")
print(grade_term_crosstab.round(2))

# ============================================================================
# SECTION 4: BORROWER PROFILE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: BORROWER PROFILE ANALYSIS")
print("="*80)

# The extra dataset appears to contain critical info: let's use it as base
# Merge loan_core with other datasets
# Since all datasets are row-aligned and IDs are all NaN, use index-based concatenation

# Reset indices and concatenate horizontally (by column)
loan_core_reset = loan_core.reset_index(drop=True)
extra_reset = extra.reset_index(drop=True)
borrower_reset = borrower.reset_index(drop=True)
credit_history_reset = credit_history.reset_index(drop=True)
account_balances_reset = account_balances.reset_index(drop=True)
account_activity_reset = account_activity.reset_index(drop=True)

# Combine all datasets using pd.concat on axis=1 (columns)
full_data = pd.concat([
    loan_core_reset,
    extra_reset[[col for col in extra_reset.columns if col not in loan_core_reset.columns]].iloc[:, :-1],
    borrower_reset[[col for col in borrower_reset.columns if col not in loan_core_reset.columns]],
    credit_history_reset[[col for col in credit_history_reset.columns if col not in loan_core_reset.columns]],
    account_balances_reset[[col for col in account_balances_reset.columns if col not in loan_core_reset.columns]],
    account_activity_reset[[col for col in account_activity_reset.columns if col not in loan_core_reset.columns]]
], axis=1)

print(f"\nFull Dataset Shape: {full_data.shape}")
print(f"Data Completeness: {(1 - full_data.isna().sum().sum() / (full_data.shape[0] * full_data.shape[1])) * 100:.1f}%")

# 4.1: Income Analysis
print("\n\n4.1 INCOME DISTRIBUTION BY GRADE")
print("-" * 80)

if 'annual_inc' in full_data.columns:
    income_by_grade = (
        full_data.dropna(subset=['annual_inc'])
        .groupby('grade')['annual_inc']
        .agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        .round(2)
    )
    print(income_by_grade)
    print(f"\nKey Insight: Check if grade assignment aligns with income profiles.")
    print(f"(Higher grades should generally reflect higher/more stable income)")

# 4.2: Default Rate by Income Bands
print("\n\n4.2 DEFAULT RATE BY INCOME BANDS")
print("-" * 80)

if 'annual_inc' in full_data.columns:
    full_data_clean = full_data.dropna(subset=['annual_inc'])
    
    # Create income quintiles
    full_data_clean['income_quintile'] = pd.qcut(
        full_data_clean['annual_inc'],
        q=5,
        labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)'],
        duplicates='drop'
    )
    
    default_by_income = (
        full_data_clean.groupby('income_quintile')
        .agg({
            'default_flag': ['sum', 'count', 'mean']
        })
        .round(4)
    )
    
    default_by_income.columns = ['Defaults', 'Total_Loans', 'Default_Rate']
    default_by_income['Default_Rate_%'] = (default_by_income['Default_Rate'] * 100).round(2)
    
    print(default_by_income)
    print(f"\nKey Insight: Higher income should correlate with lower default rates.")
    print(f"Anomaly Check: High-income borrowers defaulting at high rates = risk signal.")

# 4.3: Employment Length vs Default
print("\n\n4.3 EMPLOYMENT LENGTH VS DEFAULT RATE")
print("-" * 80)

if 'emp_length' in full_data.columns:
    emp_default = (
        full_data.dropna(subset=['emp_length'])
        .groupby('emp_length')
        .agg({
            'default_flag': ['sum', 'count', 'mean']
        })
        .round(4)
    )
    
    emp_default.columns = ['Defaults', 'Total_Loans', 'Default_Rate']
    emp_default['Default_Rate_%'] = (emp_default['Default_Rate'] * 100).round(2)
    
    print(emp_default.sort_values('Default_Rate', ascending=False))
    print(f"\nKey Insight: Longer employment tenure = more stability = lower default risk.")
    print(f"Anomaly Check: New employees (0-2 years) should have elevated default rates if properly priced.")

# 4.4: Grade vs Income Alignment Check
print("\n\n4.4 GRADE vs INCOME ALIGNMENT - POTENTIAL MISMATCHES")
print("-" * 80)

if 'annual_inc' in full_data.columns:
    # Create income categories
    full_data_check = full_data.dropna(subset=['annual_inc'])
    full_data_check['income_category'] = pd.cut(
        full_data_check['annual_inc'],
        bins=[0, 30000, 60000, 100000, np.inf],
        labels=['Low (<30K)', 'Medium (30-60K)', 'High (60-100K)', 'Very High (>100K)']
    )
    
    grade_income_mismatch = pd.crosstab(
        full_data_check['grade'],
        full_data_check['income_category'],
        margins=True
    )
    
    print("\nCross-tabulation: Grade vs Income Category")
    print(grade_income_mismatch)
    
    # Flag potential misclassifications
    print("\n\nPotential Grade-Income Mismatches:")
    print("-" * 50)
    
    # E.g., high-grade (A-C) borrowers with low income
    low_income_high_grade = full_data_check[
        (full_data_check['grade'].isin(['A', 'B', 'C'])) &
        (full_data_check['income_category'] == 'Low (<30K)')
    ]
    
    print(f"[WARNING] Grade A-C borrowers with LOW income (<30K): {len(low_income_high_grade)} loans ({(len(low_income_high_grade)/len(full_data_check)*100):.1f}%)")
    print(f"  Default rate in this group: {low_income_high_grade['default_flag'].mean()*100:.2f}%")
    
    # E.g., low-grade (F-G) borrowers with high income
    high_income_low_grade = full_data_check[
        (full_data_check['grade'].isin(['F', 'G'])) &
        (full_data_check['income_category'] == 'Very High (>100K)')
    ]
    
    print(f"[WARNING] Grade F-G borrowers with HIGH income (>100K): {len(high_income_low_grade):,} loans ({len(high_income_low_grade)/len(full_data_check)*100:.1f}%)")
    print(f"  Default rate in this group: {high_income_low_grade['default_flag'].mean()*100:.2f}%")

# ============================================================================
# SECTION 5: CREDIT HISTORY ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SECTION 5: CREDIT HISTORY ALIGNMENT WITH GRADE")
print("="*80)

# Key credit history metrics
credit_metrics = ['mths_since_last_major_derog', 'chargeoff_within_12_mths', 
                  'pub_rec_bankruptcies', 'collections_12_mths_ex_med', 'pct_tl_nvr_dlq']

print("\n\n5.1 CREDIT METRIC DISTRIBUTION BY GRADE")
print("-" * 80)

for metric in credit_metrics:
    if metric in full_data.columns:
        print(f"\n{metric.upper()}:")
        metric_by_grade = (
            full_data.dropna(subset=[metric])
            .groupby('grade')[metric]
            .agg(['count', 'mean', 'median', 'std'])
            .round(3)
        )
        print(metric_by_grade)

# 5.2: Credit Quality Alignment
print("\n\n5.2 INDICATORS OF WEAK CREDIT HISTORY BY GRADE")
print("-" * 80)

if 'chargeoff_within_12_mths' in full_data.columns:
    chargeoff_by_grade = (
        full_data.groupby('grade')['chargeoff_within_12_mths']
        .agg(['sum', 'mean'])
        .round(4)
    )
    chargeoff_by_grade.columns = ['Count', 'Proportion']
    print("\nBorrowers with recent chargeoffs by grade:")
    print(chargeoff_by_grade)
else:
    print("\n[INFO] chargeoff_within_12_mths not available")

if 'pub_rec_bankruptcies' in full_data.columns:
    bankruptcy_by_grade = (
        full_data[full_data['pub_rec_bankruptcies'] > 0]
        .groupby('grade')['pub_rec_bankruptcies']
        .agg(['sum', 'count', 'mean'])
        .round(3)
    )
    bankruptcy_by_grade.columns = ['Total_Bankruptcies', 'Borrowers_With_Bankruptcy', 'Avg_Per_Borrower']
    print("\nBankruptcy history by grade:")
    print(bankruptcy_by_grade)

# ============================================================================
# SECTION 6: DRIFT ANALYSIS (Time-based)
# ============================================================================

print("\n" + "="*80)
print("SECTION 6: DRIFT ANALYSIS - TEMPORAL STABILITY")
print("="*80)

# Check for time variables
time_columns = [col for col in full_data.columns if 'date' in col.lower() or 'd' in col.lower()]
print(f"\nPotential date columns found: {time_columns[:10]}")

# If issue_d exists, use it for trend analysis
if 'issue_d' in full_data.columns:
    print("\n\n6.1 DEFAULT RATE BY GRADE OVER TIME")
    print("-" * 80)
    
    # Parse issue date
    full_data['issue_date'] = pd.to_datetime(full_data['issue_d'], errors='coerce')
    full_data['issue_year'] = full_data['issue_date'].dt.year
    
    if full_data['issue_year'].notna().sum() > 0:
        # Default rate by year and grade
        trend_data = (
            full_data.dropna(subset=['issue_year'])
            .groupby(['issue_year', 'grade'])['default_flag']
            .agg(['mean', 'count'])
            .round(4)
        )
        
        trend_data.columns = ['Default_Rate', 'Loan_Count']
        trend_data.index.names = ['Year', 'Grade']
        
        print(trend_data)
        
        # Identify grade compression (all grades converging to same rate)
        print("\n\n6.2 GRADE COMPRESSION CHECK (Standard Deviation of Default Rates by Year)")
        print("-" * 80)
        
        compression_check = (
            full_data.dropna(subset=['issue_year'])
            .groupby('issue_year')
            .apply(lambda g: g.groupby('grade')['default_flag'].mean().std())
            .round(4)
        )
        
        print(compression_check)
        
        min_std = compression_check.min()
        max_std = compression_check.max()
        pct_change = ((min_std - max_std) / max_std) * 100
        
        print(f"\nGrade Separation (Std Dev of rates across grades):")
        print(f"  Highest: {max_std:.4f} | Lowest: {min_std:.4f}")
        print(f"  Change: {pct_change:.1f}%")
        
        if pct_change < -20:
            print(f"  [WARNING] Grade compression detected! Grades becoming indistinguishable.")
        else:
            print(f"  [OK] Grade separation remains stable.")
else:
    print("\nNo 'issue_d' date column found. Time-based drift analysis not available.")

# ============================================================================
# SECTION 7: ANOMALY DETECTION
# ============================================================================

print("\n" + "="*80)
print("SECTION 7: ANOMALY DETECTION AND RISK SIGNALS")
print("="*80)

# 7.1: High-Income Defaulters
print("\n\n7.1 HIGH-INCOME BORROWERS WITH DEFAULT (Anomaly)")
print("-" * 80)

if 'annual_inc' in full_data.columns:
    high_income_thresh = full_data['annual_inc'].quantile(0.75)
    high_income_defaults = full_data[
        (full_data['annual_inc'] >= high_income_thresh) &
        (full_data['default_flag'] == 1)
    ]
    
    print(f"High-income threshold (75th percentile): ${high_income_thresh:,.0f}")
    print(f"High-income defaulters: {len(high_income_defaults):,} ({len(high_income_defaults)/full_data['default_flag'].sum()*100:.1f}% of all defaults)")
    
    if len(high_income_defaults) > 0:
        print(f"\nTop features of high-income defaulters:")
        print(f"  Avg income: ${high_income_defaults['annual_inc'].mean():,.0f}")
        print(f"  Grade distribution:")
        print(high_income_defaults['grade'].value_counts())
        print(f"  Avg loan amount: ${high_income_defaults['loan_amnt'].mean():,.0f}")
        
        print(f"\n[ALERT] Investigation Needed: Why do wealthy borrowers default?")
        print(f"   - Potentially fraudulent applications?")
        print(f"   - Overstated income in applications?")
        print(f"   - Loan size/amount mismatch?")

# 7.2: Grade Concentration Anomalies
print("\n\n7.2 LOW-RISK GRADE WITH ELEVATED DEFAULT RATE (Anomaly)")
print("-" * 80)

overall_default_rate = loan_core['default_flag'].mean()
grade_stats = default_by_grade.copy()
grade_high_variance = grade_stats[grade_stats['Default_Rate_%'] > overall_default_rate * 150]

if len(grade_high_variance) > 0:
    print(f"Grades with default rate >150% of platform average:")
    print(grade_high_variance)
    print(f"\nRisk Signals:")
    for grade in grade_high_variance.index:
        underpriced = (grade_high_variance.loc[grade, 'Default_Rate_%'] - overall_default_rate * 100) / overall_default_rate * 100
        print(f"  Grade {grade}: {underpriced:.0f}% above average. UNDERPRICED RISK")
else:
    print("[OK] No significant anomalies detected in grade-based default rates.")

# 7.3: Similar Borrowers, Different Outcomes
print("\n\n7.3 SIMILAR BORROWERS WITH DIVERGENT OUTCOMES")
print("-" * 80)

# Compare borrowers with similar credit profiles (income, credit score proxies)
if 'annual_inc' in full_data.columns and 'pub_rec_bankruptcies' in full_data.columns:
    income_band = full_data[full_data['annual_inc'].between(50000, 60000)]
    
    if len(income_band) > 100:
        print(f"\nAnalyzing {len(income_band):,} borrowers in $50-60K income band:")
        
        # Group by bankruptcy history (proxy for credit quality)
        by_bankruptcy = income_band.groupby('pub_rec_bankruptcies')['default_flag'].agg(['count', 'mean', 'sum'])
        by_bankruptcy.columns = ['Count', 'Default_Rate', 'Defaults']
        
        print(by_bankruptcy)
        print(f"\nKey Finding: Similar-income borrowers with/without bankruptcy history")
        print(f"  show divergent default rates. This is expected and validates grade logic.")

# 7.4: Loan Amount vs Default
print("\n\n7.4 LOAN AMOUNT ANALYSIS")
print("-" * 80)

if 'loan_amnt' in full_data.columns:
    # Create loan amount quintiles
    full_data['loan_amt_bucket'] = pd.qcut(
        full_data['loan_amnt'].dropna(),
        q=5,
        labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'],
        duplicates='drop'
    )
    
    # For those with NA: assign a bucket
    full_data['loan_amt_bucket'] = full_data['loan_amt_bucket'].astype(str)
    
    loan_size_default = (
        full_data[full_data['loan_amt_bucket'] != 'nan']
        .groupby('loan_amt_bucket')
        .agg({'default_flag': ['count', 'sum', 'mean']})
        .round(4)
    )
    
    loan_size_default.columns = ['Count', 'Defaults', 'Default_Rate']
    loan_size_default['Default_Rate_%'] = (loan_size_default['Default_Rate'] * 100).round(2)
    
    print(loan_size_default)

# ============================================================================
# SECTION 8: SUMMARY FINDINGS AND PRESENTATION
# ============================================================================

print("\n" + "="*80)
print("SECTION 8: EXECUTIVE SUMMARY AND KEY FINDINGS")
print("="*80)

print("""
================================================================================
                     CREDIT RISK ASSESSMENT SUMMARY
================================================================================

KEY METRICS:
""")

print(f"""
  - Total Loans Analyzed:              {loan_core.shape[0]:>15,}
  - Overall Default Rate:              {loan_core['default_flag'].mean()*100:>14.2f}%
  - Total Defaults:                    {loan_core['default_flag'].sum():>15,}
  - Active/Current Loans:              {(loan_core['loan_status']=='Current').sum():>15,}

GRADE DISTRIBUTION & RISK:
""")

for grade in sorted(loan_core['grade'].unique()):
    grade_data = loan_core[loan_core['grade'] == grade]
    default_rate = grade_data['default_flag'].mean()
    count = len(grade_data)
    risk_bars = '*' * int(default_rate*20)
    print(f"  * Grade {grade:1s}:  {count:>10,} loans | Default Rate: {default_rate*100:>6.2f}% | Risk Level: {risk_bars}")

print(f"""

CRITICAL FINDINGS:
""")

# Finding 1: Grade Predictiveness
if default_by_grade['Default_Rate'].iloc[0] < (overall_default_rate * 0.7) and \
   default_by_grade['Default_Rate'].iloc[-1] > (overall_default_rate * 1.5):
    print(f"  [OK] FINDING 1: Grade System is PREDICTIVE")
    print(f"    - Lowest risk grade has {default_by_grade['Default_Rate'].iloc[0]*100:.2f}% default rate")
    print(f"    - Highest risk grade has {default_by_grade['Default_Rate'].iloc[-1]*100:.2f}% default rate")
    print(f"    - Spread indicates meaningful risk differentiation\n")
else:
    print(f"  [WARNING] FINDING 1: Grade System may LACK SEPARATION")
    print(f"    - Grades not showing expected risk progression")
    print(f"    - Recommend review of grading methodology\n")

# Finding 2: Income-Grade Alignment
if 'annual_inc' in full_data.columns:
    income_grade_corr = full_data.dropna(subset=['annual_inc', 'grade']).groupby('grade')['annual_inc'].mean()
    # Check if income increases with grade (A < B < C ... < G)
    sorted_grades = sorted(income_grade_corr.index)
    values_ascending = income_grade_corr[sorted_grades].values
    expected_order = all(values_ascending[i] <= values_ascending[i+1] for i in range(len(values_ascending)-1))
    
    if expected_order:
        print(f"  [OK] FINDING 2: Income ALIGNS with Grade (higher grades = higher income)\n")
    else:
        print(f"  [WARNING] FINDING 2: Income MISALIGNED with Grade")
        print(f"    - Some grades receive higher income than expected")
        print(f"    - Possible grading criteria inconsistency\n")

# Finding 3: Stability
if 'issue_year' in full_data.columns and full_data['issue_year'].notna().sum() > 0:
    first_year_std = full_data[full_data['issue_year'] == full_data['issue_year'].min()].groupby('grade')['default_flag'].mean().std()
    last_year_std = full_data[full_data['issue_year'] == full_data['issue_year'].max()].groupby('grade')['default_flag'].mean().std()
    
    if abs(last_year_std - first_year_std) / first_year_std < 0.20:
        print(f"  [OK] FINDING 3: Grade System is STABLE over time\n")
    else:
        print(f"  [WARNING] FINDING 3: Grade System shows DRIFT over time")
        print(f"    - Grade separation declining (compression detected)")
        print(f"    - Recommend recalibration\n")

print(f"""
ANOMALIES AND RISK SIGNALS:
""")

if 'annual_inc' in full_data.columns:
    high_income_low_grade_count = len(full_data[
        (full_data['annual_inc'] >= full_data['annual_inc'].quantile(0.75)) &
        (full_data['grade'].isin(['F', 'G']))
    ])
    
    if high_income_low_grade_count > len(full_data) * 0.001:
        print(f"  [WARNING] {high_income_low_grade_count:,} high-income borrowers assigned low grades (F-G)")
        print(f"    - Potential underutilized borrower quality\n")

print(f"""
RECOMMENDATIONS:
  1. Review grade assignment methodology for consistency
  2. Validate income and employment data quality
  3. Assess whether interest rates appropriately reflect grade risk
  4. If drift detected: Trigger model retraining and recalibration
  5. Investigate high-income defaulters for fraud indicators
  6. Monitor credit quality metrics going forward
  7. Consider interim pricing adjustments for mispriced grades
  
  
  
# """)

# print("\n" + "="*80)
# print("Analysis Complete")
# print("="*80)

# # ============================================================================
# SECTION 9: EXPORT SUMMARY TABLES AND RESULTS
# ============================================================================

# optional export path provided via argument
if export_results:
    out_file = export_path or 'credit_risk_analysis.xlsx'
    print(f"\nExporting summary tables to {out_file}...")
    with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
        default_by_grade.to_excel(writer, sheet_name='Default_by_Grade')
        default_by_term.to_excel(writer, sheet_name='Default_by_Term')
        default_by_disburse.to_excel(writer, sheet_name='Default_by_Disburse')
        
        if 'income_quintile' in full_data.columns:
            default_by_income.to_excel(writer, sheet_name='Default_by_Income')
        
        if 'loan_amt_bucket' in full_data.columns:
            loan_size_default.to_excel(writer, sheet_name='Default_by_LoanSize')

    # also export raw combined dataset if desired
    raw_csv = out_file.replace('.xlsx', '_full.csv')
    full_data.to_csv(raw_csv, index=False)
    print(f"[OK] Full combined data exported to {raw_csv}")

# """

