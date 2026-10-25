-- Banking & Financial Risk Analytics SQL catalog
-- DuckDB-compatible queries. Execute after registering/loading the engineered
-- dataset as a table named `loans`. Results are descriptive, not causal.

-- Query: portfolio_overview
SELECT
    COUNT(*) AS total_applications,
    SUM(CASE WHEN approved = 1 THEN 1 ELSE 0 END) AS approvals,
    AVG(CASE WHEN approved IS NOT NULL THEN approved END) AS approval_rate,
    SUM(CASE WHEN default = 1 THEN 1 ELSE 0 END) AS observed_defaults,
    COUNT(default) AS known_outcomes,
    AVG(CASE WHEN default IS NOT NULL THEN default END) AS observed_default_rate
FROM loans;

-- Query: default_rate_by_credit_score
SELECT
    credit_score_band,
    COUNT(*) AS known_outcomes,
    SUM(default) AS observed_defaults,
    AVG(default) AS observed_default_rate
FROM loans
WHERE default IN (0, 1)
GROUP BY credit_score_band
ORDER BY CASE credit_score_band
    WHEN 'Below 580' THEN 1 WHEN '580-669' THEN 2
    WHEN '670-739' THEN 3 WHEN '740+' THEN 4 ELSE 5 END;

-- Query: default_rate_by_risk_segment
SELECT
    illustrative_risk_segment,
    COUNT(*) AS known_outcomes,
    SUM(default) AS observed_defaults,
    AVG(default) AS observed_default_rate
FROM loans
WHERE default IN (0, 1)
GROUP BY illustrative_risk_segment
ORDER BY observed_default_rate DESC;

-- Query: monthly_portfolio_trend
SELECT
    DATE_TRUNC('month', application_date) AS application_month,
    COUNT(*) AS applications,
    SUM(CASE WHEN approved = 1 THEN 1 ELSE 0 END) AS approvals,
    COUNT(default) AS known_outcomes,
    SUM(CASE WHEN default = 1 THEN 1 ELSE 0 END) AS observed_defaults,
    AVG(CASE WHEN default IS NOT NULL THEN default END) AS observed_default_rate
FROM loans
WHERE application_date IS NOT NULL
GROUP BY 1
ORDER BY 1;

-- Query: loan_purpose_performance
SELECT
    loan_purpose,
    COUNT(*) AS applications,
    AVG(loan_amount) AS average_loan_amount,
    COUNT(default) AS known_outcomes,
    AVG(CASE WHEN default IS NOT NULL THEN default END) AS observed_default_rate
FROM loans
GROUP BY loan_purpose
ORDER BY applications DESC;

-- Query: risk_feature_profile
SELECT
    illustrative_risk_segment,
    COUNT(*) AS applications,
    AVG(credit_score) AS average_credit_score,
    AVG(total_debt_to_income_ratio) AS average_total_dti,
    AVG(loan_amount) AS average_loan_amount,
    AVG(CASE WHEN default IS NOT NULL THEN default END) AS observed_default_rate
FROM loans
GROUP BY illustrative_risk_segment
ORDER BY applications DESC;

-- Query: data_quality_missingness
SELECT
    COUNT(*) AS row_count,
    SUM(CASE WHEN loan_amount IS NULL THEN 1 ELSE 0 END) AS missing_loan_amount,
    SUM(CASE WHEN annual_income IS NULL THEN 1 ELSE 0 END) AS missing_annual_income,
    SUM(CASE WHEN credit_score IS NULL THEN 1 ELSE 0 END) AS missing_credit_score,
    SUM(CASE WHEN default IS NULL THEN 1 ELSE 0 END) AS missing_default_outcome
FROM loans;

-- Query: largest_loans
SELECT loan_id, customer_id, loan_amount, annual_income, credit_score,
       total_debt_to_income_ratio, illustrative_risk_segment, approved, default
FROM loans
ORDER BY loan_amount DESC NULLS LAST
LIMIT 25;

-- Query: windowed_monthly_default_trend
WITH monthly AS (
    SELECT DATE_TRUNC('month', application_date) AS application_month,
           COUNT(default) AS known_outcomes,
           AVG(CASE WHEN default IS NOT NULL THEN default END) AS observed_default_rate
    FROM loans
    WHERE application_date IS NOT NULL
    GROUP BY 1
)
SELECT application_month, known_outcomes, observed_default_rate,
       AVG(observed_default_rate) OVER (
           ORDER BY application_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
       ) AS rolling_3_month_average_default_rate
FROM monthly
ORDER BY application_month;

