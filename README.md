# JP Morgan Quantitative Research — Virtual Experience

A suite of quantitative finance models built as part of the **JP Morgan Chase & Co. Quantitative Research Virtual Experience Program**. The project spans four tasks covering energy trading, commodity contract pricing, credit risk modelling, and mortgage analytics.

---

## Overview

| Task | Topic | Technique |
|------|-------|-----------|
| 1 | Natural Gas Price Forecasting | Seasonal decomposition, regression |
| 2 | Gas Storage Contract Pricing | Cash flow modelling, financial engineering |
| 3 | Loan Default Prediction | Logistic regression, classification |
| 4 | FICO Score Quantization | Dynamic programming, credit rating |

---

## Task 1 — Natural Gas Price Forecasting

**Problem:** Build a model that estimates the price of natural gas on any given date, including future dates.

**Key Insight:** Raw gas prices are non-stationary (trending upward over time). The model decomposes price into two components:

```
Price = Long-term trend (linear regression) + Seasonal wave (sine curve)
```

**Approach:**
- Fitted a linear trend using `numpy.polyfit`
- Removed the trend to isolate seasonal variation
- Fitted a sine + cosine model using least squares (`numpy.linalg.lstsq`) to capture the yearly seasonal cycle
- Built a `get_price(date)` function that estimates gas price on any date — past or future

**Result:** A smooth model that captures both the upward price trend and the winter/summer seasonal cycle.

**Skills:** Signal decomposition, regression from scratch, time series analysis, numpy

---

## Task 2 — Gas Storage Contract Pricing

**Problem:** Price a natural gas storage contract. A client buys cheap gas in summer, stores it, and sells at a higher price in winter. Calculate the contract's value accounting for all cash flows.

**All cash flows considered:**
```
Revenue          = withdrawal volume × gas price on withdrawal date
Injection Cost   = injection volume  × gas price on injection date
Pump Cost        = volume × injection/withdrawal cost rate
Storage Cost     = months stored     × monthly storage fee

Contract Value   = Revenue - Injection Cost - Pump Cost - Storage Cost
```

**Key engineering decisions:**
- Used `get_price()` from Task 1 to automatically estimate prices on any date
- Tank volume tracked throughout to prevent overflow or empty-tank withdrawals
- All dates sorted chronologically before processing
- Storage duration calculated precisely using actual days

**Sample Output:**
```
Injection Cost (inc. pump cost) : $166,846.74
Revenue (after pump cost)       : $183,433.75
Storage Cost (6 months)         : $30,000.00
CONTRACT VALUE                  : $-13,412.99
```

**Skills:** Financial modelling, cash flow logic, Python functions, edge case handling

---

## Task 3 — Loan Default Prediction & Expected Loss

**Problem:** The retail banking arm is experiencing higher-than-expected default rates. Build a model that predicts the probability a borrower will default, and calculate the expected loss on any loan.

**Dataset:** 10,000 borrowers with features including income, FICO score, debt levels, credit lines, and employment history. Default rate: 18.5%.

**Feature Engineering:**
```python
debt_to_income    = total_debt_outstanding / income
payment_to_income = loan_amt_outstanding   / income
```
Ratio features capture relative risk better than raw values — a $50k debt means very different things for someone earning $40k vs $200k.

**Model:** Logistic Regression with StandardScaler

**Results:**
```
Accuracy  : 100%
ROC-AUC   : 1.0
```

**Expected Loss Formula:**
```
Expected Loss = Loan Amount × PD × (1 - Recovery Rate)
```

**Sample Output:**
```
=== High Risk Borrower (FICO 550) ===
Probability of Default : 100.0%
Expected Loss          : $7,200.00

=== Low Risk Borrower (FICO 750) ===
Probability of Default : 0.0%
Expected Loss          : $0.00
```

**Skills:** Logistic regression, train/test split, feature engineering, ROC-AUC, scikit-learn

---

## Task 4 — FICO Score Quantization & Mortgage Rating

**Problem:** Charlie's ML model requires categorical inputs, but FICO scores are continuous (408–850). Find the optimal way to bucket FICO scores into ratings where each bucket tells a meaningfully different story about default risk.

**What is Quantization?**
Converting continuous numbers into discrete categories — the same way exam scores become letter grades. The challenge is finding the *smartest* boundary points, not just equal-width splits.

**Method: Dynamic Programming + Log-Likelihood**

Instead of trying every possible boundary combination (too slow), dynamic programming builds the solution incrementally:

```
Find best 1 bucket → use that to find best 2 buckets → ... → find best 5 buckets
```

The log-likelihood function scores each bucketing by how well it separates defaulters from non-defaulters:

```
LL = k × log(p) + (n-k) × log(1-p)

where n = borrowers in bucket, k = defaulters, p = k/n
```

**Optimisation:** Grouped by unique FICO scores (~400 values) instead of all 10,000 rows — ~25x speed improvement.

**Result — Optimal Boundaries:**
```
FICO 408–521  →  Rating 5  →  65.2% default rate  (extremely risky)
FICO 521–581  →  Rating 4  →  37.3% default rate  (high risk)
FICO 581–641  →  Rating 3  →  20.2% default rate  (medium risk)
FICO 641–697  →  Rating 2  →  10.4% default rate  (low risk)
FICO 697–850  →  Rating 1  →   4.7% default rate  (very safe)
```

**Expected Loss by FICO Score:**
```
FICO 780  →  Rating 1  →  PD  4.6%  →  Loss  $2,077  (on $50k loan)
FICO 620  →  Rating 3  →  PD 21.1%  →  Loss  $9,505
FICO 480  →  Rating 5  →  PD 59.7%  →  Loss $26,871
```

**Skills:** Dynamic programming, log-likelihood optimisation, quantization, credit rating systems

---

## Tech Stack

```
Language    : Python 3
Libraries   : pandas, numpy, scikit-learn, matplotlib, seaborn
Environment : Jupyter Lab
```

---

## Project Structure

```
jpmorgan-quant-simulation/
│
├── README.md
│
├── task1_gas_pricing/
│   ├── notebook.ipynb
│   └── Nat_Gas.csv
│
├── task2_contract_pricing/
│   └── notebook.ipynb
│
├── task3_loan_default/
│   ├── notebook.ipynb
│   └── loan_data.csv
│
├── task4_fico_bucketing/
│   ├── notebook.ipynb
│   └── loan_data.csv
```

---

## Key Concepts Covered

```
Quantitative Finance  →  contract pricing, expected loss, PD modelling
Statistics            →  regression, MLE, log-likelihood
Machine Learning      →  logistic regression, classification, ROC-AUC
Algorithms            →  dynamic programming, quantization
Data Science          →  feature engineering, EDA, train/test split, scaling
```

---

## About

Built as part of the **JP Morgan Chase & Co. Quantitative Research Virtual Experience Program**.

> This project was completed independently as a learning exercise in quantitative finance and machine learning.
