# 📊 Customer Churn Risk & Retention Intelligence System

<p align="center">
  An end-to-end Machine Learning and Business Intelligence system designed to predict customer churn, identify high-risk customers, analyze churn drivers, estimate revenue at risk, and generate actionable retention recommendations.
</p>

---

## 🚀 Project Overview

Customer Churn Risk & Retention Intelligence System is an end-to-end machine learning and analytics platform that helps organizations understand which customers are likely to leave, why they may leave, how much revenue is exposed, and what retention action can be considered.

The system combines:

- 📊 Exploratory Data Analysis
- 🤖 Machine Learning
- 🎯 Customer Risk Segmentation
- 🔍 Churn Driver Analysis
- 💰 Revenue-at-Risk Estimation
- 💡 Retention Recommendations
- 📈 Interactive Business Dashboard
- 📤 Exportable Customer Risk Results

The complete solution is presented through an interactive Streamlit dashboard designed for business and operational decision support.

---

## 🎯 Problem Statement

Customer churn is a major challenge for subscription-based and service-oriented businesses.

Organizations often have large amounts of customer data, but identifying customers who are at risk of leaving can be difficult.

The traditional approach is often reactive:

> Customer leaves → Business discovers the problem → Retention action is too late.

This project introduces a proactive approach:

> Customer Data → ML Prediction → Risk Segmentation → Churn Drivers → Revenue Exposure → Retention Action

The system helps transform raw customer information into actionable retention intelligence.

---

## 💡 Proposed Solution

The system processes customer information and generates a **customer-level churn probability**.

Each customer is then assigned a risk category:

| Risk Level | Churn Probability |
|------------|-------------------|
| 🟢 Low | `< 0.35` |
| 🟡 Medium | `0.35 – < 0.60` |
| 🔴 High | `≥ 0.60` |

The dashboard then provides:

- Customer churn probability
- Risk category
- Churn-related factors
- Revenue exposure
- Retention recommendations
- Customer segmentation
- Business-level churn insights

---

## 🏗️ System Architecture

<p align="center">
  <img src="assets/system_architecture.png" width="1000" alt="Customer Churn System Architecture">
</p>

### Architecture Flow

```text
                    ┌─────────────────────────┐
                    │     Customer Dataset    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Data Loading & Cleaning  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Exploratory Data        │
                    │ Analysis & Processing    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Feature Engineering     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ ML Preprocessing        │
                    │ Imputation + Encoding   │
                    │ + Scaling               │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Logistic Regression     │
                    │ Churn Prediction Model  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Churn Probability       │
                    └────────────┬────────────┘
                                 │
                ┌────────────────┼────────────────┐
                ▼                ▼                ▼
          Low Risk          Medium Risk        High Risk
                │                │                │
                └────────────────┼────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ Churn Driver Analysis   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Revenue-at-Risk         │
                    │ Estimation              │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Retention Recommendations│
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Streamlit Intelligence  │
                    │ Dashboard               │
                    └─────────────────────────┘

🧠 Machine Learning Pipeline

Raw Customer Data
       │
       ▼
Data Cleaning
       │
       ▼
Missing Value Handling
       │
       ▼
Feature Separation
       │
       ├─────────────── Numerical Features
       │                    │
       │                    ▼
       │               Median Imputation
       │                    │
       │                    ▼
       │               Standard Scaling
       │
       └─────────────── Categorical Features
                            │
                            ▼
                       Most-Frequent
                       Imputation
                            │
                            ▼
                       One-Hot Encoding
                            │
                            ▼
                    Logistic Regression
                            │
                            ▼
                    Churn Probability
                            │
                            ▼
                    Risk Classification
                            │
                            ▼
                 Retention Intelligenc

🎯 Risk Segmentation

Customers are classified into three risk groups:

🟢 Low Risk

Customers with relatively low predicted churn probability.

🟡 Medium Risk

Customers showing moderate churn probability and requiring monitoring.

🔴 High Risk

Customers with high predicted churn probability and potentially requiring priority retention attention.

📈 Dashboard

The project includes an interactive Streamlit dashboard.

Dashboard KPIs
┌────────────────┬────────────────┬────────────────┬────────────────┐
│ Total Customers│ Churned Users  │  Churn Rate    │ High-Risk Users│
├────────────────┼────────────────┼────────────────┼────────────────┤
│      7,000+    │      XXXX      │      XX%       │      XXXX      │
└────────────────┴────────────────┴────────────────┴────────────────┘

🛠️ Technology Stack

| Technology         | Purpose                   |
| ------------------ | ------------------------- |
| 🐍 Python          | Core programming          |
| 🐼 Pandas          | Data manipulation         |
| 🔢 NumPy           | Numerical computation     |
| 🤖 Scikit-learn    | Machine learning          |
| 📊 Plotly          | Interactive visualization |
| 🎨 Streamlit       | Web dashboard             |
| 💾 Joblib / Pickle | Model persistence         |
| 🌐 Git             | Version control           |
| 🐙 GitHub          | Source-code hosting       |

📂 Project Structure
Customer-Churn-Risk-Retention-Intelligence-System/
│
├── app.py
├── download_data.py
├── requirements.txt
├── run.bat
├── README.md
│
├── src/
│   └── churn_engine.py
│
├── reports/
│
├── notebooks/
│
├── assets/
│   ├── system_architecture.png
│   └── ml_pipeline.png
│
└── exports/

📊 Example Customer Risk Output
Customer ID        : CUST-1045
Tenure             : 8 Months
Contract           : Month-to-Month
Monthly Charges    : ₹1,499
Churn Probability  : 82.4%
Risk Level         : HIGH

Key Signals:
• Short customer tenure
• Month-to-month contract
• Relatively high monthly charges

Recommended Action:
Priority retention outreach and suitable long-term plan review.
