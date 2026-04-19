# Credit Card Behavior Deep Dive

Uncovering spending psychology hidden inside transaction patterns.

## Dataset
- 30,000 credit card customers (Taiwan, 2005)
- Source: [UCI Credit Card Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- 25 columns — demographics + 6 months of billing/payment history
- Target: whether the customer defaulted next month (~22% default rate)

## Project Progress
- [x] Data Loading & Inspection
- [x] Data Cleaning (dirty categorical values, column renaming)
- [ ] Feature Engineering
- [ ] Exploratory Data Analysis
- [ ] Deep Insights & Anomaly Detection
- [ ] Final Dashboard

## Tech Stack
Python | pandas | numpy | matplotlib | seaborn | scipy

## Key Question
What behavioral patterns predict credit card default risk?

## How to Run
1. Download `UCI_Credit_Card.csv` from Kaggle
2. Place it in the project root folder
3. Open `code.ipynb` and run all cells