import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Card Behavior", layout="wide")


# ── Load & Clean ──────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("UCI_Credit_Card.csv")
    df.rename(columns={"default.payment.next.month": "default"}, inplace=True)
    df.drop("ID", axis=1, inplace=True)

    edu_map = {
        1: "Graduate",
        2: "University",
        3: "HighSchool",
        4: "Other",
        5: "Other",
        6: "Other",
        0: "Other",
    }
    df["EDUCATION"] = df["EDUCATION"].map(edu_map)

    mar_map = {1: "Married", 2: "Single", 3: "Other", 0: "Other"}
    df["MARRIAGE"] = df["MARRIAGE"].map(mar_map)

    df["SEX"] = df["SEX"].map({1: "Male", 2: "Female"})

    bill_cols = [
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
    ]
    pay_cols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

    df["utilization"] = df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1)
    df["payment_ratio"] = (
        df[pay_cols].sum(axis=1) / df[bill_cols].sum(axis=1).replace(0, np.nan)
    ).clip(0, 1)
    df["late_months"] = (
        df[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]] > 0
    ).sum(axis=1)
    df["bill_trend"] = df["BILL_AMT1"] - df["BILL_AMT6"]

    return df


df = load_data()

# ── Sidebar Filters ───────────────────────────────────────
st.sidebar.title("🔍 Filters")

education = st.sidebar.multiselect(
    "Education Level",
    options=df["EDUCATION"].unique(),
    default=df["EDUCATION"].unique(),
)

marriage = st.sidebar.multiselect(
    "Marital Status", options=df["MARRIAGE"].unique(), default=df["MARRIAGE"].unique()
)

sex = st.sidebar.multiselect(
    "Gender", options=df["SEX"].unique(), default=df["SEX"].unique()
)

df_filtered = df[
    (df["EDUCATION"].isin(education))
    & (df["MARRIAGE"].isin(marriage))
    & (df["SEX"].isin(sex))
]

# ── Header ────────────────────────────────────────────────
st.title("💳 Credit Card Behavior Deep Dive")
st.markdown(
    "Analyzing **30,000 credit card customers** to uncover default risk patterns."
)
st.markdown(f"Showing **{len(df_filtered):,}** customers after filters")
st.divider()

# ── KPI Row ───────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers", f"{len(df_filtered):,}")
k2.metric("Default Rate", f"{df_filtered['default'].mean()*100:.1f}%")
k3.metric("Avg Utilization", f"{df_filtered['utilization'].mean():.2f}")
k4.metric("Avg Late Months", f"{df_filtered['late_months'].mean():.2f}")
st.divider()

# ── Section 1: Demographics ───────────────────────────────
st.subheader("📊 Demographics")

col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(df_filtered["AGE"], bins=30, kde=True, ax=ax, color="mediumpurple")
    ax.set_title("Age Distribution")
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots()
    df_filtered["EDUCATION"].value_counts().plot(
        kind="bar", ax=ax, color="mediumpurple", edgecolor="white"
    )
    ax.set_title("Education Level")
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig)
    plt.close()

with col3:
    fig, ax = plt.subplots()
    edu = df_filtered.groupby("EDUCATION")["default"].mean() * 100
    edu.plot(kind="bar", ax=ax, color="salmon", edgecolor="white")
    ax.set_title("Default Rate by Education (%)")
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Section 2: Credit Limit & Utilization ─────────────────
st.subheader("💰 Credit Limit & Utilization")

df_filtered["limit_group"] = pd.cut(
    df_filtered["LIMIT_BAL"],
    bins=[0, 50000, 150000, 300000, 1000000],
    labels=["Low (0-50k)", "Mid (50-150k)", "High (150-300k)", "VHigh (300k+)"],
)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.boxplot(
        data=df_filtered,
        x="limit_group",
        y="utilization",
        ax=ax,
        palette="Purples",
        order=["Low (0-50k)", "Mid (50-150k)", "High (150-300k)", "VHigh (300k+)"],
    )
    ax.set_title("Utilization by Credit Limit Group")
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots()
    df_filtered.groupby("limit_group")["default"].mean() * 100
    dr = df_filtered.groupby("limit_group")["default"].mean() * 100
    dr.plot(kind="bar", ax=ax, color="mediumpurple", edgecolor="white")
    ax.set_title("Default Rate by Limit Group (%)")
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Section 3: Payment Behavior ───────────────────────────
st.subheader("💳 Payment Behavior")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    pt = df_filtered.pivot_table(
        values="payment_ratio", index="EDUCATION", columns="MARRIAGE", aggfunc="mean"
    )
    sns.heatmap(pt, annot=True, fmt=".2f", cmap="YlOrRd_r", ax=ax, linewidths=0.5)
    ax.set_title("Avg Payment Ratio by Education x Marriage")
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots()
    df_filtered["late_months"].value_counts().sort_index().plot(
        kind="bar", ax=ax, color="steelblue", edgecolor="white"
    )
    ax.set_title("Late Payment Months Distribution")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Section 4: Default Trajectory ────────────────────────
st.subheader("📈 Bill Trend Before Default")

bill_cols = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
]
months = ["M-6", "M-5", "M-4", "M-3", "M-2", "M-1"]

defaulters = df_filtered[df_filtered["default"] == 1]
non_defaulters = df_filtered[df_filtered["default"] == 0]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(
    months,
    defaulters[bill_cols].mean().values,
    marker="o",
    label="Default",
    color="salmon",
)
axes[0].plot(
    months,
    non_defaulters[bill_cols].mean().values,
    marker="o",
    label="Non-default",
    color="steelblue",
)
axes[0].set_title("Avg Bill Amount: Last 6 Months")
axes[0].legend()

pay_cols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
axes[1].plot(
    months,
    defaulters[pay_cols].mean().values,
    marker="o",
    label="Default",
    color="salmon",
)
axes[1].plot(
    months,
    non_defaulters[pay_cols].mean().values,
    marker="o",
    label="Non-default",
    color="steelblue",
)
axes[1].set_title("Avg Payment Amount: Last 6 Months")
axes[1].legend()

st.pyplot(fig)
plt.close()

st.divider()

# ── Section 5: Raw Data ───────────────────────────────────
st.subheader("🗂️ Raw Data")
st.dataframe(df_filtered.head(100))
