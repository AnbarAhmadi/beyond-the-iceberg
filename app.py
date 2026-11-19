# -*- coding: utf-8 -*-
"""
Titanic Passenger Insights - Streamlit dashboard
Programming for Data Science (YBSB3003) - W10 activity
"""

import pandas as pd
import plotly.express as px
import streamlit as st

# =====================================================
# 0. Page setup and colors
# =====================================================
st.set_page_config(page_title="Beyond the Iceberg", page_icon="🚢", layout="wide")

NAVY = "#1F3A5F"
TEAL = "#2A9D8F"
SAND = "#E9C46A"

# Same colors for the same categories in every chart
SURVIVAL_COLORS = {"Survived": TEAL, "Did not survive": NAVY}
CLASS_COLORS = {"1st": TEAL, "2nd": SAND, "3rd": NAVY}
SEX_COLORS = {"female": TEAL, "male": NAVY}

AGE_LABELS = ["Child", "Teen", "Young Adult", "Adult", "Older Adult"]
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"


# =====================================================
# 1. Load the data (local file first, URL as backup)
# =====================================================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("titanic.csv")
    except FileNotFoundError:
        return pd.read_csv(DATA_URL)


@st.cache_data
def prepare_data(raw):
    df = raw.copy()  # keep the raw data untouched for comparison

    # --- Missing values ---
    df["age"] = df["age"].fillna(df["age"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    df = df.drop(columns=["deck"])  # too many missing values to use

    # --- Min-Max normalization (originals are kept) ---
    for col in ["age", "fare"]:
        df[col + "_norm"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # --- Discretization: age -> age_group ---
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 12, 18, 35, 60, float("inf")],
        labels=AGE_LABELS,
        include_lowest=True,
    )

    # --- Coding sex and embarked as numbers (survived is NOT recoded) ---
    df["sex_dummy"] = df["sex"].map({"female": 0, "male": 1})
    df["embarked_dummy"] = df["embarked"].map({"C": -1, "Q": 0, "S": 1})
    coded_cols = ["sex_dummy", "embarked_dummy"]

    # Readable labels used only for charts
    df["survival_label"] = df["survived"].map({1: "Survived", 0: "Did not survive"})
    df["class_label"] = df["pclass"].map({1: "1st", 2: "2nd", 3: "3rd"})
    return df, coded_cols


raw_df = load_data()
clean_df, dummy_cols = prepare_data(raw_df)


# =====================================================
# 2. Sidebar filters
# =====================================================
st.sidebar.header("Filters")
class_choice = st.sidebar.selectbox("Passenger class", ["All", "1st", "2nd", "3rd"])
sex_choice = st.sidebar.selectbox("Sex", ["All", "female", "male"])
age_choice = st.sidebar.selectbox("Age group", ["All"] + AGE_LABELS)
st.sidebar.caption("Filters apply to every tab.")

filtered = clean_df.copy()
if class_choice != "All":
    filtered = filtered[filtered["class_label"] == class_choice]
if sex_choice != "All":
    filtered = filtered[filtered["sex"] == sex_choice]
if age_choice != "All":
    filtered = filtered[filtered["age_group"] == age_choice]


# Small helper: survival rate (%) by a grouping column
def survival_rate_by(data, col):
    out = data.groupby(col, observed=True)["survived"].mean().mul(100).round(1).reset_index()
    return out.rename(columns={"survived": "survival_rate"})


def style_chart(fig):
    fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=360, legend_title_text="")
    return fig


# =====================================================
# 3. Header
# =====================================================
st.title("🚢 Beyond the Iceberg")
st.markdown("##### Titanic Passenger Insights · Explore survival patterns across passenger groups")

if filtered.empty:
    st.warning("No passengers match these filters. Please change the sidebar selection.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Explore", "Preprocessing", "Data Explorer"])


# =====================================================
# TAB 1 - Overview
# =====================================================
with tab1:
    n = len(filtered)
    survivors = int(filtered["survived"].sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Passengers in selection", f"{n:,}")
    k2.metric("Survivors", f"{survivors:,}")
    k3.metric("Survival rate", f"{survivors / n:.1%}")
    k4.metric("Average fare", f"£{filtered['fare'].mean():.2f}")

    st.divider()
    c1, c2 = st.columns(2)

    counts = filtered["survival_label"].value_counts().reset_index()
    counts.columns = ["Outcome", "Passengers"]
    fig = px.bar(counts, x="Outcome", y="Passengers", color="Outcome",
                 color_discrete_map=SURVIVAL_COLORS, text="Passengers",
                 title="Survival counts",
                 category_orders={"Outcome": ["Survived", "Did not survive"]})
    c1.plotly_chart(style_chart(fig), use_container_width=True, key="chart_1")

    rate_class = survival_rate_by(filtered, "class_label")
    fig = px.bar(rate_class, x="class_label", y="survival_rate", color="class_label",
                 color_discrete_map=CLASS_COLORS, text="survival_rate",
                 title="Survival rate by passenger class",
                 labels={"class_label": "Passenger class", "survival_rate": "Survival rate (%)"},
                 category_orders={"class_label": ["1st", "2nd", "3rd"]})
    fig.update_layout(showlegend=False)
    c2.plotly_chart(style_chart(fig), use_container_width=True, key="chart_2")

    st.caption("Note: these rates describe the passengers in this dataset only.")


# =====================================================
# TAB 2 - Explore
# =====================================================
with tab2:
    st.subheader("Compare passenger groups")
    c1, c2 = st.columns(2)

    fig = px.bar(survival_rate_by(filtered, "class_label"), x="class_label", y="survival_rate",
                 color="class_label", color_discrete_map=CLASS_COLORS, text="survival_rate",
                 title="Survival rate by passenger class",
                 labels={"class_label": "Passenger class", "survival_rate": "Survival rate (%)"},
                 category_orders={"class_label": ["1st", "2nd", "3rd"]})
    fig.update_layout(showlegend=False)
    c1.plotly_chart(style_chart(fig), use_container_width=True, key="chart_3")

    fig = px.bar(survival_rate_by(filtered, "sex"), x="sex", y="survival_rate",
                 color="sex", color_discrete_map=SEX_COLORS, text="survival_rate",
                 title="Survival rate by sex",
                 labels={"sex": "Sex", "survival_rate": "Survival rate (%)"})
    fig.update_layout(showlegend=False)
    c2.plotly_chart(style_chart(fig), use_container_width=True, key="chart_4")

    c3, c4 = st.columns(2)
    fig = px.histogram(filtered, x="age", color="survival_label", nbins=30, barmode="overlay",
                       color_discrete_map=SURVIVAL_COLORS, opacity=0.75,
                       title="Age distribution",
                       labels={"age": "Age (years, missing values filled with median)"})
    fig.update_yaxes(title="Passengers")
    c3.plotly_chart(style_chart(fig), use_container_width=True, key="chart_5")

    fig = px.box(filtered, x="class_label", y="fare", color="class_label",
                 color_discrete_map=CLASS_COLORS, title="Fare by passenger class",
                 labels={"class_label": "Passenger class", "fare": "Fare (£)"},
                 category_orders={"class_label": ["1st", "2nd", "3rd"]})
    fig.update_layout(showlegend=False)
    c4.plotly_chart(style_chart(fig), use_container_width=True, key="chart_6")

    # --- Finding (based on the full dataset so it stays stable) ---
    by_sex = clean_df.groupby("sex")["survived"].mean()
    by_class = clean_df.groupby("pclass")["survived"].mean()
    st.info(
        f"**Finding:** In this dataset, female passengers had a much higher survival rate "
        f"({by_sex['female']:.0%}) than male passengers ({by_sex['male']:.0%}), and survival rates "
        f"were higher in 1st class ({by_class[1]:.0%}) than in 3rd class ({by_class[3]:.0%}). "
        f"This is an observed association, not proof that sex or class caused survival."
    )


# =====================================================
# TAB 3 - Preprocessing
# =====================================================
with tab3:
    st.subheader("Dataset inspection")
    i1, i2 = st.columns(2)
    i1.metric("Rows", raw_df.shape[0])
    i2.metric("Columns", raw_df.shape[1])

    with st.expander("Column names and data types"):
        st.dataframe(pd.DataFrame({"column": raw_df.columns, "dtype": raw_df.dtypes.astype(str).values}),
                     hide_index=True, use_container_width=True)
    with st.expander("First five rows"):
        st.dataframe(raw_df.head(), use_container_width=True)
    with st.expander("Selecting with .loc and .iloc"):
        st.code('raw_df.loc[0:4, ["survived", "pclass", "sex", "age", "fare", "embarked"]]')
        st.dataframe(raw_df.loc[0:4, ["survived", "pclass", "sex", "age", "fare", "embarked"]],
                     use_container_width=True)
        st.code("raw_df.iloc[10:15, 0:4]")
        st.dataframe(raw_df.iloc[10:15, 0:4], use_container_width=True)

    st.divider()
    s1, s2, s3, s4 = st.tabs(["1. Missing values", "2. Normalization", "3. Age discretization", "4. Dummy coding"])

    with s1:
        st.caption(
            f"Missing **age** values were filled with the median ({raw_df['age'].median():.1f}) and missing "
            f"**embarked** values with the most frequent port ('{raw_df['embarked'].mode()[0]}'). "
            "The **deck** column was dropped because most of its values are missing."
        )
        missing = pd.DataFrame({
            "Missing before": raw_df.isna().sum(),
            "Missing after": clean_df.reindex(columns=raw_df.columns).isna().sum().astype(str),
        })
        missing.loc["deck", "Missing after"] = "dropped"
        st.dataframe(missing, use_container_width=True)

    with s2:
        st.caption(
            "Min-Max normalization rescales a column to the 0–1 range: "
            "x_norm = (x − min) / (max − min). The original columns are kept."
        )
        st.dataframe(clean_df[["age", "age_norm", "fare", "fare_norm"]].head(10).round(3),
                     use_container_width=True)

    with s3:
        st.caption(
            "Discretization turns the continuous age into groups: 0–12 Child, 13–18 Teen, "
            "19–35 Young Adult, 36–60 Adult, 61+ Older Adult."
        )
        d1, d2 = st.columns([2, 1])
        d1.dataframe(clean_df[["age", "age_group"]].head(10), use_container_width=True)
        d2.dataframe(clean_df["age_group"].value_counts().sort_index().rename("passengers"),
                     use_container_width=True)

    with s4:
        st.caption(
            "Each category is replaced by a number in a single column. "
            "sex: female = 0, male = 1. embarked: C (Cherbourg) = -1, Q (Queenstown) = 0, S (Southampton) = 1. "
            "'survived' is already 0/1, so it is not recoded, and the coded columns are not normalized."
        )
        b1, b2 = st.columns(2)
        b1.markdown("**Before**")
        b1.dataframe(clean_df[["sex", "embarked"]].head(10), use_container_width=True)

        b2.markdown("**After**")
        b2.dataframe(clean_df[["sex_dummy", "embarked_dummy"]].head(10), use_container_width=True)


# =====================================================
# TAB 4 - Data Explorer
# =====================================================
with tab4:
    st.subheader("Data Explorer")
    view = st.radio("Select view", ["Raw", "Cleaned", "Transformed"], horizontal=True)

    base_cols = ["survived", "pclass", "sex", "age", "fare", "embarked"]
    if view == "Raw":
        source = raw_df.loc[filtered.index]  # same passengers as the filters, original values
        defaults = base_cols
    elif view == "Cleaned":
        source = filtered[[c for c in raw_df.columns if c in filtered.columns]]
        defaults = base_cols
    else:
        source = filtered.drop(columns=["survival_label", "class_label"])
        defaults = ["survived", "pclass", "age", "age_norm", "fare", "fare_norm", "age_group"] + dummy_cols

    columns = st.multiselect("Choose columns to display", list(source.columns), default=defaults)
    st.write(f"**{len(source):,} rows** match the current filters.")
    st.dataframe(source[columns] if columns else source, use_container_width=True, height=420)
