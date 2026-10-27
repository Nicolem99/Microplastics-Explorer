from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Insights",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Insights")
st.write(
    "Explore historical measurements, compare countries and identify "
    "patterns in the available data."
)

data_path = Path(__file__).resolve().parent.parent / "processed_microplastics.csv"

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    st.error("The dataset could not be found.")
    st.stop()

required_columns = {"year", "country", "total_ug_per_kg"}

if not required_columns.issubset(df.columns):
    st.error("The dataset is missing required columns.")
    st.stop()

df = df[["year", "country", "total_ug_per_kg"]].copy()
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["total_ug_per_kg"] = pd.to_numeric(
    df["total_ug_per_kg"], errors="coerce"
)
df["country"] = df["country"].astype("string").str.strip()
df = df.dropna()
df = df[df["country"] != ""]

if df.empty:
    st.error("No usable data was found.")
    st.stop()

df["year"] = df["year"].astype(int)

earliest_year = int(df["year"].min())
latest_year = int(df["year"].max())

st.caption(
    f"The dataset contains observations from {earliest_year} to "
    f"{latest_year}. All charts on this page show recorded data."
)

st.subheader("Dataset overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Countries", f"{df['country'].nunique()}")
col2.metric("Recorded years", f"{df['year'].nunique()}")
col3.metric("Observations", f"{len(df):,}")
col4.metric(
    "Average recorded value",
    f"{df['total_ug_per_kg'].mean():.1f} µg/kg",
)

st.subheader("🌍 Historical trend")

yearly_data = (
    df.groupby("year", as_index=False)
    .agg(
        average_value=("total_ug_per_kg", "mean"),
        median_value=("total_ug_per_kg", "median"),
        countries=("country", "nunique"),
    )
    .sort_values("year")
)

fig_trend = px.line(
    yearly_data,
    x="year",
    y="average_value",
    markers=True,
    labels={
        "year": "Year",
        "average_value": "Average recorded value (µg/kg)",
    },
    title="Average recorded value by year",
)

fig_trend.update_xaxes(tickformat="d", separatethousands=False)

st.plotly_chart(fig_trend, use_container_width=True)

st.caption(
    "The yearly average is based on the observations available for each "
    "year. Changes in country coverage may affect the overall trend."
)

st.subheader("Compare countries")

selected_countries = st.multiselect(
    "Choose countries",
    options=sorted(df["country"].unique()),
    default=sorted(df["country"].unique())[:3],
)

if selected_countries:
    comparison = (
        df[df["country"].isin(selected_countries)]
        .groupby(["year", "country"], as_index=False)["total_ug_per_kg"]
        .mean()
    )

    fig_comparison = px.line(
        comparison,
        x="year",
        y="total_ug_per_kg",
        color="country",
        markers=True,
        labels={
            "year": "Year",
            "total_ug_per_kg": "Total microplastics (µg/kg)",
            "country": "Country",
        },
        title="Historical measurements by country",
    )

    fig_comparison.update_xaxes(
        tickformat="d",
        separatethousands=False,
    )

    st.plotly_chart(fig_comparison, use_container_width=True)
else:
    st.info("Choose at least one country to display the comparison.")

st.subheader("Countries with the highest recorded averages")

ranking = (
    df.groupby("country", as_index=False)
    .agg(
        average_value=("total_ug_per_kg", "mean"),
        observations=("total_ug_per_kg", "size"),
        last_year=("year", "max"),
    )
    .sort_values("average_value", ascending=False)
    .head(10)
)

fig_ranking = px.bar(
    ranking.sort_values("average_value"),
    x="average_value",
    y="country",
    orientation="h",
    labels={
        "average_value": "Average recorded value (µg/kg)",
        "country": "Country",
    },
    title="Top 10 countries by recorded average",
)

st.plotly_chart(fig_ranking, use_container_width=True)

st.dataframe(
    ranking.rename(
        columns={
            "country": "Country",
            "average_value": "Average value (µg/kg)",
            "observations": "Observations",
            "last_year": "Latest recorded year",
        }
    ).round(2),
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "These rankings describe the available dataset, not necessarily "
    "current conditions. Differences in sampling methods and the number "
    "of observations can affect comparisons."
)
