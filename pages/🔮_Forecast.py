from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Forecast",
    page_icon="🔮",
    layout="wide",
)

st.title("🔮 Forecast")
st.write(
    "Explore historical measurements and estimate how microplastic "
    "levels could change in the future."
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

current_year = pd.Timestamp.now().year
latest_data_year = int(df["year"].max())

st.info(
    f"The dataset contains measurements up to {latest_data_year}. "
    "Forecast values are estimates based on historical trends, "
    "not actual measurements."
)

st.warning(
    "The forecast uses a simple linear regression. Because the dataset "
    "contains only a limited number of historical years, long-term "
    "estimates are highly uncertain."
)

countries = sorted(df["country"].unique().tolist())
country_options = ["World"] + countries

selected_country = st.selectbox(
    "Select a country",
    country_options,
    index=0,
)

forecast_years = st.slider(
    "Years into the future",
    min_value=1,
    max_value=30,
    value=5,
)

target_year = current_year + forecast_years

if selected_country == "World":
    history = (
        df.groupby("year", as_index=False)["total_ug_per_kg"]
        .mean()
        .sort_values("year")
    )
else:
    history = (
        df[df["country"] == selected_country]
        .groupby("year", as_index=False)["total_ug_per_kg"]
        .mean()
        .sort_values("year")
    )

if len(history) < 3:
    st.warning("At least three historical years are needed for a forecast.")
    st.stop()

years = history["year"].to_numpy(dtype=float)
values = history["total_ug_per_kg"].to_numpy(dtype=float)

slope, intercept = np.polyfit(years, values, 1)

latest_year = int(years[-1])
latest_value = float(values[-1])

forecast_years_list = np.arange(
    max(latest_year, current_year) + 1,
    target_year + 1,
)

if len(forecast_years_list) == 0:
    forecast_years_list = np.array([target_year])

predictions = np.maximum(
    slope * forecast_years_list + intercept,
    0,
)

target_estimate = max(
    float(slope * target_year + intercept),
    0,
)

change = target_estimate - latest_value

if latest_value != 0:
    change_percent = change / abs(latest_value) * 100
else:
    change_percent = None

st.subheader(f"Forecast summary: {selected_country}")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest measured value",
    f"{latest_value:.1f} µg/kg",
    help=f"Latest available observation: {latest_year}.",
)

col2.metric(
    f"Estimate for {target_year}",
    f"{target_estimate:.1f} µg/kg",
    help="This is a model estimate, not a measured value.",
)

col3.metric(
    "Projected change",
    f"{change:+.1f} µg/kg",
    delta=(
        f"{change_percent:+.1f}%"
        if change_percent is not None
        else None
    ),
    delta_color="inverse",
)

if change > 0:
    st.error(
        f"The model estimates an increase from {latest_value:.1f} µg/kg "
        f"in {latest_year} to {target_estimate:.1f} µg/kg in {target_year}."
    )
elif change < 0:
    st.success(
        f"The model estimates a decrease from {latest_value:.1f} µg/kg "
        f"in {latest_year} to {target_estimate:.1f} µg/kg in {target_year}."
    )
else:
    st.info("The model estimates no overall change by the selected year.")

st.caption(
    "These results are mathematical extrapolations. They do not show "
    "actual current levels or guarantee future changes."
)

st.subheader("Historical data and forecast")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=years,
        y=values,
        mode="lines+markers",
        name="Historical measurements",
    )
)

fig.add_trace(
    go.Scatter(
        x=years,
        y=slope * years + intercept,
        mode="lines",
        name="Historical trend",
        line=dict(dash="dot"),
    )
)

fig.add_trace(
    go.Scatter(
        x=np.concatenate(([latest_year], forecast_years_list)),
        y=np.concatenate(([latest_value], predictions)),
        mode="lines+markers",
        name="Forecast",
        line=dict(dash="dash", color="#cb181d"),
    )
)

fig.add_vline(
    x=current_year,
    line_dash="dash",
    line_color="gray",
    annotation_text="Current year",
)

fig.update_layout(
    title=f"Microplastics trend: {selected_country}",
    xaxis_title="Year",
    yaxis_title="Total microplastics (µg/kg)",
    hovermode="x unified",
)

fig.update_xaxes(tickformat="d", separatethousands=False)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Forecast values")

forecast_table = pd.DataFrame(
    {
        "Year": forecast_years_list,
        "Estimated total (µg/kg)": np.round(predictions, 2),
    }
)

st.dataframe(forecast_table, hide_index=True, use_container_width=True)

st.subheader("🌍 Projected change around the world")

map_rows = []

for country in countries:
    country_history = (
        df[df["country"] == country]
        .groupby("year", as_index=False)["total_ug_per_kg"]
        .mean()
        .sort_values("year")
    )

    if len(country_history) < 3:
        continue

    country_years = country_history["year"].to_numpy(dtype=float)
    country_values = country_history["total_ug_per_kg"].to_numpy(dtype=float)

    country_slope, country_intercept = np.polyfit(
        country_years, country_values, 1
    )

    baseline = float(country_values[-1])
    estimate = max(
        float(country_slope * target_year + country_intercept),
        0,
    )

    if baseline == 0:
        continue

    change_pct = (estimate - baseline) / abs(baseline) * 100

    map_rows.append(
        {
            "Country": country,
            "Change (%)": change_pct,
            "Latest value (µg/kg)": baseline,
            "Last measurement year": int(country_years[-1]),
            "Estimate (µg/kg)": estimate,
        }
    )

map_df = pd.DataFrame(map_rows)

if not map_df.empty:
    max_change = max(
        abs(map_df["Change (%)"].min()),
        abs(map_df["Change (%)"].max()),
        1,
    )

    map_fig = go.Figure(
        go.Choropleth(
            locations=map_df["Country"],
            locationmode="country names",
            z=map_df["Change (%)"],
            text=map_df["Country"],
            customdata=map_df[
                [
                    "Latest value (µg/kg)",
                    "Last measurement year",
                    "Estimate (µg/kg)",
                ]
            ].to_numpy(),
            colorscale=[
                [0, "#238b45"],
                [0.5, "#f7f7f7"],
                [1, "#cb181d"],
            ],
            zmin=-max_change,
            zmax=max_change,
            zmid=0,
            colorbar_title="Change (%)",
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Projected change: %{z:.1f}%<br>"
                "Last measured value: %{customdata[0]:.1f} µg/kg<br>"
                "Last measurement year: %{customdata[1]:.0f}<br>"
                "Estimated value: %{customdata[2]:.1f} µg/kg"
                "<extra></extra>"
            ),
        )
    )

    map_fig.update_layout(
        title=f"Estimated change by {target_year}",
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
        ),
    )

    st.plotly_chart(map_fig, use_container_width=True)

    st.caption(
        "Red indicates an estimated increase; green indicates an "
        "estimated decrease. Countries are compared with their own "
        "latest available measurement, which may come from different "
        "years. Grey areas may have insufficient data or unmatched names."
    )

    st.dataframe(
        map_df.sort_values("Change (%)", ascending=False).round(2),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.warning("There is not enough country-level data to create the map.")

with st.expander("About this forecast"):
    st.write(
        "A linear regression is fitted to the available historical values. "
        "The model extends this trend into the future and limits negative "
        "estimates to zero. It does not account for changes in sampling, "
        "diet, measurement methods or other external factors."
    )
    st.write(
        "The World option uses the average of the available country "
        "observations for each year. It is not a population-weighted "
        "global estimate."
    )
