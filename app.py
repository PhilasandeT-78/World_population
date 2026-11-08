"""
DAS580S - Streamlit PROTOTYPE of the Power BI dashboard (layout + number check).
Run:  pip install streamlit plotly pandas statsmodels scipy openpyxl
      streamlit run app.py
Reads the 'Cleaned_data' sheet of python_reference_outputs.xlsx (same folder).
Swap in your own Excel-cleaned workbook later: keep the column names below.
"""
from pathlib import Path
import numpy as np, pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
from scipy import stats
import streamlit as st

st.set_page_config(page_title="World Population – DAS580S", layout="wide")
_here = Path(__file__).parent
FILE = next((p for p in [_here / "python_reference_outputs.xlsx",
                         _here / "world_population.xlsx_outputs.xlsx"] if p.exists()),
            _here / "python_reference_outputs.xlsx")


@st.cache_data
def load():
    d = pd.read_excel(FILE, sheet_name="Cleaned_data")
    # same calculated columns you will create in Power BI (see DAX file)
    d["FertBand"] = pd.cut(d.FertilityRate, [0, 1.5, 2.1, 3.5, 99],
                           labels=["1: <=1.5", "2: 1.5-2.1", "3: 2.1-3.5", "4: >3.5"]).astype(str)
    d["HighFert"] = np.where(d.FertilityRate > 2.1, "High", "Low")
    return d


df = load()

# --- sanity guard: stale/bad data file? (World population 2023 is ~8.1 bn)
_tot = df.loc[df.Year == 2023, "Population"].sum()
if _tot > 9e9:
    st.error(f"Data check failed: countries sum to {_tot/1e9:.1f} bn in 2023 (should be ~8.1 bn). "
             "This is an OLD python_reference_outputs.xlsx that still contains UN regional aggregates. "
             "Re-download the latest workbook and restart.")
    st.stop()

# ------------------------------------------------------------------ slicers
st.sidebar.header("Slicers")
yr = st.sidebar.slider("Year", 1950, 2023, (1950, 2023))
decades = st.sidebar.multiselect("Decade", sorted(df.Decade.unique()), default=sorted(df.Decade.unique()))
bands = st.sidebar.multiselect("Fertility band", sorted(df.FertBand.unique()), default=sorted(df.FertBand.unique()))
countries = st.sidebar.multiselect("Country (blank = all)", sorted(df.Entity.unique()))
drop_out = st.sidebar.checkbox("Exclude flagged growth outliers", value=False)
st.sidebar.caption("Slicers drive pages 1-2. Page 3 tests use fixed years (as in the report), like "
                   "turning off slicer interactions in Power BI.")

f = df[df.Year.between(*yr) & df.Decade.isin(decades) & df.FertBand.isin(bands)]
if countries:
    f = f[f.Entity.isin(countries)]
if drop_out:
    f = f[~f.GrowthOutlier]
if f.empty:
    st.warning("No rows match the slicers.")
    st.stop()

st.title("World population: fertility and growth, 1950–2023")
tab1, tab2, tab3 = st.tabs(["1 · Overview", "2 · Relationships", "3 · Hypothesis tests"])

# ================================================================== PAGE 1
with tab1:
    st.subheader("Fertility has fallen sharply and growth has slowed")
    last = f.Year.max()
    c = st.columns(5)
    c[0].metric("Countries", f"{f.Entity.nunique():,}")
    c[1].metric("Avg fertility rate", f"{f.FertilityRate.mean():.2f}")
    c[2].metric("Median fertility", f"{f.FertilityRate.median():.2f}")
    c[3].metric("Avg growth rate (%)", f"{f.GrowthRate.mean():.2f}")
    c[4].metric(f"Population {last}", f"{f.loc[f.Year == last, 'Population'].sum() / 1e9:.2f} bn")

    L, R = st.columns(2)
    by_year = f.groupby("Year").agg(Fertility=("FertilityRate", "mean"), Growth=("GrowthRate", "mean")).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=by_year.Year, y=by_year.Fertility, name="Avg fertility"), secondary_y=False)
    fig.add_trace(go.Scatter(x=by_year.Year, y=by_year.Growth, name="Avg growth %"), secondary_y=True)
    fig.update_layout(title="Fertility and growth over time", height=360, legend=dict(orientation="h"))
    L.plotly_chart(fig, width="stretch")

    by_dec = f.groupby("Decade").FertilityRate.mean().reset_index()
    R.plotly_chart(px.bar(by_dec, x="Decade", y="FertilityRate", title="Average fertility by decade", height=360),
                   width="stretch")

    L, R = st.columns(2)
    L.plotly_chart(px.histogram(f, x="GrowthRate", nbins=60, range_x=[-10, 15],
                                title="Distribution of growth rate (axis clipped to -10..15)", height=360),
                   width="stretch")
    m = f[f.Year == last]
    loc_kw = dict(locations="Code") if "Code" in m.columns else dict(locations="Entity", locationmode="country names")
    R.plotly_chart(px.choropleth(m, color="FertilityRate", hover_name="Entity", color_continuous_scale="Viridis",
                                 title=f"Fertility rate by country, {last}", height=360, **loc_kw), width="stretch")

    L, R = st.columns(2)
    L.markdown(f"**Top 10 growth, {last}**")
    L.dataframe(m.nlargest(10, "GrowthRate")[["Entity", "GrowthRate", "FertilityRate"]].round(2), hide_index=True)
    R.markdown("**Summary by decade**")
    R.dataframe(f.groupby("Decade").agg(Mean_fert=("FertilityRate", "mean"), Median_fert=("FertilityRate", "median"),
                                        SD_fert=("FertilityRate", "std"), Mean_growth=("GrowthRate", "mean"),
                                        SD_growth=("GrowthRate", "std")).round(3))

# ================================================================== PAGE 2
with tab2:
    st.subheader("Higher fertility goes with faster growth, but explains only part of the story")
    n = len(f)
    r = f.FertilityRate.corr(f.GrowthRate)
    slope, intercept = np.polyfit(f.FertilityRate, f.GrowthRate, 1)
    c = st.columns(4)
    c[0].metric("Pearson r", f"{r:.3f}")
    c[1].metric("R² (simple)", f"{r**2:.3f}")
    c[2].metric("Slope", f"{slope:.3f}")
    c[3].metric("Observations", f"{n:,}")

    L, R = st.columns([3, 2])
    L.plotly_chart(px.scatter(f, x="FertilityRate", y="GrowthRate", color="Decade", opacity=0.35,
                              trendline="ols", trendline_scope="overall", trendline_color_override="black",
                              range_y=[-10, 15], title="Fertility vs growth (y clipped to -10..15)", height=420),
                   width="stretch")
    corr = f[["FertilityRate", "GrowthRate", "LogPop", "Year"]].corr().round(2)
    R.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                             title="Correlation matrix", height=420), width="stretch")

    st.markdown("**Multiple regression (GrowthRate ~ FertilityRate + LogPop + Year), fitted on the slicer selection**")
    m2 = smf.ols("GrowthRate ~ FertilityRate + LogPop + Year", f).fit()
    coef = pd.DataFrame({"Coefficient": m2.params, "Std error": m2.bse, "t": m2.tvalues, "p-value": m2.pvalues})
    L, R = st.columns([3, 2])
    L.dataframe(coef.round(5))
    R.metric("R² / adj R²", f"{m2.rsquared:.3f} / {m2.rsquared_adj:.3f}")
    R.caption("With all slicers cleared this should equal the notebook's 'Full (OLS)' column "
              "and Excel's ToolPak output (R² 0.185).")
    samp = pd.DataFrame({"Fitted": m2.fittedvalues, "Residual": m2.resid}).sample(min(4000, n), random_state=1)
    st.plotly_chart(px.scatter(samp, x="Fitted", y="Residual", opacity=0.3, range_y=[-10, 15],
                               title="Residuals vs fitted (heteroscedasticity / outliers check)", height=360),
                    width="stretch")

# ================================================================== PAGE 3
with tab3:
    st.subheader("The gap between high- and low-fertility countries is large, not just detectable")
    st.caption("Fixed samples (one row per country) so observations are independent; unaffected by slicers.")
    alpha = 0.05
    y23 = df[df.Year == 2023]
    a, b = y23[y23.HighFert == "High"].GrowthRate, y23[y23.HighFert == "Low"].GrowthRate
    t1 = stats.ttest_ind(a, b, equal_var=False)
    d1 = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)

    w = df[df.Year.isin([1990, 2023])].pivot(index="Entity", columns="Year", values="FertilityRate").dropna()
    t2 = stats.ttest_rel(w[1990], w[2023])
    dz = (w[1990] - w[2023]).mean() / (w[1990] - w[2023]).std()

    groups = [g.GrowthRate.values for _, g in y23.groupby("FertBand")]
    an = stats.f_oneway(*groups)
    sst = ((y23.GrowthRate - y23.GrowthRate.mean()) ** 2).sum()
    ssb = sum(len(g) * (g.mean() - y23.GrowthRate.mean()) ** 2 for g in groups)

    dec = lambda p: "Reject H₀" if p < alpha else "Fail to reject H₀"
    T1, T2, T3 = st.columns(3)
    with T1:
        st.markdown("**Test 1 · Welch t: 2023 growth, high vs low fertility**")
        st.caption("H₀: equal mean growth · H₁: means differ · α = 0.05")
        st.plotly_chart(px.bar(pd.DataFrame({"Group": ["High (>2.1)", "Low (≤2.1)"], "Mean growth %": [a.mean(), b.mean()]}),
                               x="Group", y="Mean growth %", height=300), width="stretch")
        st.metric("t", f"{t1.statistic:.2f}"); st.metric("p-value", f"{t1.pvalue:.2e}")
        st.metric("Cohen's d", f"{d1:.2f}"); st.success(dec(t1.pvalue))
    with T2:
        st.markdown("**Test 2 · Paired t: fertility 1990 vs 2023**")
        st.caption("H₀: mean difference = 0 · H₁: ≠ 0 · α = 0.05")
        st.plotly_chart(px.bar(pd.DataFrame({"Year": ["1990", "2023"], "Mean fertility": [w[1990].mean(), w[2023].mean()]}),
                               x="Year", y="Mean fertility", height=300), width="stretch")
        st.metric("t", f"{t2.statistic:.2f}"); st.metric("p-value", f"{t2.pvalue:.2e}")
        st.metric("Cohen's dz", f"{dz:.2f}"); st.success(dec(t2.pvalue))
    with T3:
        st.markdown("**Test 3 · ANOVA: 2023 growth across fertility bands**")
        st.caption("H₀: all band means equal · H₁: at least one differs · α = 0.05")
        st.plotly_chart(px.box(y23.sort_values("FertBand"), x="FertBand", y="GrowthRate", height=300),
                        width="stretch")
        st.metric("F", f"{an.statistic:.2f}"); st.metric("p-value", f"{an.pvalue:.2e}")
        st.metric("η²", f"{ssb / sst:.3f}"); st.success(dec(an.pvalue))

    st.markdown("### Recommendations (write these in your own words)")
    st.info("Example structure: 1) Track fertility as an early-warning indicator of growth. "
            "2) Do not forecast one country's demand from this model alone (R² ≈ 0.19). "
            "3) Treat war/migration shocks separately from the trend.")
