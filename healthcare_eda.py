"""
Healthcare Analytics -- Doctor Visits
Streamlit Dashboard version of the EDA script
Run with:  streamlit run streamlit_app.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Healthcare Analytics -- Doctor Visits",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# AESTHETICS
# ============================================================
PALETTE   = ["#3b82d4", "#e05c5c", "#22b07d", "#f5a623", "#9b59b6", "#1abc9c"]
BG        = "#f7f8fa"
ACCENT    = "#3b82d4"
DANGER    = "#e05c5c"
SUCCESS   = "#22b07d"
MUTED     = "#57606a"

plt.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : BG,
    "axes.grid"         : True,
    "grid.color"        : "#e5e7eb",
    "grid.linewidth"    : 0.7,
    "font.family"       : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.titleweight"  : "bold",
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 11,
    "xtick.labelsize"   : 10,
    "ytick.labelsize"   : 10,
    "legend.fontsize"   : 10,
    "figure.dpi"        : 130,
})

os.makedirs("plots", exist_ok=True)

# CHANGE THIS if your CSV lives somewhere else / is named differently
CSV_PATH = "1776250375-P2-Healthcare Analytics for Doctor Visits.csv"


def save_and_show(fig, name, caption=None):
    """Save the figure to ./plots AND render it inline in the dashboard."""
    fig.savefig(f"plots/{name}.png", bbox_inches="tight", dpi=130)
    st.pyplot(fig, use_container_width=True)
    if caption:
        st.caption(caption)
    plt.close(fig)


def insight(text):
    st.info(text)


# ============================================================
# LOAD DATA (cached so the dashboard doesn't reload on every click)
# ============================================================
@st.cache_data
def load_data(path):
    return pd.read_csv(path, index_col=0)


st.title("🏥 Healthcare Analytics — Doctor Visits")
st.caption("Data Understanding, Preprocessing & Exploratory Data Analysis with Storytelling Insights")

if not os.path.exists(CSV_PATH):
    st.error(f"Could not find '{CSV_PATH}'. Update CSV_PATH near the top of this script, "
             f"or place the CSV next to streamlit_app.py.")
    st.stop()

df = load_data(CSV_PATH)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
sections = [
    "Overview & Data Dictionary",
    "Preprocessing",
    "1. Target Distribution",
    "2. Gender & Visits",
    "3. Age & Visits",
    "4. Income & Visits",
    "5. Illness Burden",
    "6. Reduced Activity Days",
    "7. Insurance Type",
    "8. Chronic Conditions",
    "9. Correlation Heatmap",
    "10. Feature Impact",
    "11. Health Score",
    "12. Age x Insurance Heatmap",
    "Key Findings Summary",
]
choice = st.sidebar.radio("Jump to section", sections)

numeric_cols     = ["visits", "age", "income", "illness", "reduced", "health"]
categorical_cols = ["gender", "private", "freepoor", "freerepat", "nchronic", "lchronic"]

data_dict = {
    "visits"   : "Number of doctor visits in the past 2 weeks (target)",
    "gender"   : "Patient gender (male / female)",
    "age"      : "Age in decades (scaled: 19 = ~19 yrs, 0.72 = ~72 yrs -- actual age)",
    "income"   : "Annual household income (scaled 0-1.5)",
    "illness"  : "Number of illnesses in past 2 weeks (0-5)",
    "reduced"  : "Days of reduced activity due to illness in past 2 weeks (0-14)",
    "health"   : "General health score (higher = more consultations needed)",
    "private"  : "Has private health insurance (yes/no)",
    "freepoor" : "Has free government insurance -- poor (yes/no)",
    "freerepat": "Has free government insurance -- repatriate (yes/no)",
    "nchronic" : "Has a non-limiting chronic condition (yes/no)",
    "lchronic" : "Has a limiting chronic condition (yes/no)",
}

# ============================================================
# PREPROCESSING (always run, needed by every section)
# ============================================================
bin_map = {"yes": 1, "no": 0}
for col in categorical_cols[1:]:
    df[col + "_enc"] = df[col].map(bin_map)
df["gender_enc"] = (df["gender"] == "female").astype(int)
df["age_yrs"] = (df["age"] * 100).round(0).astype(int)

bins_i, lbls_i = [0, 0.25, 0.50, 0.75, 1.0, 1.5], ["Very Low", "Low", "Medium", "High", "Very High"]
df["income_bin"] = pd.cut(df["income"], bins=bins_i, labels=lbls_i, include_lowest=True)

bins_a, lbls_a = [0, 25, 40, 55, 75], ["Young (-25)", "Middle (26-40)", "Senior (41-55)", "Elderly (56+)"]
df["age_group"] = pd.cut(df["age_yrs"], bins=bins_a, labels=lbls_a)

df["insurance_type"] = "None"
df.loc[df["private"] == "yes",   "insurance_type"] = "Private"
df.loc[df["freerepat"] == "yes", "insurance_type"] = "Free Repatriate"
df.loc[df["freepoor"] == "yes",  "insurance_type"] = "Free Poor"

df["any_chronic"] = ((df["nchronic"] == "yes") | (df["lchronic"] == "yes")).astype(int)
df["visited"] = (df["visits"] >= 1).astype(int)
df["reduced_grp"] = df["reduced"].clip(upper=8)

zero_pct = (df["visits"] == 0).mean() * 100

# ============================================================
# SECTION: OVERVIEW
# ============================================================
if choice == "Overview & Data Dictionary":
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isnull().sum().sum()))

    st.subheader("Data Dictionary")
    st.table(pd.DataFrame(data_dict.items(), columns=["Column", "Description"]))

    st.subheader("Numeric Summary")
    st.dataframe(df[numeric_cols].describe().round(3))

    st.subheader("Categorical Summary")
    for col in categorical_cols:
        vc = df[col].value_counts()
        pct = (vc / len(df) * 100).round(1)
        summary = pd.DataFrame({"count": vc, "pct": pct})
        st.write(f"**{col}**")
        st.dataframe(summary)

# ============================================================
# SECTION: PREPROCESSING
# ============================================================
elif choice == "Preprocessing":
    st.subheader("Preprocessing Steps Applied")
    st.markdown("""
    - Binary encoding applied to yes/no columns
    - Gender encoded (1 = female, 0 = male)
    - Age scaled back to actual years
    - Income bins created (5 levels)
    - Age group bins created (4 levels)
    - Insurance type labelled
    - `any_chronic` flag created
    - `visited` flag created (1 = at least 1 visit)
    """)
    st.write(f"**Final dataset shape:** {df.shape}")
    st.dataframe(df.head(20))

# ============================================================
# PLOT 1
# ============================================================
elif choice == "1. Target Distribution":
    st.header("Plot 1 — Doctor Visit Frequency: The Zero-Inflation Story")
    vc = df["visits"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = [DANGER if v == 0 else ACCENT for v in vc.index]
    axes[0].bar(vc.index, vc.values, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_xlabel("Number of Doctor Visits (past 2 weeks)")
    axes[0].set_ylabel("Number of Patients")
    axes[0].set_title("Visit Frequency Distribution")
    axes[0].annotate(f"{zero_pct:.1f}%\nof patients\nhad 0 visits",
                      xy=(0, vc[0]), xytext=(2.5, 3500),
                      arrowprops=dict(arrowstyle="->", color=DANGER),
                      fontsize=10, color=DANGER, fontweight="bold")
    for bar, val in zip(axes[0].patches, vc.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                      str(val), ha="center", va="bottom", fontsize=9, color=MUTED)

    pie_vals = [zero_pct, 100 - zero_pct]
    pie_lbls = [f"No Visit\n({zero_pct:.1f}%)", f">=1 Visit\n({100-zero_pct:.1f}%)"]
    wedges, texts, autotexts = axes[1].pie(
        pie_vals, labels=pie_lbls, colors=[DANGER, ACCENT],
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2))
    [t.set_fontsize(11) for t in texts + autotexts]
    axes[1].set_title("Visit vs No-Visit Split")
    plt.tight_layout()
    save_and_show(fig, "01_target_distribution")

    insight(f"{zero_pct:.1f}% of patients made zero doctor visits — a textbook zero-inflated "
            f"distribution. Any predictive model must account for the excess zeros "
            f"(Zero-Inflated Poisson or Negative Binomial).")

# ============================================================
# PLOT 2
# ============================================================
elif choice == "2. Gender & Visits":
    st.header("Plot 2 — Gender & Doctor Visit Patterns")
    gender_mean = df.groupby("gender")["visits"].mean()
    gender_counts = df.groupby(["gender", "visits"]).size().unstack(fill_value=0)
    gender_pct = gender_counts.div(gender_counts.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(gender_mean.index, gender_mean.values, color=[ACCENT, DANGER],
                edgecolor="white", linewidth=1.2, width=0.5)
    axes[0].set_ylabel("Average Doctor Visits")
    axes[0].set_title("Mean Visits by Gender")
    for i, (g, v) in enumerate(gender_mean.items()):
        axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom",
                      fontsize=12, fontweight="bold", color=MUTED)

    bottom = np.zeros(2)
    colors_v = [DANGER, "#f5a623", ACCENT, SUCCESS, "#9b59b6"]
    for idx, col in enumerate(range(min(5, gender_pct.shape[1]))):
        val = gender_pct.iloc[:, col].values
        axes[1].bar(gender_pct.index, val, bottom=bottom, label=f"{col} visits",
                    color=colors_v[idx], edgecolor="white", linewidth=0.6)
        bottom += val
    axes[1].set_ylabel("Percentage of Patients (%)")
    axes[1].set_title("Visit Distribution by Gender (stacked %)")
    axes[1].legend(loc="upper right")
    plt.tight_layout()
    save_and_show(fig, "02_gender_visits")

    insight("Female patients average 0.362 visits vs 0.236 for males — a 53% higher rate. "
            "Women tend to seek medical help more proactively, consistent with global "
            "health-seeking behaviour research.")

# ============================================================
# PLOT 3
# ============================================================
elif choice == "3. Age & Visits":
    st.header("Plot 3 — Age & Doctor Visits: Elderly Dominate Utilisation")
    age_grp_mean = df.groupby("age_group", observed=True)["visits"].mean()
    age_grp_cnt  = df["age_group"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(age_grp_cnt.index.astype(str), age_grp_cnt.values, color=ACCENT, edgecolor="white")
    axes[0].set_xlabel("Number of Patients")
    axes[0].set_title("Patient Population by Age Group")
    for i, v in enumerate(age_grp_cnt.values):
        axes[0].text(v + 20, i, str(v), va="center", fontsize=10, color=MUTED)

    colors_ag = [SUCCESS, "#f5a623", ACCENT, DANGER]
    axes[1].bar(age_grp_mean.index.astype(str), age_grp_mean.values, color=colors_ag,
                edgecolor="white", linewidth=1)
    axes[1].set_ylabel("Average Doctor Visits")
    axes[1].set_title("Mean Visits by Age Group")
    for i, v in enumerate(age_grp_mean.values):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom",
                      fontsize=11, fontweight="bold", color=MUTED)
    plt.tight_layout()
    save_and_show(fig, "03_age_visits")

    insight("The Elderly (56+) group visits the doctor at a rate of 0.39 — the highest of all "
            "age groups. Young adults form the largest cohort yet visit far less. Healthcare "
            "demand is clearly driven by the ageing population.")

# ============================================================
# PLOT 4
# ============================================================
elif choice == "4. Income & Visits":
    st.header("Plot 4 — Income & Doctor Visits: Poverty Barrier to Healthcare")
    inc_mean = df.groupby("income_bin", observed=True)["visits"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(inc_mean.index.astype(str), inc_mean.values, "o-", color=DANGER,
                 lw=2.5, ms=9, markeredgecolor="white", markeredgewidth=1.5)
    axes[0].fill_between(range(len(inc_mean)), inc_mean.values, alpha=0.15, color=DANGER)
    axes[0].set_ylabel("Average Doctor Visits")
    axes[0].set_title("Mean Visits by Income Level")
    axes[0].set_xticks(range(len(inc_mean)))
    axes[0].set_xticklabels(inc_mean.index.astype(str), rotation=15)
    for i, v in enumerate(inc_mean.values):
        axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom",
                      fontsize=10, fontweight="bold", color=DANGER)

    axes[1].hist(df["income"], bins=30, color=ACCENT, edgecolor="white", linewidth=0.5)
    axes[1].axvline(df["income"].median(), color=DANGER, lw=2, linestyle="--",
                     label=f"Median = {df['income'].median():.2f}")
    axes[1].axvline(df["income"].mean(), color=SUCCESS, lw=2, linestyle="--",
                     label=f"Mean = {df['income'].mean():.2f}")
    axes[1].set_xlabel("Income (scaled)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Income Distribution of Patients")
    axes[1].legend()
    plt.tight_layout()
    save_and_show(fig, "04_income_visits")

    insight("There is a clear inverse relationship — Very Low income patients visit 0.398 times "
            "on average vs 0.207 for High income. The Very High group rises slightly (0.225), "
            "likely because wealthier individuals can better afford healthcare.")

# ============================================================
# PLOT 5
# ============================================================
elif choice == "5. Illness Burden":
    st.header("Plot 5 — Illness Count & Visits: More Sick, More Visits")
    ill_mean = df.groupby("illness")["visits"].mean()
    ill_cnt  = df["illness"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(ill_cnt.index, ill_cnt.values, color=ACCENT, edgecolor="white")
    axes[0].set_xlabel("Number of Illnesses (past 2 weeks)")
    axes[0].set_ylabel("Patient Count")
    axes[0].set_title("Illness Count Distribution")
    for bar, v in zip(axes[0].patches, ill_cnt.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                      str(v), ha="center", fontsize=9, color=MUTED)

    axes[1].bar(ill_mean.index, ill_mean.values,
                color=[plt.cm.RdYlGn_r(i/5) for i in range(6)], edgecolor="white", linewidth=0.8)
    axes[1].set_xlabel("Number of Illnesses (past 2 weeks)")
    axes[1].set_ylabel("Average Doctor Visits")
    axes[1].set_title("Mean Visits by Illness Count")
    for i, v in enumerate(ill_mean.values):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    save_and_show(fig, "05_illness_visits")

    insight("Every additional illness drives up visit rates dramatically. Patients with 5 "
            "illnesses visit ~10x more (0.814) than those with zero (0.079). Illness count is "
            "the strongest categorical predictor after 'reduced activity days'.")

# ============================================================
# PLOT 6
# ============================================================
elif choice == "6. Reduced Activity Days":
    st.header("Plot 6 — Activity Reduction & Visits: Severity Drives Demand")
    red_mean = df.groupby("reduced_grp")["visits"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(red_mean.index, red_mean.values,
                color=[plt.cm.Reds(0.3 + 0.07 * i) for i in range(len(red_mean))], edgecolor="white")
    axes[0].set_xlabel("Days of Reduced Activity (0-8+)")
    axes[0].set_ylabel("Average Doctor Visits")
    axes[0].set_title("Mean Visits by Days of Reduced Activity")
    for i, v in enumerate(red_mean.values):
        axes[0].text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    np.random.seed(42)
    jit_x = df["reduced"] + np.random.uniform(-0.2, 0.2, len(df))
    jit_y = df["visits"]  + np.random.uniform(-0.15, 0.15, len(df))
    axes[1].scatter(jit_x, jit_y, alpha=0.08, s=12, color=ACCENT)
    m, b = np.polyfit(df["reduced"], df["visits"], 1)
    x_line = np.linspace(0, 14, 100)
    r2 = np.corrcoef(df["reduced"], df["visits"])[0, 1] ** 2
    axes[1].plot(x_line, m * x_line + b, color=DANGER, lw=2.5, label=f"Trend (r2={r2:.2f})")
    axes[1].set_xlabel("Days of Reduced Activity")
    axes[1].set_ylabel("Doctor Visits")
    axes[1].set_title("Scatter: Reduced Activity vs Visits")
    axes[1].legend()
    plt.tight_layout()
    save_and_show(fig, "06_reduced_visits")

    insight("Reduced activity days has the HIGHEST correlation with visits (r=0.419). Patients "
            "bedridden for 14 days average close to 1 full visit, vs near-zero for active "
            "patients. This is the single best proxy for healthcare urgency.")

# ============================================================
# PLOT 7
# ============================================================
elif choice == "7. Insurance Type":
    st.header("Plot 7 — Insurance Type: Free Repatriate Patients Visit Most")
    ins_mean = df.groupby("insurance_type")["visits"].mean().sort_values(ascending=False)
    ins_visit_pct = df.groupby("insurance_type")["visited"].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_ins = [DANGER, ACCENT, SUCCESS, "#f5a623"]
    bars = axes[0].bar(ins_mean.index, ins_mean.values, color=colors_ins,
                        edgecolor="white", linewidth=1, width=0.55)
    axes[0].set_ylabel("Average Doctor Visits")
    axes[0].set_title("Mean Visits by Insurance Type")
    for bar, v in zip(bars, ins_mean.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.3f}",
                      ha="center", va="bottom", fontsize=11, fontweight="bold", color=MUTED)

    axes[1].bar(ins_visit_pct.index, ins_visit_pct.values, color=colors_ins,
                edgecolor="white", width=0.55)
    axes[1].set_ylabel("% of Patients Who Visited (>=1 visit)")
    axes[1].set_title("% Who Visited at Least Once by Insurance Type")
    for i, v in enumerate(ins_visit_pct.values):
        axes[1].text(i, v + 0.3, f"{v:.1f}%", ha="center", va="bottom",
                      fontsize=11, fontweight="bold", color=MUTED)
    plt.tight_layout()
    save_and_show(fig, "07_insurance_visits")

    insight("Free Repatriate patients visit most (0.467 avg), while the Free Poor group visits "
            "least (0.158). This paradox suggests poverty is a stronger barrier to healthcare "
            "than insurance access — poverty-linked social factors suppress utilisation even "
            "when care is free.")

# ============================================================
# PLOT 8
# ============================================================
elif choice == "8. Chronic Conditions":
    st.header("Plot 8 — Chronic Conditions: Limiting Conditions Triple Visit Rates")
    chron_data = {
        "No Chronic": df[df["any_chronic"] == 0]["visits"].mean(),
        "Any Chronic": df[df["any_chronic"] == 1]["visits"].mean(),
        "Non-Limiting\nChronic Only": df[(df["nchronic"] == "yes") & (df["lchronic"] == "no")]["visits"].mean(),
        "Limiting\nChronic": df[df["lchronic"] == "yes"]["visits"].mean(),
    }
    gender_chron = df.groupby(["gender", "lchronic"])["visits"].mean().unstack()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bars = axes[0].bar(chron_data.keys(), chron_data.values(),
                        color=[SUCCESS, ACCENT, "#f5a623", DANGER], edgecolor="white", linewidth=1)
    axes[0].set_ylabel("Average Doctor Visits")
    axes[0].set_title("Mean Visits by Chronic Condition Status")
    for bar, v in zip(bars, chron_data.values()):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.3f}",
                      ha="center", va="bottom", fontsize=11, fontweight="bold", color=MUTED)

    gender_chron.plot(kind="bar", ax=axes[1], color=[SUCCESS, DANGER], edgecolor="white", linewidth=0.8)
    axes[1].set_xlabel("Gender")
    axes[1].set_ylabel("Average Doctor Visits")
    axes[1].set_title("Visits by Gender — Limiting Chronic Condition")
    axes[1].set_xticklabels(["Female", "Male"], rotation=0)
    axes[1].legend(title="Limiting Chronic", labels=["No", "Yes"])
    plt.tight_layout()
    save_and_show(fig, "08_chronic_visits")

    insight("Patients with a LIMITING chronic condition average 0.603 visits — more than 3x "
            "those with none (0.192). Women with limiting chronic conditions average 0.694 "
            "visits — the highest utilisation segment in the entire dataset.")

# ============================================================
# PLOT 9
# ============================================================
elif choice == "9. Correlation Heatmap":
    st.header("Plot 9 — Correlation Matrix: All Features vs Doctor Visits")
    enc_cols = ["visits", "age", "income", "illness", "reduced", "health",
                "gender_enc", "private_enc", "freepoor_enc", "freerepat_enc",
                "nchronic_enc", "lchronic_enc"]
    corr = df[enc_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt=".2f", linewidths=0.5,
                square=True, ax=ax, annot_kws={"size": 9}, cbar_kws={"shrink": 0.8})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    save_and_show(fig, "09_correlation_heatmap")

    top_corr = corr["visits"].abs().drop("visits").sort_values(ascending=False)
    st.write("**Top correlations with visits:**")
    rows = []
    for feat, val in top_corr.items():
        sign = "+" if corr["visits"][feat] > 0 else "-"
        rows.append({"feature": feat, "r": f"{sign}{abs(val):.3f}"})
    st.dataframe(pd.DataFrame(rows))

    insight("Reduced activity days (r=0.419) and illness count (r=0.224) are the dominant "
            "predictors. Income is negatively correlated (r=-0.077): wealthier patients visit "
            "less overall.")

# ============================================================
# PLOT 10
# ============================================================
elif choice == "10. Feature Impact":
    st.header("Plot 10 — Feature Impact: Marginal Effect on Visit Rate")
    feature_impact = {
        "Limiting Chronic": df.groupby("lchronic")["visits"].mean()["yes"] - df.groupby("lchronic")["visits"].mean()["no"],
        "5 Illnesses vs 0": df[df["illness"] == 5]["visits"].mean() - df[df["illness"] == 0]["visits"].mean(),
        "Female vs Male": df[df["gender"] == "female"]["visits"].mean() - df[df["gender"] == "male"]["visits"].mean(),
        "Free Repat vs None": df[df["freerepat"] == "yes"]["visits"].mean() - df[df["freerepat"] == "no"]["visits"].mean(),
        "Non-Limiting Chr.": df[df["nchronic"] == "yes"]["visits"].mean() - df[df["nchronic"] == "no"]["visits"].mean(),
        "Very Low vs High\nIncome": df[df["income_bin"] == "Very Low"]["visits"].mean() - df[df["income_bin"] == "High"]["visits"].mean(),
        "Private Insurance": df[df["private"] == "yes"]["visits"].mean() - df[df["private"] == "no"]["visits"].mean(),
        "Free Poor vs None": df[df["freepoor"] == "yes"]["visits"].mean() - df[df["freepoor"] == "no"]["visits"].mean(),
    }
    impact_series = pd.Series(feature_impact).sort_values()
    colors_imp = [DANGER if v > 0 else SUCCESS for v in impact_series.values]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(impact_series.index, impact_series.values, color=colors_imp, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color=MUTED, lw=1.2, linestyle="--")
    ax.set_xlabel("<- Average Doctor Visits (group A minus group B) ->")
    for i, v in enumerate(impact_series.values):
        label = f"+{v:.3f}" if v > 0 else f"{v:.3f}"
        ax.text(v + 0.003 if v > 0 else v - 0.003, i, label,
                va="center", ha="left" if v > 0 else "right", fontsize=10, fontweight="bold", color=MUTED)
    plt.tight_layout()
    save_and_show(fig, "10_feature_impact")

    insight("Limiting chronic condition has the single largest marginal impact (+0.341) on "
            "visits. Free Poor insurance REDUCES visits vs no insurance (-0.150), confirming "
            "that despite financial support, socioeconomic barriers override insurance benefits.")

# ============================================================
# PLOT 11
# ============================================================
elif choice == "11. Health Score":
    st.header("Plot 11 — General Health Score: Sicker Patients Drive Demand")
    health_mean = df.groupby("health")["visits"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(df["health"], bins=13, color=ACCENT, edgecolor="white", linewidth=0.5, range=(-0.5, 12.5))
    axes[0].set_xlabel("Health Score (0=excellent, higher=worse)")
    axes[0].set_ylabel("Number of Patients")
    axes[0].set_title("Distribution of General Health Score")
    axes[0].axvline(df["health"].mean(), color=DANGER, lw=2, linestyle="--",
                     label=f"Mean = {df['health'].mean():.2f}")
    axes[0].legend()

    axes[1].plot(health_mean.index, health_mean.values, "o-", color=DANGER, lw=2.5, ms=8,
                 markeredgecolor="white", markeredgewidth=1.5)
    axes[1].fill_between(health_mean.index, health_mean.values, alpha=0.15, color=DANGER)
    axes[1].set_xlabel("Health Score")
    axes[1].set_ylabel("Average Doctor Visits")
    axes[1].set_title("Mean Visits by Health Score")
    plt.tight_layout()
    save_and_show(fig, "11_health_visits")

    insight("58.3% of patients have a health score of 0 (excellent health) yet form the pool of "
            "zero-visitors. As the health score climbs, visit rates rise steadily, confirming "
            "health score as a valid proxy for care demand intensity.")

# ============================================================
# PLOT 12
# ============================================================
elif choice == "12. Age x Insurance Heatmap":
    st.header("Plot 12 — Avg Doctor Visits: Age Group x Insurance Type")
    heat_data = df.groupby(["age_group", "insurance_type"], observed=True)["visits"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.5, ax=ax,
                annot_kws={"size": 10, "weight": "bold"})
    ax.set_xlabel("Insurance Type")
    ax.set_ylabel("Age Group")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    plt.tight_layout()
    save_and_show(fig, "12_age_insurance_heatmap")

    insight("Elderly Free Repatriate patients have the highest combined visit rate. Young "
            "uninsured patients have near-zero visit rates — the most underserved segment. "
            "Private insurance shows highest impact in the elderly, confirming wealth-based "
            "access disparity.")

# ============================================================
# SUMMARY
# ============================================================
elif choice == "Key Findings Summary":
    st.header("EDA Complete — Key Findings Summary")
    st.markdown("""
    1. **Zero inflation:** 79.8% of patients had zero doctor visits. Models must use
       Zero-Inflated Poisson or Negative Binomial.
    2. **Strongest predictor:** Reduced activity days (r=0.419) and illness count (r=0.224)
       drive visits most.
    3. **Chronic conditions:** Limiting chronic → 3x visit rate. Non-limiting still raises
       visits by 27%.
    4. **Gender gap:** Women visit 53% more than men on average, amplified by limiting
       chronic conditions.
    5. **Income paradox:** Very poor visit 0.398 vs rich 0.207, but the Free Poor insurance
       group visits LEAST (0.158) — poverty suppresses access despite financial support.
    6. **Free Repatriate:** Visits 81% more than uninsured — well-designed government
       programs work for this group.
    7. **Insurance type:** Private insurance has marginal impact; government programs show
       larger utilisation differences.
    8. **Age effect:** Elderly visit most; young are least likely to seek care, creating a
       prevention gap.
    """)
    st.success("All 12 plots have also been saved to ./plots/ as PNG files.")