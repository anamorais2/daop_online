import numpy as np
import pandas as pd
from scipy import stats


ALPHA_95 = 0.05
ALPHA_99 = 0.01

DATASETS = ["PneumoniaMnist"]# , "DermaMnist", "PneumoniaMnist", "OrganCMnist"]
METRICS = ["AUC", "ACC"]

REFERENCES = [
    "Baseline (ResNet-18)",
    "Auto-sklearn",
    "AutoKeras",
    "Google AutoML",
]

DAOP_VARIANTS = [
    "DAOP Online Mean (ResNet-18)"
]


DATA = {
    "PneumoniaMnist": {
        "DAOP Online Mean (ResNet-18)": {
        "AUC": [0.949, 0.963, 0.9658, 0.9697, 0.949, 0.966, 0.954, 0.969, 0.927, 0.951],
        "ACC": [0.885, 0.875, 0.899, 0.883, 0.888, 0.877, 0.816, 0.904, 0.854, 0.867],
    },
    "Baseline (ResNet-18)": {"AUC": 0.944, "ACC": 0.854},
    "Auto-sklearn":         {"AUC": 0.942, "ACC": 0.855},
    "AutoKeras":            {"AUC": 0.947, "ACC": 0.878},
    "Google AutoML":        {"AUC": 0.991, "ACC": 0.946},
    }
}


def holm_adjust(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m)
    prev = 0.0
    for k, i in enumerate(order):
        val = (m - k) * pvals[i]
        val = max(val, prev)
        val = min(val, 1.0)
        adj[i] = val
        prev = val
    return adj

rows = []

for ds in DATASETS:
    for metric in METRICS:
        for daop in DAOP_VARIANTS:
            if daop not in DATA[ds]:
                continue

            runs = np.array(DATA[ds][daop][metric], dtype=float)
            mean = runs.mean()
            sd = runs.std(ddof=1)
            n = len(runs)

            # Normality test (informative)
            shapiro_p = stats.shapiro(runs).pvalue if n >= 3 else None

            for ref in REFERENCES:
                mu0 = DATA[ds][ref][metric]

                # t-test (one-sample, one-sided)
                t_stat, p_two = stats.ttest_1samp(runs, mu0)
                p_t = p_two / 2 if t_stat > 0 else 1.0

                # Wilcoxon (one-sided)
                diffs = runs - mu0
                if np.any(np.abs(diffs) > 0):
                    p_w = stats.wilcoxon(diffs, alternative="greater").pvalue
                else:
                    p_w = 1.0

                # Effect size
                d = (mean - mu0) / sd if sd > 0 else np.inf

                rows.append({
                    "Dataset": ds,
                    "Metric": metric,
                    "DAOP": daop,
                    "Reference": ref,
                    "Mean": mean,
                    "SD": sd,
                    "Delta": mean - mu0,
                    "Shapiro_p": shapiro_p,
                    "p_t_one_sided": p_t,
                    "p_wilcoxon": p_w,
                    "Cohens_d": d,
                })

df = pd.DataFrame(rows)

# Holm correction within each family
df["p_t_holm"] = np.nan
df["p_w_holm"] = np.nan

for key, idx in df.groupby(["Dataset", "Metric", "DAOP"]).groups.items():
    df.loc[idx, "p_t_holm"] = holm_adjust(df.loc[idx, "p_t_one_sided"].values)
    df.loc[idx, "p_w_holm"] = holm_adjust(df.loc[idx, "p_wilcoxon"].values)

# ----------------------------
# SIGNIFICANCE FLAGS (95% and 99%)
# ----------------------------
df["sig_t_95"] = df["p_t_holm"] < ALPHA_95
df["sig_w_95"] = df["p_w_holm"] < ALPHA_95

df["sig_t_99"] = df["p_t_holm"] < ALPHA_99
df["sig_w_99"] = df["p_w_holm"] < ALPHA_99

# ----------------------------
# PRETTY DISPLAY
# ----------------------------
def fmt_p(p):
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"

df_disp = df.copy()
df_disp["p_t_one_sided"] = df_disp["p_t_one_sided"].map(fmt_p)
df_disp["p_t_holm"] = df_disp["p_t_holm"].map(fmt_p)
df_disp["p_wilcoxon"] = df_disp["p_wilcoxon"].map(fmt_p)
df_disp["p_w_holm"] = df_disp["p_w_holm"].map(fmt_p)
df_disp["Shapiro_p"] = df_disp["Shapiro_p"].apply(lambda x: None if x is None else fmt_p(x))

# ----------------------------
# BUILD "P-TABLES" + SIGNIFICANCE TABLES
# ----------------------------
def make_ptable(metric, daop, col):
    sub = df_disp[(df_disp["Metric"] == metric) & (df_disp["DAOP"] == daop)]
    piv = sub.pivot_table(index="Reference", columns="Dataset", values=col, aggfunc="first")
    return piv.reindex(REFERENCES)

def make_sigtable(metric, daop, col):
    sub = df[(df["Metric"] == metric) & (df["DAOP"] == daop)]
    piv = sub.pivot_table(index="Reference", columns="Dataset", values=col, aggfunc="first")
    return piv.reindex(REFERENCES)

for metric in METRICS:
    for daop in DAOP_VARIANTS:
        print("\n" + "=" * 90)
        print(f"{metric} — {daop} — Holm-corrected t-test p-values")
        print(make_ptable(metric, daop, "p_t_holm"))
        print("\n" + "-" * 90)
        print(f"{metric} — {daop} — t-test significance @ 95% (True/False)")
        print(make_sigtable(metric, daop, "sig_t_95"))
        print("\n" + "-" * 90)
        print(f"{metric} — {daop} — t-test significance @ 99% (True/False)")
        print(make_sigtable(metric, daop, "sig_t_99"))
        print("\n" + "-" * 90)
        print(f"{metric} — {daop} — Holm-corrected Wilcoxon p-values")
        print(make_ptable(metric, daop, "p_w_holm"))
        print("\n" + "-" * 90)
        print(f"{metric} — {daop} — Wilcoxon significance @ 95% (True/False)")
        print(make_sigtable(metric, daop, "sig_w_95"))
        print("\n" + "-" * 90)
        print(f"{metric} — {daop} — Wilcoxon significance @ 99% (True/False)")
        print(make_sigtable(metric, daop, "sig_w_99"))

# ----------------------------
# OPTIONAL: LaTeX EXPORT
# ----------------------------
ptab = make_ptable("AUC", "DAOP Mean (ResNet-18)", "p_t_holm")
print(ptab.to_latex(escape=False, column_format="l" + "c"*len(ptab.columns)))

sigtab_99 = make_sigtable("AUC", "DAOP Mean (ResNet-18)", "sig_t_99")
print(sigtab_99.to_latex(escape=False, column_format="l" + "c"*len(sigtab_99.columns)))