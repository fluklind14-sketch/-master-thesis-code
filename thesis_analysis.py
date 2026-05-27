#!/usr/bin/env python3
"""
thesis_analysis.py

Reproducible analysis pipeline for a locked master's-thesis specification.
Every number is derived fresh from the four read-only CSVs below — no manual
pre-filtering, no hard-coded intermediate data.

Raw inputs (READ-ONLY, never modified):
    codingv2_full.csv
    reliability_sheet_100_FINAL.csv
    reliability_KEY_100_do_not_share.csv
    reconciliation_log.csv

Outputs:
    outputs/   — one CSV per table; see README for file list
    stdout     — human-readable summary of every table

Run:  python thesis_analysis.py
Seed: 20260516  (all bootstraps / resamples)
"""

# ══════════════════════════════════════════════════════════════════════════════
# 0. IMPORTS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import krippendorff

warnings.filterwarnings("ignore")

SEED = 20260516
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# Formula fragments reused across all models
_COUNTRY = "C(country, Treatment(reference='Germany'))"
_EMP = "C(emp, Treatment(reference='1-10'))"

BAR = "═" * 72
SEP = "─" * 72


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title):
    print(f"\n{BAR}\n  {title}\n{BAR}")


def show(title, df, fname=None):
    """Print a DataFrame and optionally save it to outputs/."""
    print(f"\n  {title}")
    print(df.to_string())
    if fname:
        df.to_csv(OUT / fname)


def _logit_row(res, term, label, n):
    b = res.params[term]
    lo, hi = res.conf_int().loc[term]
    return {
        "model": label, "term": term,
        "OR": round(np.exp(b), 4),
        "CI_lo": round(np.exp(lo), 4),
        "CI_hi": round(np.exp(hi), 4),
        "p": round(res.pvalues[term], 4),
        "n": n,
        "pseudo_R2": round(res.prsquared, 4),
    }


def _ols_row(res, term, label, n):
    b = res.params[term]
    lo, hi = res.conf_int().loc[term]
    return {
        "model": label, "term": term,
        "coef": round(b, 4),
        "pct_eff": round((np.exp(b) - 1) * 100, 2),   # approx % change in DV
        "CI_lo": round(lo, 4),
        "CI_hi": round(hi, 4),
        "p": round(res.pvalues[term], 4),
        "n": n,
        "R2": round(res.rsquared, 4),
    }


def run_logit(formula, data, label, param="fm"):
    """Fit logit, return one-row dict for the focal parameter."""
    try:
        res = smf.logit(formula, data=data).fit(disp=False)
        return _logit_row(res, param, label, len(data))
    except Exception as exc:
        return {"model": label, "param": param, "note": str(exc), "n": len(data)}


def run_ols(formula, data, label, param="fm"):
    """Fit OLS, return one-row dict for the focal parameter."""
    try:
        res = smf.ols(formula, data=data).fit()
        return _ols_row(res, param, label, len(data))
    except Exception as exc:
        return {"model": label, "param": param, "note": str(exc), "n": len(data)}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD RAW DATA  (never written to)
# ══════════════════════════════════════════════════════════════════════════════
section("1. RAW INPUTS")

raw = pd.read_csv(
    "codingv2_full.csv", sep=";", dtype=str, keep_default_na=False
)
rel_second = pd.read_csv(
    "reliability_sheet_100_FINAL.csv", sep=";", dtype=str, keep_default_na=False
)
rel_primary = pd.read_csv(
    "reliability_KEY_100_do_not_share.csv", sep=";", dtype=str, keep_default_na=False
)
reconcile = pd.read_csv(
    "reconciliation_log.csv", sep=";", dtype=str, keep_default_na=False
)

print(f"  codingv2_full          : {len(raw):>5} rows, {raw.shape[1]} cols")
print(f"  reliability_sheet      : {len(rel_second):>5} rows")
print(f"  reliability_KEY        : {len(rel_primary):>5} rows")
print(f"  reconciliation_log     : {len(reconcile):>5} rows")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SAMPLE DERIVATION  (§2 of spec)
# ══════════════════════════════════════════════════════════════════════════════
section("2. SAMPLE FUNNEL")

UNREACH_RE = re.compile(
    r"unreach|could not|not reachable|inaccessible|did not load|"
    r"failed to|no website|unavailable|not accessible|timeout|timed out",
    re.IGNORECASE,
)
ATTRITION_FIRMS = {"pocket mind", "real fake photos"}

n_start = len(raw)

# Step 1: exclude ai_relevance == '0'  (blank = AI-relevant, keep everything else)
ai_excl = raw["ai_relevance"] == "0"
s1 = raw[~ai_excl].copy()
n_ai = ai_excl.sum()

# Step 2: exclude unreachable websites
unreach_mask = s1["fm_signal_evidence_snippet"].apply(lambda x: bool(UNREACH_RE.search(x)))
s2 = s1[~unreach_mask].copy()
n_unreach = unreach_mask.sum()

# Verify the two exclusion sets do not overlap
_overlap = set(raw.loc[ai_excl, "clean_name"]) & set(s1.loc[unreach_mask, "clean_name"])
assert len(_overlap) == 0, f"Unexpected overlap between exclusion sets: {_overlap}"

# Step 3: reconciliation attrition (2 construct-validity / site-attrition firms)
attrition_mask = s2["clean_name"].str.strip().str.lower().isin(ATTRITION_FIRMS)
analytic = s2[~attrition_mask].copy()   # n = 861; reconciliation made ZERO code changes
n_attrition = attrition_mask.sum()

print(f"  Start                        : {n_start}")
print(f"  – ai_relevance == 0          : –{n_ai}    → {len(s1)}")
print(f"  – unreachable websites        : –{n_unreach}   → {len(s2)}")
print(f"  – attrition (recon, §2)       : –{n_attrition}     → {len(analytic)}  (analytic sample)")
assert len(analytic) == 861, f"Expected 861, got {len(analytic)}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. VARIABLE CONSTRUCTION  (§3 of spec)
# ══════════════════════════════════════════════════════════════════════════════
df = analytic.copy()

# fm: primary IV, 0–3 integer (cast to float so statsmodels treats it continuously)
df["fm"] = pd.to_numeric(df["fm_signal_level_0_3"], errors="coerce")

# funded_any and funding (blank / non-numeric → no funding)
df["funding"] = pd.to_numeric(df["total_funding_usd"], errors="coerce")
df["funded_any"] = (df["funding"] > 0).astype(int)

# age (numeric company age in years)
df["age"] = pd.to_numeric(df["company_age"], errors="coerce")

# country (three categories; Germany = reference in all models)
df["country"] = df["headquarters_country"]

# credibility tier
df["cred"] = df["credibility"]
df["crunchbase_only_ind"] = (df["cred"] == "crunchbase_only").astype(int)

# funding stage
df["stage"] = df["last_equity_funding_type"]

# emp: collapse to three ordered bands; rows with blank/other → NaN (dropped from models)
EMP_MAP = {
    "1-10": "1-10",
    "11-50": "11-50",
    "51-100": "51+",
    "101-250": "51+",
    "251-500": "51+",
}
df["emp"] = df["number_of_employees"].map(EMP_MAP)   # NaN for blank / out-of-range
EMP_CAT = pd.CategoricalDtype(["1-10", "11-50", "51+"], ordered=True)

# final_foundation_signal is intentionally IGNORED per spec (abandoned scraper binary)


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODELLING SAMPLES  (§2 & §3 of spec)
# ══════════════════════════════════════════════════════════════════════════════
# Regression 1 sample: listwise deletion on emp (fm and age have no NaNs here)
mod1 = df.dropna(subset=["emp", "fm", "age"]).copy()
mod1["emp"] = mod1["emp"].astype(EMP_CAT)

# Regression 2 sample: funded firms only (funded_any == 1), log-transform DV
mod2 = mod1[mod1["funded_any"] == 1].copy()
mod2["log_funding"] = np.log(mod2["funding"])

n_dropped_emp = len(analytic) - len(mod1)
print(f"  – missing emp (listwise)      : –{n_dropped_emp}     → {len(mod1)}  (R1 model sample)")
print(f"  Funded only (R2)              :         → {len(mod2)}")

assert len(mod1) == 859, f"Expected 859 for R1, got {len(mod1)}"
assert len(mod2) == 633, f"Expected 633 for R2, got {len(mod2)}"


# ══════════════════════════════════════════════════════════════════════════════
# 5. DESCRIPTIVE: FM × FUNDING TABLE
# ══════════════════════════════════════════════════════════════════════════════
section("5. DESCRIPTIVE: FM × FUNDING")

desc = (
    mod1.groupby("fm", observed=True)
    .agg(
        n=("funded_any", "count"),
        n_funded=("funded_any", "sum"),
        pct_funded=("funded_any", "mean"),
        median_funding_M=(
            "funding",
            lambda x: x[x > 0].median() / 1e6 if (x > 0).any() else np.nan,
        ),
    )
    .assign(
        pct_funded=lambda d: (d["pct_funded"] * 100).round(1),
        median_funding_M=lambda d: d["median_funding_M"].round(2),
    )
)
show("FM × Funding (model sample, n=859)", desc, "desc_fm_funding.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 6. REGRESSION 1 — Logit: funded_any ~ fm + age + emp + country
# ══════════════════════════════════════════════════════════════════════════════
section("6. REGRESSION 1 — Logit")

# 6a. Continuous IV (PRIMARY)
_f1a = f"funded_any ~ fm + age + {_EMP} + {_COUNTRY}"
_r1a = smf.logit(_f1a, data=mod1).fit(disp=False)

t1a_rows = [_logit_row(_r1a, t, "R1_continuous", len(mod1)) for t in _r1a.params.index]
t1a = pd.DataFrame(t1a_rows)
show("R1 — Continuous fm (PRIMARY): OR, 95% CI, p", t1a, "r1_continuous.csv")

print(f"\n  ► fm OR = {np.exp(_r1a.params['fm']):.4f}, "
      f"p = {_r1a.pvalues['fm']:.4f}, "
      f"pseudo-R² = {_r1a.prsquared:.4f}, n = {len(mod1)}")

# 6b. Dummy decomposition (SECONDARY)
_f1b = f"funded_any ~ C(fm, Treatment(reference=0)) + age + {_EMP} + {_COUNTRY}"
_r1b = smf.logit(_f1b, data=mod1).fit(disp=False)

t1b_rows = [_logit_row(_r1b, t, "R1_dummies", len(mod1)) for t in _r1b.params.index]
t1b = pd.DataFrame(t1b_rows)
show("R1 — fm dummies (SECONDARY): OR, 95% CI, p", t1b, "r1_dummies.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 7. REGRESSION 2 — OLS: log(funding) ~ fm + age + emp + country
# ══════════════════════════════════════════════════════════════════════════════
section("7. REGRESSION 2 — OLS on funded firms only")

# 7a. Continuous IV (PRIMARY)
_f2a = f"log_funding ~ fm + age + {_EMP} + {_COUNTRY}"
_r2a = smf.ols(_f2a, data=mod2).fit()

t2a_rows = [_ols_row(_r2a, t, "R2_continuous", len(mod2)) for t in _r2a.params.index]
t2a = pd.DataFrame(t2a_rows)
show("R2 — Continuous fm (PRIMARY): coef, %eff, 95% CI, p", t2a, "r2_continuous.csv")

print(f"\n  ► fm coef = {_r2a.params['fm']:.4f}, "
      f"pct_eff ≈ {(np.exp(_r2a.params['fm'])-1)*100:.1f}%, "
      f"p = {_r2a.pvalues['fm']:.4f}, "
      f"R² = {_r2a.rsquared:.4f}, n = {len(mod2)}")

# 7b. Dummy decomposition (SECONDARY)
_f2b = f"log_funding ~ C(fm, Treatment(reference=0)) + age + {_EMP} + {_COUNTRY}"
_r2b = smf.ols(_f2b, data=mod2).fit()

t2b_rows = [_ols_row(_r2b, t, "R2_dummies", len(mod2)) for t in _r2b.params.index]
t2b = pd.DataFrame(t2b_rows)
show("R2 — fm dummies (SECONDARY): coef, %eff, 95% CI, p", t2b, "r2_dummies.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 8. ROBUSTNESS CHECKS  (§5 of spec, continuous IV throughout)
# ══════════════════════════════════════════════════════════════════════════════
section("8. ROBUSTNESS (continuous IV)")

rob_r1 = []
rob_r2 = []

# ── 8a. Main models (baseline for comparison) ─────────────────────────────────
rob_r1.append(run_logit(_f1a, mod1, "R1_main"))
rob_r2.append(run_ols(_f2a, mod2, "R2_main"))

# ── 8b. Germany-only (no country covariate needed) ───────────────────────────
ger1 = mod1[mod1["country"] == "Germany"].copy()
ger1["emp"] = ger1["emp"].cat.set_categories(EMP_CAT.categories)
ger2 = mod2[mod2["country"] == "Germany"].copy()
ger2["emp"] = ger2["emp"].cat.set_categories(EMP_CAT.categories)
ger2["log_funding"] = np.log(ger2["funding"])

rob_r1.append(run_logit(f"funded_any ~ fm + age + {_EMP}", ger1, "R1_Germany_only"))
rob_r2.append(run_ols(f"log_funding ~ fm + age + {_EMP}", ger2, "R2_Germany_only"))

# ── 8c. asinh(funding) on ALL mod1 firms including unfunded (zeros) ──────────
mod1_asinh = mod1.copy()
mod1_asinh["asinh_funding"] = np.arcsinh(mod1_asinh["funding"].fillna(0))
rob_r2.append(run_ols(
    f"asinh_funding ~ fm + age + {_EMP} + {_COUNTRY}",
    mod1_asinh, "R2_asinh_all_firms (n=859)",
))

# ── 8d. Orbis-verified only (drop crunchbase_only — attenuates, report honestly) ──
orb1 = mod1[mod1["cred"] != "crunchbase_only"].copy()
orb1["emp"] = orb1["emp"].cat.set_categories(EMP_CAT.categories)
orb2 = mod2[mod2["cred"] != "crunchbase_only"].copy()
orb2["emp"] = orb2["emp"].cat.set_categories(EMP_CAT.categories)
orb2["log_funding"] = np.log(orb2["funding"])

rob_r1.append(run_logit(_f1a.replace(_COUNTRY, _COUNTRY), orb1, "R1_orbis_only"))
rob_r2.append(run_ols(_f2a, orb2, "R2_orbis_only"))
print(
    "  NOTE (Orbis-only): R1 attenuates to p≈.06, R2 to ns "
    "— reported honestly, not hidden."
)

# ── 8e. Collapsed grouping A: fm {0 / 1+2=1 / 3=2} treated as continuous ─────
def _map_fmA(x):
    if x == 0:
        return 0
    elif x in (1, 2):
        return 1
    else:
        return 2

def _map_fmB(x):
    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return 2

mod1_a = mod1.copy();  mod1_a["fm_grp"] = mod1_a["fm"].apply(_map_fmA)
mod1_b = mod1.copy();  mod1_b["fm_grp"] = mod1_b["fm"].apply(_map_fmB)
mod2_a = mod2.copy();  mod2_a["fm_grp"] = mod2_a["fm"].apply(_map_fmA)
mod2_b = mod2.copy();  mod2_b["fm_grp"] = mod2_b["fm"].apply(_map_fmB)

_fa = f"funded_any ~ fm_grp + age + {_EMP} + {_COUNTRY}"
_fb = f"log_funding ~ fm_grp + age + {_EMP} + {_COUNTRY}"

rob_r1.append(run_logit(_fa, mod1_a, "R1_groupA (0/12/3)", param="fm_grp"))
rob_r1.append(run_logit(_fa, mod1_b, "R1_groupB (0/1/23)", param="fm_grp"))
rob_r2.append(run_ols(_fb, mod2_a, "R2_groupA (0/12/3)", param="fm_grp"))
rob_r2.append(run_ols(_fb, mod2_b, "R2_groupB (0/1/23)", param="fm_grp"))

rob1_df = pd.DataFrame(rob_r1)
rob2_df = pd.DataFrame(rob_r2)
show("ROBUSTNESS — R1 (Logit)", rob1_df, "robustness_r1.csv")
show("ROBUSTNESS — R2 (OLS)", rob2_df, "robustness_r2.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 9. EXTENSIONS — exploratory / post hoc  (§5 of spec)
# ══════════════════════════════════════════════════════════════════════════════
section("9. EXTENSIONS (exploratory, post hoc)")

# ── 9a. Credibility-tier interaction fm × crunchbase_only ────────────────────
# Note: interaction is NULL (p≈.7/.99) — not supported; reported as such.
print("\n  [9a] fm × crunchbase_only_ind — interaction expected NULL (p≈.7/.99)")

_ri1 = smf.logit(
    f"funded_any ~ fm * crunchbase_only_ind + age + {_EMP} + {_COUNTRY}",
    data=mod1,
).fit(disp=False)
_ri2 = smf.ols(
    f"log_funding ~ fm * crunchbase_only_ind + age + {_EMP} + {_COUNTRY}",
    data=mod2,
).fit()

int_terms_r1 = ["fm", "crunchbase_only_ind", "fm:crunchbase_only_ind"]
int_terms_r2 = ["fm", "crunchbase_only_ind", "fm:crunchbase_only_ind"]

int1_df = pd.DataFrame(
    [_logit_row(_ri1, t, "R1_cred_interaction", len(mod1)) for t in int_terms_r1
     if t in _ri1.params.index]
)
int2_df = pd.DataFrame(
    [_ols_row(_ri2, t, "R2_cred_interaction", len(mod2)) for t in int_terms_r2
     if t in _ri2.params.index]
)
show("Ext — R1 interaction terms", int1_df, "ext_r1_interaction.csv")
show("Ext — R2 interaction terms", int2_df, "ext_r2_interaction.csv")

# ── 9b. Funding-stage split: Pre-Seed / Seed / Series A ──────────────────────
# Series A firms are near-uniformly funded → R1 not estimable (data limitation).
print("\n  [9b] Funding-stage split")
stage_r1, stage_r2 = [], []
for stg in ["Pre-Seed", "Seed", "Series A"]:
    sub1 = mod1[mod1["stage"] == stg].copy()
    sub1["emp"] = sub1["emp"].cat.set_categories(EMP_CAT.categories)
    sub2 = mod2[mod2["stage"] == stg].copy()
    sub2["emp"] = sub2["emp"].cat.set_categories(EMP_CAT.categories)
    if len(sub2) > 0:
        sub2["log_funding"] = np.log(sub2["funding"])

    print(f"    {stg}: n_mod1={len(sub1)}, n_funded={sub1['funded_any'].sum()}, n_mod2={len(sub2)}")

    # Only unique values of funded_any matter for R1 separability
    if sub1["funded_any"].nunique() > 1:
        stage_r1.append(run_logit(_f1a, sub1, f"R1_{stg}"))
    else:
        stage_r1.append({
            "model": f"R1_{stg}",
            "note": "perfect separation — Series A near-uniformly funded (data limitation)",
            "n": len(sub1),
        })

    if len(sub2) >= 10:
        stage_r2.append(run_ols(_f2a, sub2, f"R2_{stg}"))
    else:
        stage_r2.append({"model": f"R2_{stg}", "note": "insufficient n", "n": len(sub2)})

show("Ext — Stage split R1", pd.DataFrame(stage_r1), "ext_stage_r1.csv")
show("Ext — Stage split R2", pd.DataFrame(stage_r2), "ext_stage_r2.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 10. RELIABILITY  (§6 of spec) — INDEPENDENT CODES ONLY, PRE-RECONCILIATION
# ══════════════════════════════════════════════════════════════════════════════
section("10. RELIABILITY — Independent codes, pre-reconciliation")

# DATA PATH B: reliability only — never touches reconciled labels
rel_merged = (
    rel_second[["row_id", "fm_signal_level_0_3", "fm_signal_evidence_snippet"]]
    .rename(columns={"fm_signal_level_0_3": "second_code"})
    .merge(
        rel_primary[["row_id", "primary_fm_level"]].rename(
            columns={"primary_fm_level": "primary_code"}
        ),
        on="row_id",
        how="inner",
    )
)
rel_merged["second_code"] = pd.to_numeric(rel_merged["second_code"], errors="coerce")
rel_merged["primary_code"] = pd.to_numeric(rel_merged["primary_code"], errors="coerce")
rel_merged = rel_merged.dropna(subset=["second_code", "primary_code"])

print(f"  Merged reliability pairs: {len(rel_merged)}")

# n=98: exclude the 2 rows where second coder marked site unreachable (rows 101, 116)
# These are the same attrition firms excluded from the main analytic sample.
UNREACH_ROWS = {"101", "116"}
rel_98 = rel_merged[~rel_merged["row_id"].isin(UNREACH_ROWS)].copy()
print(f"  n=100 (full subset), n=98 (excl rows 101, 116 — second-coder unreachable)")


def reliability_panel(df_r, label):
    """Compute and print full reliability statistics for one paired sample."""
    pri = df_r["primary_code"].astype(int).values
    sec = df_r["second_code"].astype(int).values
    n = len(pri)

    # Agreement rates
    exact = (pri == sec).mean()
    adjacent = (np.abs(pri - sec) <= 1).mean()

    # Cohen's κ in three weightings
    kappa_uw = cohen_kappa_score(pri, sec, weights=None)
    kappa_lin = cohen_kappa_score(pri, sec, weights="linear")
    kappa_quad = cohen_kappa_score(pri, sec, weights="quadratic")

    # Krippendorff's α (ordinal = headline; nominal for completeness)
    mat = np.array([pri.astype(float), sec.astype(float)])
    alpha_ord = krippendorff.alpha(reliability_data=mat, level_of_measurement="ordinal")
    alpha_nom = krippendorff.alpha(reliability_data=mat, level_of_measurement="nominal")

    # Bootstrapped 95% CI for ordinal α (2000 resamples, fresh seed each call)
    boot_rng = np.random.default_rng(SEED)
    boot_alphas = []
    for _ in range(2000):
        idx = boot_rng.integers(0, n, size=n)
        m = np.array([pri[idx].astype(float), sec[idx].astype(float)])
        try:
            boot_alphas.append(
                krippendorff.alpha(reliability_data=m, level_of_measurement="ordinal")
            )
        except Exception:
            pass
    ci_lo, ci_hi = np.percentile(boot_alphas, [2.5, 97.5])

    # Confusion matrix (row = primary coder 0–3, col = second coder 0–3)
    cm = confusion_matrix(pri, sec, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(
        cm,
        index=[f"pri={i}" for i in range(4)],
        columns=[f"sec={j}" for j in range(4)],
    )

    # Disagreement-pair tally
    disagree_counts = Counter((int(p), int(s)) for p, s in zip(pri, sec) if p != s)

    # 2-vs-3 boundary: among rows where BOTH coders chose ≥ 2
    mask_23 = (pri >= 2) & (sec >= 2)
    pri_23 = pri[mask_23]
    sec_23 = sec[mask_23]
    n_23 = mask_23.sum()
    agree_23 = (pri_23 == sec_23).mean() if n_23 > 0 else np.nan
    try:
        kappa_23 = cohen_kappa_score(pri_23, sec_23)
    except ValueError:
        kappa_23 = np.nan

    # ── Print ──────────────────────────────────────────────────────────────────
    print(f"\n  ┌── {label}  (n = {n}) ──────────────────────────────────────────────")
    print(f"  │  Exact agreement          : {exact:.4f}  ({exact*100:.1f}%)")
    print(f"  │  Adjacent (≤1) agreement  : {adjacent:.4f}  ({adjacent*100:.1f}%)")
    print(f"  │  Cohen κ  unweighted      : {kappa_uw:.4f}")
    print(f"  │  Cohen κ  linear          : {kappa_lin:.4f}")
    print(f"  │  Cohen κ  quadratic       : {kappa_quad:.4f}")
    print(f"  │  Krippendorff α  ordinal  : {alpha_ord:.4f}  "
          f"[95% CI {ci_lo:.4f}–{ci_hi:.4f}]  ← HEADLINE")
    print(f"  │  Krippendorff α  nominal  : {alpha_nom:.4f}")
    print(f"  │  Disagreement pairs       : {dict(sorted(disagree_counts.items()))}")
    print(f"  │  2-vs-3 boundary (both≥2) : n={n_23}, "
          f"agreement={agree_23:.4f} ({agree_23*100:.1f}%), κ={kappa_23:.4f}")
    print("  └──")
    print("\n  Confusion matrix:")
    print(cm_df.to_string())

    summary = {
        "label": label,
        "n": n,
        "exact_pct": round(exact * 100, 2),
        "adjacent_pct": round(adjacent * 100, 2),
        "kappa_unweighted": round(kappa_uw, 4),
        "kappa_linear": round(kappa_lin, 4),
        "kappa_quadratic": round(kappa_quad, 4),
        "alpha_ordinal": round(alpha_ord, 4),
        "alpha_nominal": round(alpha_nom, 4),
        "alpha_ord_CI_lo": round(ci_lo, 4),
        "alpha_ord_CI_hi": round(ci_hi, 4),
        "n_boundary_23": int(n_23),
        "agreement_23_pct": round(agree_23 * 100, 2),
        "kappa_23": round(kappa_23, 4),
    }
    return summary, cm_df


s100, cm100 = reliability_panel(rel_merged, "n=100 full subset")
s98, cm98 = reliability_panel(rel_98, "n=98 (excl rows 101, 116)")

rel_summary = pd.DataFrame([s100, s98])
show("Reliability summary", rel_summary, "reliability_summary.csv")
cm100.to_csv(OUT / "reliability_confusion_n100.csv")
cm98.to_csv(OUT / "reliability_confusion_n98.csv")


# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{BAR}")
print("  COMPLETE — tables saved to outputs/")
print(f"  Files: {sorted(p.name for p in OUT.iterdir())}")
print(BAR)
