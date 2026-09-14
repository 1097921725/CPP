"""Locked V1 + nested V1/V2 analysis for the integrated CPP/rhGH study.

The script exports aggregate results only. Direct identifiers are never written.
All cohort rules, model formulas, and unresolved source-data queries are explicit.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE


ROOT = Path(__file__).resolve().parents[3]
WORKBOOK = ROOT / "data" / "CPP-260721清洁版数据.xlsx"
OUT = ROOT / "analysis" / "output" / "v4.0" / "cpp_rhgh_integrated_20260910"
TABLES = OUT / "tables"
SOURCE = OUT / "figure_source_data"
for directory in (OUT, TABLES, SOURCE):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 20260910
PRIOR_AUDIT_SHA256 = "e7d9b52da6e186f79424eee255839032a119155c9d03bb36f6fafc8668f7fbd4"
FORM_MAP = {
    "长效卡式瓶": "Long-acting cartridge",
    "短效水针": "Short-acting aqueous",
    "短效粉剂": "Short-acting powder",
}
FORM_ORDER = ["Long-acting cartridge", "Short-acting aqueous", "Short-acting powder"]
REGION_ORDER = ["East", "Central", "North", "Northeast", "South", "West", "Unknown"]
PROVINCE_TO_REGION = {
    **{p: "East" for p in ["上海", "江苏", "浙江", "安徽", "福建", "江西", "山东"]},
    **{p: "Central" for p in ["河南", "湖北", "湖南"]},
    **{p: "North" for p in ["北京", "天津", "河北", "山西", "内蒙古"]},
    **{p: "Northeast" for p in ["辽宁", "吉林", "黑龙江"]},
    **{p: "South" for p in ["广东", "广西", "海南"]},
    **{p: "West" for p in ["重庆", "四川", "贵州", "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆"]},
}
ADHERENCE = "是否遵医嘱【备注：未遵医嘱包含中途换药、停药、减少用药剂量等，请根据情况如实填写】"
HEIGHT_V2_DATE = "身高测量日期（身高测量日期必须在复诊日期±14天期间）"
PAH = "预测成年身高（备注：PAH=当时身高/P(本人骨龄年龄时身高占本地区最终身高百分比) *100(cm)）【单位：CM】"
DIRECT_IDENTIFIERS = {
    "姓名", "门诊号", "住院号", "姓名字母缩写", "融合UserID", "集团UserID", "医疗UserID", "提交所用IP"
}
POSITIVE_CPP = re.compile(r"中枢性性早熟|中枢性早熟|真性性早熟|CPP", re.I)
NEGATIVE_CPP = re.compile(r"外周性性早熟|假性性早熟|非中枢性性早熟", re.I)
GNRHA_TERMS = re.compile(
    r"GNRH|亮丙瑞林|曲普瑞林|达菲林|达必佳|贝依|抑那通|博恩诺康|抑制针|抑制剂", re.I
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(series: pd.Series) -> pd.Series:
    direct = pd.to_numeric(series, errors="coerce").astype("float64")
    recover = direct.isna() & series.notna()
    if recover.any():
        extracted = series.astype("string").str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
        direct.loc[recover] = pd.to_numeric(extracted.loc[recover], errors="coerce").astype("float64")
    return direct.astype(float)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).upper()
    return re.sub(r"[\s,，;；:：。.!！、()（）\[\]【】]+", "", text)


def diagnosis_class(value: object) -> str:
    text = normalize_text(value)
    positive = bool(POSITIVE_CPP.search(text))
    negative = bool(NEGATIVE_CPP.search(text))
    if positive and negative:
        return "Ambiguous"
    if negative:
        return "Explicit exclusion"
    if positive:
        return "Explicit CPP"
    return "Non-specific/other"


def classify_region(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).split("-")[0]
    for province, region in PROVINCE_TO_REGION.items():
        if province in text:
            return region
    return "Unknown"


def classify_gnrha(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = normalize_text(value)
    if text == "否":
        return 0.0
    if GNRHA_TERMS.search(text):
        return 1.0
    return 0.0


def qsummary(series: pd.Series) -> dict[str, float | int | None]:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {"n": 0, "mean": None, "sd": None, "median": None, "q1": None, "q3": None, "min": None, "max": None}
    return {
        "n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
        "median": float(x.median()), "q1": float(x.quantile(.25)), "q3": float(x.quantile(.75)),
        "min": float(x.min()), "max": float(x.max()),
    }


def empirical_binary_icc(data: pd.DataFrame, n_boot: int = 3000) -> tuple[float, float, float]:
    grouped = data.groupby("hospital_code")["long_acting"].agg(["size", "sum"])
    n, successes = grouped["size"].to_numpy(float), grouped["sum"].to_numpy(float)

    def estimate(nn: np.ndarray, ss: np.ndarray) -> float:
        rates = ss / nn
        overall = ss.sum() / nn.sum()
        k = len(nn)
        ms_between = np.sum(nn * (rates - overall) ** 2) / (k - 1)
        ms_within = np.sum(nn * rates * (1 - rates)) / (nn.sum() - k)
        n0 = (nn.sum() - np.sum(nn ** 2) / nn.sum()) / (k - 1)
        return float((ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within))

    point = estimate(n, successes)
    rng = np.random.default_rng(SEED)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(n), len(n))
        boot[i] = estimate(n[idx], successes[idx])
    return point, float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def logistic_random_intercept_icc(
    data: pd.DataFrame,
    *,
    adjusted: bool,
    vcp_p: float = 0.5,
    min_hospital_n: int = 1,
) -> dict[str, float | int | str]:
    """Estimate hospital clustering for binary formulation use on the latent scale.

    A logistic random-intercept model is fitted with variational Bayes.  The
    hospital standard deviation is modeled on the log scale.  The latent-scale
    ICC uses the logistic level-1 variance pi^2/3.  Weakly informative prior
    sensitivity and hospital-size sensitivity are reported separately.
    """
    required = [
        "long_acting", "hospital_code", "age", "male", "height_std", "bmi_std",
        "bone_age_advancement", "concurrent_gnrha", "year_centered",
        "log_hospital_volume", "region",
    ]
    model_data = data[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    hospital_n = model_data.groupby("hospital_code").size()
    model_data = model_data[
        model_data["hospital_code"].isin(hospital_n[hospital_n >= min_hospital_n].index)
    ].copy()
    fixed = (
        "age + male + height_std + bmi_std + bone_age_advancement + concurrent_gnrha + "
        "year_centered + log_hospital_volume + C(region)"
        if adjusted else "1"
    )
    model = BinomialBayesMixedGLM.from_formula(
        f"long_acting ~ {fixed}",
        {"hospital": "0 + C(hospital_code)"},
        model_data,
        vcp_p=vcp_p,
        fe_p=2.0,
    )
    n_parameters = model.k_fep + model.k_vcp + model.k_vc
    fitted = model.fit_vb(
        mean=np.zeros(n_parameters),
        sd=np.full(n_parameters, math.exp(-0.5)),
        minim_opts={"gtol": 1e-5, "maxiter": 3000},
    )
    converged = bool(fitted.optim_retvals.get("success", False))
    gradient_norm = float(np.linalg.norm(fitted.optim_retvals.get("jac", np.array([np.nan]))))
    if not converged or not np.isfinite(gradient_norm) or gradient_norm > 1e-3:
        raise RuntimeError(
            "Hospital logistic random-intercept model failed the convergence gate: "
            + str(fitted.optim_retvals.get("message", "unknown optimizer status"))
            + f"; gradient norm={gradient_norm:.6g}"
        )
    log_sd = float(fitted.vcp_mean[0])
    log_sd_se = float(fitted.vcp_sd[0])
    hospital_sd = math.exp(log_sd)
    hospital_sd_low = math.exp(log_sd - 1.96 * log_sd_se)
    hospital_sd_high = math.exp(log_sd + 1.96 * log_sd_se)

    def latent_icc(sd: float) -> float:
        variance = sd ** 2
        return variance / (variance + math.pi ** 2 / 3)

    return {
        "model": "Adjusted" if adjusted else "Null",
        "method": "Bayesian logistic random-intercept GLMM (variational Bayes)",
        "scale": "Latent logistic",
        "n": int(len(model_data)),
        "hospitals": int(model_data["hospital_code"].nunique()),
        "min_hospital_n": int(min_hospital_n),
        "vcp_p": float(vcp_p),
        "hospital_sd": hospital_sd,
        "hospital_variance": hospital_sd ** 2,
        "icc": latent_icc(hospital_sd),
        "ci_low": latent_icc(hospital_sd_low),
        "ci_high": latent_icc(hospital_sd_high),
        "interval_type": "Approximate 95% credible interval",
        "median_odds_ratio": math.exp(0.67448975 * math.sqrt(2) * hospital_sd),
        "log_sd_posterior_mean": log_sd,
        "log_sd_posterior_sd": log_sd_se,
        "optimizer_converged": converged,
        "gradient_norm": gradient_norm,
    }


def smd_continuous(x1: pd.Series, x0: pd.Series, w1=None, w0=None) -> float:
    a, b = pd.to_numeric(x1, errors="coerce"), pd.to_numeric(x0, errors="coerce")
    if w1 is None:
        a, b = a.dropna(), b.dropna()
        if min(len(a), len(b)) < 2:
            return np.nan
        pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0
    ma = np.average(a, weights=w1); mb = np.average(b, weights=w0)
    va = np.average((a - ma) ** 2, weights=w1); vb = np.average((b - mb) ** 2, weights=w0)
    pooled = math.sqrt((va + vb) / 2)
    return float((ma - mb) / pooled) if pooled > 0 else 0.0


def smd_binary(x1: pd.Series, x0: pd.Series, w1=None, w0=None) -> float:
    a, b = pd.to_numeric(x1, errors="coerce"), pd.to_numeric(x0, errors="coerce")
    p1 = float(a.mean()) if w1 is None else float(np.average(a, weights=w1))
    p0 = float(b.mean()) if w0 is None else float(np.average(b, weights=w0))
    pooled = math.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
    return (p1 - p0) / pooled if pooled > 0 else 0.0


def format_cont(series: pd.Series) -> str:
    x = pd.to_numeric(series, errors="coerce").dropna()
    return "NA" if x.empty else f"{x.mean():.2f} ({x.std(ddof=1):.2f})"


def format_cat(series: pd.Series) -> str:
    x = pd.to_numeric(series, errors="coerce").dropna()
    return "NA" if x.empty else f"{int(x.sum())} ({100*x.mean():.1f}%)"


def prepare_v1(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "id": raw["受试者系统id"],
        "diagnosis_raw": raw["疾病诊断"].astype("string"),
        "age": numeric(raw["年龄"]), "sex": raw["性别"].astype("string"),
        "hospital_raw": raw["所在医院"].astype("string"),
        "province_city": raw["医院所在省市"].astype("string"),
        "visit_date": pd.to_datetime(raw["就诊日期"], errors="coerce"),
        "height_date": pd.to_datetime(raw["身高测量日期"], errors="coerce"),
        "height": numeric(raw["身高【单位：CM】"]), "weight": numeric(raw["体重【单位：kg】"]),
        "bmi_recorded": numeric(raw["BMI"]), "bone_age": numeric(raw["骨龄诊断（单位：岁）"]),
        "form_raw": raw["药品规格"].astype("string"),
        "concomitant_raw": raw["是否合并其他用药"].astype("string"),
    })
    df["diagnosis_class"] = df["diagnosis_raw"].map(diagnosis_class)
    df["formulation"] = df["form_raw"].map(FORM_MAP)
    df["long_acting"] = (df["formulation"] == "Long-acting cartridge").astype(int)
    df["male"] = (df["sex"] == "男").astype(int)
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    df["bone_age_advancement"] = df["bone_age"] - df["age"]
    df["region"] = df["province_city"].map(classify_region)
    df["visit_year"] = df["visit_date"].dt.year
    df["concurrent_gnrha"] = df["concomitant_raw"].map(classify_gnrha)
    df["eligible_diagnosis"] = df["diagnosis_class"].eq("Explicit CPP")
    df["eligible_age"] = df["age"].between(2, 18)
    df["eligible_sex"] = df["sex"].isin(["女", "男"])
    df["eligible_form"] = df["formulation"].notna()
    df["eligible_core"] = (
        df["hospital_raw"].notna() & df["visit_date"].notna() & df["height"].notna() & df["weight"].notna()
    )
    df["eligible_baseline"] = df[["eligible_diagnosis", "eligible_age", "eligible_sex", "eligible_form", "eligible_core"]].all(axis=1)
    return df


def prepare_v2(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "id": raw["受试者系统id"], "sex_fu": raw["性别"].astype("string"),
        "hospital_fu": raw["所在医院"].astype("string"),
        "stage_fu": raw["阶段名称"].astype("string"),
        "visit_date_fu": pd.to_datetime(raw["复诊日期"], errors="coerce"),
        "height_date_fu": pd.to_datetime(raw[HEIGHT_V2_DATE], errors="coerce"),
        "height_fu": numeric(raw["身高【单位：CM】"]),
        "bone_age_fu": numeric(raw["骨龄诊断（单位：岁）"]),
        "pah_fu": numeric(raw[PAH]), "form_raw_fu": raw["药品规格"].astype("string"),
        "adherence_fu": raw[ADHERENCE].astype("string"),
        "adverse_fu": raw["是否发生不良反应"].astype("string"),
    }).assign(formulation_fu=lambda x: x["form_raw_fu"].map(FORM_MAP))


def cohort_flow(v1: pd.DataFrame, v2: pd.DataFrame, longitudinal: pd.DataFrame) -> pd.DataFrame:
    steps = [
        ("V1 source records", len(v1)),
        ("Explicit CPP diagnosis", v1["eligible_diagnosis"].sum()),
        ("Age 2-18 years", (v1["eligible_diagnosis"] & v1["eligible_age"]).sum()),
        ("Valid sex and rhGH formulation", (v1["eligible_diagnosis"] & v1["eligible_age"] & v1["eligible_sex"] & v1["eligible_form"]).sum()),
        ("Complete baseline core variables", v1["eligible_baseline"].sum()),
        ("Linked to V2", longitudinal["id"].nunique()),
        ("Visit interval 120-240 days", longitudinal["window_primary"].sum()),
        ("Nonmissing height outcome", (longitudinal["window_primary"] & longitudinal["height_velocity_visit"].notna()).sum()),
        ("Confirmatory outcome after unresolved plausibility queries set missing", longitudinal["primary_eligible"].sum()),
    ]
    return pd.DataFrame(steps, columns=["Step", "N"]).assign(Excluded=lambda x: x["N"].shift(1) - x["N"])


def prepare_longitudinal(baseline: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    keep = baseline[baseline["eligible_baseline"]].copy()
    if keep["id"].duplicated().any() or v2["id"].duplicated().any():
        raise ValueError("G03 failed: V1 or V2 contains duplicate coded IDs")
    m = v2.merge(keep, on="id", how="inner", validate="one_to_one")
    m["sex_match"] = m["sex_fu"].eq(m["sex"])
    m["hospital_match"] = m["hospital_fu"].eq(m["hospital_raw"])
    if not m["sex_match"].all() or not m["hospital_match"].all():
        raise ValueError("G03 failed: sex or hospital differs between V1 and V2")
    m["followup_days"] = (m["visit_date_fu"] - m["visit_date"]).dt.days
    m["height_change"] = m["height_fu"] - m["height"]
    m["height_velocity_visit"] = m["height_change"] / m["followup_days"] * 365.25
    m["height_date_bl_valid"] = (m["height_date"] - m["visit_date"]).dt.days.abs().le(14)
    m["height_date_fu_valid"] = (m["height_date_fu"] - m["visit_date_fu"]).dt.days.abs().le(14)
    m["measurement_days"] = (m["height_date_fu"] - m["height_date"]).dt.days
    m["height_velocity_measurement"] = m["height_change"] / m["measurement_days"] * 365.25
    m["window_primary"] = m["followup_days"].between(120, 240)
    m["window_strict"] = m["followup_days"].between(150, 210)
    m["outcome_plausibility_query"] = ~m["height_velocity_visit"].between(0, 20)
    m.loc[m["height_velocity_visit"].isna(), "outcome_plausibility_query"] = True
    m["height_velocity_primary"] = m["height_velocity_visit"].where(~m["outcome_plausibility_query"])
    m["primary_eligible"] = m["window_primary"] & m["height_velocity_primary"].notna()
    m["stable_formulation"] = m["formulation_fu"].eq(m["formulation"])
    m["adherent"] = m["adherence_fu"].eq("是")
    m["bone_age_change"] = m["bone_age_fu"] - m["bone_age"]
    m["delta_ba_ca"] = m["bone_age_change"] / (m["followup_days"] / 365.25)
    return m


def add_analysis_columns(baseline: pd.DataFrame, longitudinal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = baseline[baseline["eligible_baseline"]].copy()
    b["hospital_code"] = pd.factorize(b["hospital_raw"])[0]
    b["hospital_volume"] = b.groupby("hospital_raw")["id"].transform("size")
    b["log_hospital_volume"] = np.log(b["hospital_volume"])
    for col in ["height", "bmi"]:
        b[f"{col}_std"] = (b[col] - b[col].mean()) / b[col].std(ddof=1)
    b["year_centered"] = b["visit_year"] - b["visit_year"].median()
    b["region"] = pd.Categorical(b["region"], categories=REGION_ORDER)
    lookup = b.set_index("id")[["hospital_code", "hospital_volume", "log_hospital_volume", "height_std", "bmi_std", "year_centered"]]
    l = longitudinal.drop(columns=[c for c in lookup.columns if c in longitudinal.columns], errors="ignore").join(lookup, on="id")
    l["region"] = pd.Categorical(l["region"], categories=REGION_ORDER)
    return b, l


def gee_fit(data: pd.DataFrame, outcome: str, formula_terms: list[str], *, family, weights=None) -> tuple[object, pd.DataFrame]:
    raw_terms = [re.fullmatch(r"C\((.+)\)", term).group(1) if re.fullmatch(r"C\((.+)\)", term) else term for term in formula_terms]
    columns = list(dict.fromkeys([outcome, "hospital_code", *raw_terms]))
    model = data[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if weights is not None:
        model["_weight"] = pd.Series(weights, index=data.index).reindex(model.index)
        model = model.dropna(subset=["_weight"])
        w = model["_weight"].to_numpy()
    else:
        w = None
    formula = f"{outcome} ~ " + " + ".join(formula_terms)
    fitted = GEE.from_formula(
        formula, groups="hospital_code", data=model, family=family,
        cov_struct=Exchangeable(), weights=w,
    ).fit(maxiter=200)
    out = pd.DataFrame({
        "term": fitted.params.index, "estimate": fitted.params.values,
        "se": fitted.bse.values, "p": fitted.pvalues.values,
    })
    out["ci_low"] = out["estimate"] - 1.96 * out["se"]
    out["ci_high"] = out["estimate"] + 1.96 * out["se"]
    out["n"] = len(model); out["hospitals"] = model["hospital_code"].nunique()
    return fitted, out


def baseline_table(data: pd.DataFrame, group: str) -> pd.DataFrame:
    rows = []
    specs = [
        ("Age, years", "age", "cont"), ("Male sex", "male", "cat"),
        ("Height, cm", "height", "cont"), ("BMI, kg/m²", "bmi", "cont"),
        ("Bone-age advancement, years", "bone_age_advancement", "cont"),
        ("Concurrent GnRHa", "concurrent_gnrha", "cat"),
        ("Visit year", "visit_year", "cont"), ("Hospital volume", "hospital_volume", "cont"),
    ]
    g1, g0 = data[data[group] == 1], data[data[group] == 0]
    for label, col, kind in specs:
        if kind == "cont":
            smd = smd_continuous(g1[col], g0[col]); values = [format_cont(data[col]), format_cont(g1[col]), format_cont(g0[col])]
        else:
            smd = smd_binary(g1[col], g0[col]); values = [format_cat(data[col]), format_cat(g1[col]), format_cat(g0[col])]
        rows.append({"Characteristic": label, "Overall": values[0], "Long-acting": values[1], "Short-acting": values[2], "SMD": smd})
    return pd.DataFrame(rows)


def followup_selection_table(baseline: pd.DataFrame) -> pd.DataFrame:
    followed = baseline[baseline["observed_primary"] == 1]
    not_followed = baseline[baseline["observed_primary"] == 0]
    rows = []
    for label, col, kind in [
        ("Long-acting rhGH", "long_acting", "cat"), ("Age, years", "age", "cont"),
        ("Male sex", "male", "cat"), ("Height, cm", "height", "cont"),
        ("BMI, kg/m²", "bmi", "cont"), ("Bone-age advancement, years", "bone_age_advancement", "cont"),
        ("Concurrent GnRHa", "concurrent_gnrha", "cat"), ("Visit year", "visit_year", "cont"),
    ]:
        if kind == "cont":
            smd = smd_continuous(followed[col], not_followed[col]); fmt = format_cont
        else:
            smd = smd_binary(followed[col], not_followed[col]); fmt = format_cat
        rows.append({"Characteristic": label, "Observed primary outcome": fmt(followed[col]), "Not observed": fmt(not_followed[col]), "SMD": smd})
    return pd.DataFrame(rows)


def make_ps_matrix(data: pd.DataFrame, include_treatment: bool) -> pd.DataFrame:
    numeric_cols = ["age", "male", "height_std", "bmi_std", "bone_age_advancement", "concurrent_gnrha", "year_centered", "log_hospital_volume"]
    if include_treatment:
        numeric_cols.append("long_acting")
    x = data[numeric_cols].copy()
    for col in x:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(pd.to_numeric(x[col], errors="coerce").median())
    region = pd.get_dummies(data["region"].fillna("Unknown"), prefix="region", drop_first=True, dtype=float)
    x = pd.concat([x, region], axis=1).astype(float)
    means, sds = x.mean(), x.std(ddof=0).replace(0, 1)
    return (x - means) / sds


def logistic_probabilities(x: pd.DataFrame, y: pd.Series) -> np.ndarray:
    model = LogisticRegression(C=np.inf, max_iter=3000, random_state=SEED)
    model.fit(x, y.astype(int))
    return model.predict_proba(x)[:, 1]


def weight_diagnostics(name: str, weights: pd.Series, treatment: pd.Series | None = None) -> dict[str, float | str]:
    w = pd.to_numeric(weights, errors="coerce").dropna()
    row = {
        "analysis": name, "n": int(w.size), "mean": float(w.mean()), "sd": float(w.std(ddof=1)),
        "min": float(w.min()), "p1": float(w.quantile(.01)), "median": float(w.median()),
        "p99": float(w.quantile(.99)), "max": float(w.max()),
        "effective_sample_size": float(w.sum() ** 2 / np.sum(w ** 2)),
    }
    if treatment is not None:
        t = treatment.reindex(w.index)
        for label, value in [("long", 1), ("short", 0)]:
            ww = w[t == value]
            row[f"ess_{label}"] = float(ww.sum() ** 2 / np.sum(ww ** 2)) if len(ww) else np.nan
    return row


def treatment_balance(data: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    rows = []
    treated, control = data["long_acting"].eq(1), data["long_acting"].eq(0)
    for label, col, kind in [
        ("Age", "age", "cont"), ("Male", "male", "binary"), ("Height", "height", "cont"),
        ("BMI", "bmi", "cont"), ("Bone-age advancement", "bone_age_advancement", "cont"),
        ("Concurrent GnRHa", "concurrent_gnrha", "binary"), ("Visit year", "visit_year", "cont"),
        ("Hospital volume", "hospital_volume", "cont"),
    ]:
        before = smd_binary(data.loc[treated, col], data.loc[control, col]) if kind == "binary" else smd_continuous(data.loc[treated, col], data.loc[control, col])
        after = smd_binary(data.loc[treated, col], data.loc[control, col], weights[treated], weights[control]) if kind == "binary" else smd_continuous(data.loc[treated, col], data.loc[control, col], weights[treated], weights[control])
        rows.append({"covariate": label, "smd_unweighted": before, "smd_overlap_weighted": after})
    return pd.DataFrame(rows)


def extract_effect(result: pd.DataFrame, term: str, analysis: str) -> dict[str, object]:
    row = result[result["term"] == term]
    if row.empty:
        return {"analysis": analysis, "term": term, "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n": 0, "hospitals": 0}
    item = row.iloc[0]
    return {"analysis": analysis, "term": term, "estimate": item.estimate, "ci_low": item.ci_low, "ci_high": item.ci_high, "p": item.p, "n": int(item.n), "hospitals": int(item.hospitals)}


def main() -> None:
    if not WORKBOOK.exists():
        raise FileNotFoundError(WORKBOOK)
    v1_raw = pd.read_excel(WORKBOOK, sheet_name="V1")
    v2_raw = pd.read_excel(WORKBOOK, sheet_name="V2")
    identifier_columns = sorted((set(v1_raw.columns) | set(v2_raw.columns)) & DIRECT_IDENTIFIERS)
    v1, v2 = prepare_v1(v1_raw), prepare_v2(v2_raw)
    if v1["id"].duplicated().any() or v2["id"].duplicated().any():
        raise ValueError("G03 failed: coded IDs are not unique")

    diagnosis_map = v1.groupby(["diagnosis_raw", "diagnosis_class"], dropna=False).size().rename("N").reset_index()
    diagnosis_map.sort_values(["diagnosis_class", "N"], ascending=[True, False]).to_csv(TABLES / "diagnosis_mapping_for_clinical_review.csv", index=False, encoding="utf-8-sig")

    longitudinal = prepare_longitudinal(v1, v2)
    baseline, longitudinal = add_analysis_columns(v1, longitudinal)
    observed_ids = set(longitudinal.loc[longitudinal["primary_eligible"], "id"])
    baseline["observed_primary"] = baseline["id"].isin(observed_ids).astype(int)

    flow = cohort_flow(v1, v2, longitudinal)
    flow.to_csv(TABLES / "table0_cohort_flow.csv", index=False, encoding="utf-8-sig")
    baseline_table(baseline, "long_acting").to_csv(TABLES / "table1_v1_baseline_characteristics.csv", index=False, encoding="utf-8-sig")
    followup_selection_table(baseline).to_csv(TABLES / "table3_followup_selection.csv", index=False, encoding="utf-8-sig")
    formulation_counts = baseline.groupby("formulation", observed=False).agg(
        n=("id", "size"), hospitals=("hospital_code", "nunique")
    ).reset_index()
    formulation_counts["percent"] = formulation_counts["n"] / len(baseline) * 100
    formulation_counts.to_csv(TABLES / "v1_formulation_counts.csv", index=False, encoding="utf-8-sig")

    baseline_terms = [
        "age", "male", "height_std", "bmi_std", "bone_age_advancement", "concurrent_gnrha",
        "year_centered", "log_hospital_volume", "C(region)",
    ]
    _, selection_model = gee_fit(baseline, "long_acting", baseline_terms, family=Binomial())
    selection_model["odds_ratio"] = np.exp(selection_model["estimate"])
    selection_model["or_ci_low"] = np.exp(selection_model["ci_low"])
    selection_model["or_ci_high"] = np.exp(selection_model["ci_high"])
    selection_model.to_csv(TABLES / "table2_formulation_selection_gee.csv", index=False, encoding="utf-8-sig")

    # Primary hospital-clustering analysis: binary logistic random-intercept
    # model on the latent scale.  Empirical observed-scale ICC is retained as a
    # complementary sensitivity analysis because the two quantities are not
    # numerically interchangeable.
    hospital_null = logistic_random_intercept_icc(baseline, adjusted=False)
    hospital_adjusted = logistic_random_intercept_icc(baseline, adjusted=True)
    hospital_adjusted["proportional_change_in_variance_vs_null"] = (
        1 - hospital_adjusted["hospital_variance"] / hospital_null["hospital_variance"]
    )
    hospital_icc, hospital_icc_low, hospital_icc_high = empirical_binary_icc(baseline)
    empirical_row = {
        "model": "Unadjusted empirical ICC",
        "method": "One-way binary ANOVA with hospital bootstrap",
        "scale": "Observed binary",
        "n": int(len(baseline)), "hospitals": int(baseline["hospital_code"].nunique()),
        "min_hospital_n": 1, "vcp_p": np.nan, "hospital_sd": np.nan,
        "hospital_variance": np.nan, "icc": hospital_icc,
        "ci_low": hospital_icc_low, "ci_high": hospital_icc_high,
        "interval_type": "Hospital-bootstrap 95% confidence interval",
        "median_odds_ratio": np.nan, "log_sd_posterior_mean": np.nan,
        "log_sd_posterior_sd": np.nan,
        "proportional_change_in_variance_vs_null": np.nan,
    }
    pd.DataFrame([hospital_adjusted, hospital_null, empirical_row]).to_csv(
        TABLES / "hospital_heterogeneity.csv", index=False, encoding="utf-8-sig"
    )

    heterogeneity_sensitivity = []
    for min_hospital_n in (1, 5, 10):
        for vcp_p in (0.25, 0.5, 1.0):
            row = logistic_random_intercept_icc(
                baseline, adjusted=True, vcp_p=vcp_p, min_hospital_n=min_hospital_n
            )
            heterogeneity_sensitivity.append(row)
    pd.DataFrame(heterogeneity_sensitivity).to_csv(
        TABLES / "hospital_heterogeneity_sensitivity.csv", index=False, encoding="utf-8-sig"
    )

    long_primary = longitudinal[longitudinal["primary_eligible"]].copy()
    long_primary["short_powder"] = (long_primary["formulation"] == "Short-acting powder").astype(int)
    long_primary["long_form"] = (long_primary["formulation"] == "Long-acting cartridge").astype(int)
    baseline_table(long_primary, "long_acting").to_csv(TABLES / "table4_longitudinal_baseline_characteristics.csv", index=False, encoding="utf-8-sig")
    followup_rates = baseline.groupby("formulation", observed=False).agg(
        baseline_n=("id", "size"), observed_primary_n=("observed_primary", "sum")
    ).reset_index()
    followup_rates["observed_primary_percent"] = followup_rates["observed_primary_n"] / followup_rates["baseline_n"] * 100
    followup_rates.to_csv(TABLES / "followup_rates_by_formulation.csv", index=False, encoding="utf-8-sig")
    growth_descriptive = long_primary.groupby("formulation", observed=False)["height_velocity_primary"].agg(
        n="size", mean="mean", sd="std", median="median",
        q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75),
    ).reset_index()
    growth_descriptive.to_csv(TABLES / "growth_response_descriptive.csv", index=False, encoding="utf-8-sig")

    outcome_terms = [
        "long_acting", "age", "male", "height_std", "bmi_std", "bone_age_advancement",
        "followup_days", "concurrent_gnrha", "year_centered", "log_hospital_volume", "C(region)",
    ]
    _, primary_model = gee_fit(long_primary, "height_velocity_primary", outcome_terms, family=Gaussian())
    primary_model.to_csv(TABLES / "table5_primary_growth_gee.csv", index=False, encoding="utf-8-sig")

    three_terms = [t for t in outcome_terms if t != "long_acting"]
    three_terms = ["long_form", "short_powder", *three_terms]
    _, three_model = gee_fit(long_primary, "height_velocity_primary", three_terms, family=Gaussian())
    three_model.to_csv(TABLES / "table6_three_formulation_exploratory.csv", index=False, encoding="utf-8-sig")

    pah_data = longitudinal[longitudinal["window_primary"] & longitudinal["pah_fu"].between(130, 200)].copy()
    _, pah_model = gee_fit(pah_data, "pah_fu", outcome_terms, family=Gaussian())
    pah_model.to_csv(TABLES / "exploratory_followup_pah_gee.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"outcome": "Bone-age change, years", **qsummary(longitudinal.loc[longitudinal["window_primary"], "bone_age_change"])},
        {"outcome": "Delta BA/CA", **qsummary(longitudinal.loc[longitudinal["window_primary"], "delta_ba_ca"])},
        {"outcome": "Follow-up PAH, cm", **qsummary(pah_data["pah_fu"])},
    ]).to_csv(TABLES / "exploratory_outcomes_descriptive.csv", index=False, encoding="utf-8-sig")

    sensitivities = [extract_effect(primary_model, "long_acting", "Primary: visit dates, 120-240 days")]
    strict = longitudinal[longitudinal["primary_eligible"] & longitudinal["window_strict"]].copy()
    _, result = gee_fit(strict, "height_velocity_primary", outcome_terms, family=Gaussian())
    sensitivities.append(extract_effect(result, "long_acting", "Strict window: 150-210 days"))

    stable = long_primary[long_primary["stable_formulation"] & long_primary["adherent"]].copy()
    _, result = gee_fit(stable, "height_velocity_primary", outcome_terms, family=Gaussian())
    sensitivities.append(extract_effect(result, "long_acting", "Stable formulation and adherent"))

    measured = longitudinal[
        longitudinal["height_date_bl_valid"] & longitudinal["height_date_fu_valid"]
        & longitudinal["measurement_days"].between(120, 240)
        & longitudinal["height_velocity_measurement"].between(0, 20)
    ].copy()
    measured["height_velocity_measurement_primary"] = measured["height_velocity_measurement"]
    _, result = gee_fit(measured, "height_velocity_measurement_primary", outcome_terms, family=Gaussian())
    sensitivities.append(extract_effect(result, "long_acting", "Validated measurement dates"))

    # Treatment overlap weights among the observed outcome cohort.
    x_treat = make_ps_matrix(long_primary, include_treatment=False)
    ps_treat = logistic_probabilities(x_treat, long_primary["long_acting"])
    ps_treat = np.clip(ps_treat, 1e-4, 1 - 1e-4)
    ow = pd.Series(np.where(long_primary["long_acting"].eq(1), 1 - ps_treat, ps_treat), index=long_primary.index)
    _, result = gee_fit(long_primary, "height_velocity_primary", outcome_terms, family=Gaussian(), weights=ow)
    sensitivities.append(extract_effect(result, "long_acting", "Treatment overlap weighting"))
    treatment_balance(long_primary, ow).to_csv(TABLES / "treatment_overlap_balance.csv", index=False, encoding="utf-8-sig")

    # Probability of having an analyzable primary outcome, estimated in the full strict-CPP V1 cohort.
    x_obs = make_ps_matrix(baseline, include_treatment=True)
    p_obs = logistic_probabilities(x_obs, baseline["observed_primary"])
    p_obs = np.clip(p_obs, 1e-4, 1 - 1e-4)
    prevalence = baseline["observed_primary"].mean()
    baseline["ipow"] = prevalence / p_obs
    observed_weights = baseline.set_index("id")["ipow"].reindex(long_primary["id"]).set_axis(long_primary.index)
    lo, hi = observed_weights.quantile([.01, .99])
    observed_weights_truncated = observed_weights.clip(lo, hi)
    _, result = gee_fit(long_primary, "height_velocity_primary", outcome_terms, family=Gaussian(), weights=observed_weights)
    sensitivities.append(extract_effect(result, "long_acting", "Follow-up probability weighting"))
    _, result = gee_fit(long_primary, "height_velocity_primary", outcome_terms, family=Gaussian(), weights=observed_weights_truncated)
    sensitivities.append(extract_effect(result, "long_acting", "Follow-up weighting, 1/99% truncated"))

    # Hospital fixed-effects sensitivity, restricted to hospitals with both exposure groups.
    overlap_hospitals = long_primary.groupby("hospital_code")["long_acting"].nunique()
    fe = long_primary[long_primary["hospital_code"].isin(overlap_hospitals[overlap_hospitals == 2].index)].copy()
    fe_formula = (
        "height_velocity_primary ~ long_acting + age + male + height_std + bmi_std + "
        "bone_age_advancement + followup_days + concurrent_gnrha + year_centered + C(hospital_code)"
    )
    fe_model = sm.OLS.from_formula(fe_formula, data=fe).fit(cov_type="cluster", cov_kwds={"groups": fe["hospital_code"]})
    sensitivities.append({
        "analysis": "Hospital fixed effects, dual-use hospitals", "term": "long_acting",
        "estimate": fe_model.params.get("long_acting", np.nan),
        "ci_low": fe_model.conf_int().loc["long_acting", 0] if "long_acting" in fe_model.params else np.nan,
        "ci_high": fe_model.conf_int().loc["long_acting", 1] if "long_acting" in fe_model.params else np.nan,
        "p": fe_model.pvalues.get("long_acting", np.nan), "n": int(fe.nunique()["id"]),
        "hospitals": int(fe["hospital_code"].nunique()),
    })

    sensitivity_df = pd.DataFrame(sensitivities)
    sensitivity_df.to_csv(TABLES / "table7_sensitivity_analyses.csv", index=False, encoding="utf-8-sig")

    weight_rows = [
        weight_diagnostics("Treatment overlap weights", ow, long_primary["long_acting"]),
        weight_diagnostics("Follow-up probability weights", observed_weights, long_primary["long_acting"]),
        weight_diagnostics("Follow-up weights truncated", observed_weights_truncated, long_primary["long_acting"]),
    ]
    pd.DataFrame(weight_rows).to_csv(TABLES / "weight_diagnostics.csv", index=False, encoding="utf-8-sig")

    # Hospital source table uses neutral codes, never source hospital names.
    hospital = baseline.groupby("hospital_code").agg(
        patients=("id", "size"), long_acting_n=("long_acting", "sum"),
        region=("region", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
    ).reset_index()
    hospital["hospital"] = hospital["hospital_code"].map(lambda x: f"Hospital {x+1:03d}")
    hospital["long_acting_percent"] = hospital["long_acting_n"] / hospital["patients"] * 100
    hospital.drop(columns="hospital_code").to_csv(SOURCE / "hospital_formulation_rates.csv", index=False, encoding="utf-8-sig")
    long_primary[["formulation", "long_acting", "height_velocity_primary", "height_change", "followup_days"]].to_csv(
        SOURCE / "growth_response_distribution.csv", index=False, encoding="utf-8-sig"
    )
    selection_model.to_csv(SOURCE / "formulation_selection_forest.csv", index=False, encoding="utf-8-sig")
    sensitivity_df.to_csv(SOURCE / "growth_sensitivity_forest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"treatment_ps": ps_treat, "overlap_weight": ow.to_numpy(), "long_acting": long_primary["long_acting"].to_numpy()}).to_csv(
        SOURCE / "treatment_overlap_distribution.csv", index=False, encoding="utf-8-sig"
    )

    primary_effect = extract_effect(primary_model, "long_acting", "Primary")
    flags = {
        "v1_duplicate_ids": int(v1["id"].duplicated().sum()),
        "v2_duplicate_ids": int(v2["id"].duplicated().sum()),
        "v2_unmatched_to_v1": int((~v2["id"].isin(set(v1["id"]))).sum()),
        "diagnosis_ambiguous": int(v1["diagnosis_class"].eq("Ambiguous").sum()),
        "linked_stage_missing": int(longitudinal["stage_fu"].isna().sum()),
        "visit_interval_nonpositive": int(longitudinal["followup_days"].le(0).sum()),
        "visit_interval_outside_120_240": int((~longitudinal["window_primary"]).sum()),
        "height_velocity_plausibility_queries": int(longitudinal["outcome_plausibility_query"].sum()),
        "negative_bone_age_change": int(longitudinal["bone_age_change"].lt(0).sum()),
        "delta_ba_ca_outside_0_3": int((~longitudinal["delta_ba_ca"].between(0, 3)).sum()),
        "formulation_switches": int((~longitudinal["stable_formulation"]).sum()),
        "nonadherent_or_modified": int((~longitudinal["adherent"]).sum()),
        "adverse_event_yes": int(longitudinal["adverse_fu"].eq("是").sum()),
    }
    summary = {
        "study_title": "Formulation choice patterns of rhGH and their association with six-month growth response among children with CPP in China",
        "workbook_sha256": sha256(WORKBOOK), "seed": SEED,
        "v1_source_n": int(len(v1)), "v2_source_n": int(len(v2)),
        "strict_cpp_v1_n": int(v1["eligible_diagnosis"].sum()),
        "baseline_cohort_n": int(len(baseline)), "baseline_hospitals": int(baseline["hospital_code"].nunique()),
        "linked_v2_n": int(len(longitudinal)), "primary_longitudinal_n": int(len(long_primary)),
        "primary_longitudinal_hospitals": int(long_primary["hospital_code"].nunique()),
        "primary_formulation_counts": {k: int(v) for k, v in long_primary["formulation"].value_counts().items()},
        "followup_days": qsummary(longitudinal["followup_days"]),
        "height_velocity_primary": qsummary(long_primary["height_velocity_primary"]),
        "primary_long_vs_short_mean_difference_cm_per_year": primary_effect,
        "hospital_heterogeneity": {
            "primary_method": hospital_adjusted["method"],
            "scale": hospital_adjusted["scale"],
            "adjusted_latent_icc": hospital_adjusted["icc"],
            "credible_interval_low": hospital_adjusted["ci_low"],
            "credible_interval_high": hospital_adjusted["ci_high"],
            "median_odds_ratio": hospital_adjusted["median_odds_ratio"],
            "null_latent_icc": hospital_null["icc"],
            "proportional_change_in_variance_vs_null": hospital_adjusted["proportional_change_in_variance_vs_null"],
            "empirical_observed_icc": hospital_icc,
            "empirical_confidence_interval_low": hospital_icc_low,
            "empirical_confidence_interval_high": hospital_icc_high,
        },
        "flags": flags,
        "direct_identifier_columns_present_but_never_exported_count": len(identifier_columns),
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "input": {"path": str(WORKBOOK), "sha256": sha256(WORKBOOK), "size_bytes": WORKBOOK.stat().st_size},
        "version_warning": {
            "prior_audit_sha256": PRIOR_AUDIT_SHA256,
            "current_sha256_differs_from_prior_audit": sha256(WORKBOOK) != PRIOR_AUDIT_SHA256,
        },
        "software": {
            "python": sys.version, "platform": platform.platform(), "pandas": pd.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "cohort_rules": {
            "diagnosis_positive": POSITIVE_CPP.pattern, "diagnosis_negative": NEGATIVE_CPP.pattern,
            "age": "2-18 years", "primary_followup_window_days": [120, 240],
            "primary_outcome_query_range_cm_per_year": [0, 20],
            "main_exposure": "V1 long-acting cartridge vs pooled short-acting rhGH",
        },
        "unresolved_source_data_policy": "Implausible primary outcomes are set missing; aggregate query counts are exported. No source value is silently changed.",
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    qc_lines = [
        "# Locked analysis QC report", "",
        f"- Workbook SHA256: `{summary['workbook_sha256']}`",
        f"- Prior-audit hash mismatch: {summary['workbook_sha256'] != PRIOR_AUDIT_SHA256}",
        f"- V1 source: {len(v1):,}; V2 source: {len(v2):,}",
        f"- Explicit CPP baseline cohort: {len(baseline):,} across {baseline['hospital_code'].nunique()} hospitals",
        f"- Linked V2: {len(longitudinal):,}; confirmatory outcome cohort: {len(long_primary):,}",
        f"- Primary adjusted long-vs-short difference: {primary_effect['estimate']:.3f} cm/year "
        f"(95% CI {primary_effect['ci_low']:.3f} to {primary_effect['ci_high']:.3f}; P={primary_effect['p']:.4g})",
        "", "## Unresolved source-data queries", "",
    ]
    qc_lines.extend([f"- {key}: {value}" for key, value in flags.items()])
    qc_lines += [
        "", "## Interpretation gate", "",
        "The inferential outputs are reproducible but are not marked source-verified. Records outside the prespecified height-velocity range were not silently corrected; they were set missing for the confirmatory analysis. Bone-age results remain exploratory until source adjudication.",
        "No direct patient identifier or source hospital name is present in any exported table.",
    ]
    (OUT / "QC_REPORT.md").write_text("\n".join(qc_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
