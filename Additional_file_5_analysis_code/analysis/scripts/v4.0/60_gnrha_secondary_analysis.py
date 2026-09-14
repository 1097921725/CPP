"""Prespecified secondary effectiveness analysis by recorded concurrent GnRHa.

This script reuses the locked cohort construction from analysis 44 and exports
aggregate results only. It performs no safety analysis and writes no identifiers.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t
from statsmodels.genmod.cov_struct import Exchangeable


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("locked_analysis", HERE / "44_cpp_rhgh_integrated_analysis.py")
LOCKED = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(LOCKED)

OUT = LOCKED.OUT / "gnrha_secondary_20260911"
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_gee(data: pd.DataFrame, formula: str):
    fitted = sm.GEE.from_formula(
        formula,
        groups="hospital_code",
        data=data,
        family=sm.families.Gaussian(),
        cov_struct=Exchangeable(),
    ).fit(maxiter=200, cov_type="bias_reduced")
    if not fitted.converged:
        raise RuntimeError(f"GEE did not converge: {formula}")
    return fitted


def contrast(
    result, weights: dict[str, float], label: str, n: int, hospitals: int, unit: str,
    *, reference_hospitals: int | None = None,
) -> dict:
    names = list(result.params.index)
    vector = np.array([weights.get(name, 0.0) for name in names], dtype=float)
    estimate = float(vector @ result.params.to_numpy())
    variance = float(vector @ np.asarray(result.cov_params()) @ vector)
    se = float(np.sqrt(max(variance, 0.0)))
    # Contrasts from an interaction model use the cluster count of that full
    # fitted model, even when the displayed N/hospital count describes a stratum.
    df = (reference_hospitals if reference_hospitals is not None else hospitals) - 1
    critical = float(student_t.ppf(0.975, df))
    p = float(2 * student_t.sf(abs(estimate / se), df)) if se > 0 else np.nan
    return {
        "analysis": label,
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
        "p": p,
        "n": int(n),
        "hospitals": int(hospitals),
        "df": int(df),
        "unit": unit,
        "inference": "bias-reduced hospital-clustered GEE; t reference with hospitals minus 1 df",
    }


def main() -> None:
    expected_hash = "2c4e4f510154ab74ebfd1d9d17b2be7d88b98d3d51f2007a8f029cb39d5861aa"
    observed_hash = sha256(LOCKED.WORKBOOK)
    if observed_hash != expected_hash:
        raise RuntimeError(f"Workbook hash changed: {observed_hash}")

    v1 = LOCKED.prepare_v1(pd.read_excel(LOCKED.WORKBOOK, sheet_name="V1"))
    v2 = LOCKED.prepare_v2(pd.read_excel(LOCKED.WORKBOOK, sheet_name="V2"))
    baseline, longitudinal = LOCKED.add_analysis_columns(v1, LOCKED.prepare_longitudinal(v1, v2))
    primary = longitudinal.loc[longitudinal["primary_eligible"]].copy()

    if len(baseline) != 3606 or len(primary) != 735:
        raise RuntimeError(f"Locked cohort mismatch: baseline={len(baseline)}, primary={len(primary)}")
    if primary["concurrent_gnrha"].isna().any():
        raise RuntimeError("Missing recorded-concomitant classification in primary cohort")

    covariates = (
        "age + male + height_std + bmi_std + bone_age_advancement + followup_days + "
        "year_centered + log_hospital_volume + C(region)"
    )
    subgroup = primary.loc[primary["concurrent_gnrha"].eq(1)].copy()
    subgroup_model = fit_gee(subgroup, "height_velocity_primary ~ long_acting + " + covariates)
    subgroup_effect = contrast(
        subgroup_model, {"long_acting": 1.0},
        "Recorded concurrent GnRHa subgroup: long-acting versus pooled short-acting",
        len(subgroup), subgroup["hospital_code"].nunique(), "cm/year",
    )

    interaction_model = fit_gee(
        primary,
        "height_velocity_primary ~ long_acting * concurrent_gnrha + " + covariates,
    )
    hospitals = primary["hospital_code"].nunique()
    no_record_effect = contrast(
        interaction_model, {"long_acting": 1.0},
        "No explicit GnRHa record: long-acting versus pooled short-acting",
        int(primary["concurrent_gnrha"].eq(0).sum()),
        primary.loc[primary["concurrent_gnrha"].eq(0), "hospital_code"].nunique(), "cm/year",
        reference_hospitals=hospitals,
    )
    recorded_effect = contrast(
        interaction_model, {"long_acting": 1.0, "long_acting:concurrent_gnrha": 1.0},
        "Recorded concurrent GnRHa: long-acting versus pooled short-acting (interaction model)",
        len(subgroup), subgroup["hospital_code"].nunique(), "cm/year",
        reference_hospitals=hospitals,
    )
    interaction = contrast(
        interaction_model, {"long_acting:concurrent_gnrha": 1.0},
        "Formulation-by-recorded-GnRHa interaction",
        len(primary), hospitals, "cm/year difference in contrasts",
    )

    subgroup = subgroup.assign(days180=subgroup["followup_days"] - 180)
    ancova_model = fit_gee(
        subgroup,
        "height_fu ~ long_acting * days180 + age + male + height_std + bmi_std + "
        "bone_age_advancement + year_centered + log_hospital_volume + C(region)",
    )
    ancova_effect = contrast(
        ancova_model, {"long_acting": 1.0},
        "Recorded concurrent GnRHa subgroup: adjusted height contrast at day 180",
        len(subgroup), subgroup["hospital_code"].nunique(), "cm",
    )

    effects = pd.DataFrame([subgroup_effect, recorded_effect, no_record_effect, interaction, ancova_effect])
    effects.to_csv(OUT / "gnrha_effectiveness_estimates.csv", index=False, encoding="utf-8-sig")

    counts = (
        primary.groupby(["concurrent_gnrha", "long_acting"], observed=False)
        .agg(n=("id", "size"), hospitals=("hospital_code", "nunique"),
             mean_velocity=("height_velocity_primary", "mean"),
             sd_velocity=("height_velocity_primary", "std"),
             mean_followup_days=("followup_days", "mean"))
        .reset_index()
    )
    counts["record_status"] = counts["concurrent_gnrha"].map({1.0: "Recorded concurrent GnRHa", 0.0: "No explicit GnRHa record"})
    counts["formulation_group"] = counts["long_acting"].map({1: "Long-acting", 0: "Pooled short-acting"})
    counts.to_csv(OUT / "gnrha_effectiveness_descriptive.csv", index=False, encoding="utf-8-sig")

    model_terms = []
    for model_name, result in [("subgroup", subgroup_model), ("interaction", interaction_model), ("day180_ancova", ancova_model)]:
        for term in result.params.index:
            model_terms.append({
                "model": model_name, "term": term, "estimate": float(result.params[term]),
                "se": float(result.bse[term]), "p_normal_reference": float(result.pvalues[term]),
                "converged": bool(result.converged), "working_correlation": float(np.asarray(result.cov_struct.dep_params)),
            })
    pd.DataFrame(model_terms).to_csv(OUT / "gnrha_model_terms.csv", index=False, encoding="utf-8-sig")

    summary = {
        "workbook_sha256": observed_hash,
        "analysis_scope": "effectiveness only; no safety analysis",
        "record_definition": "Explicit GnRHa-related term in the baseline concomitant-medication field; absence is labelled no explicit record, not non-use.",
        "baseline_n": int(len(baseline)),
        "primary_n": int(len(primary)),
        "primary_hospitals": int(hospitals),
        "recorded_gnrha_n": int(len(subgroup)),
        "recorded_gnrha_hospitals": int(subgroup["hospital_code"].nunique()),
        "no_explicit_record_n": int(primary["concurrent_gnrha"].eq(0).sum()),
        "effects": effects.to_dict(orient="records"),
    }
    (OUT / "gnrha_secondary_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
