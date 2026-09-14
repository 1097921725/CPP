"""Aggregate audit of baseline concomitant-medication classification; no identifiers."""
from pathlib import Path
import importlib.util
import pandas as pd

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m", here / "44_cpp_rhgh_integrated_analysis.py")
m = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(m)
v1 = m.prepare_v1(pd.read_excel(m.WORKBOOK, sheet_name="V1"))
v2 = m.prepare_v2(pd.read_excel(m.WORKBOOK, sheet_name="V2"))
_, longitudinal = m.add_analysis_columns(v1, m.prepare_longitudinal(v1, v2))
primary = longitudinal[longitudinal.primary_eligible].copy()
audit = (primary.groupby(["concurrent_gnrha", "concomitant_raw"], dropna=False).size()
         .rename("n").reset_index().sort_values(["concurrent_gnrha", "n"], ascending=[True, False]))
out = m.OUT / "gnrha_secondary_20260911" / "gnrha_record_classification_audit.csv"
audit.to_csv(out, index=False, encoding="utf-8-sig")
print(audit.to_string(index=False))
print(out)
