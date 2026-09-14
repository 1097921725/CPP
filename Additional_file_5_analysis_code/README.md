# Reproducible analysis code

This archive contains the Python scripts used for the reported cohort construction, hospital-variation analysis, growth-outcome analysis, numerical audits, GnRHa exploratory analysis, and publication figures. It contains no individual-level clinical data.

## Required input

Place the authorized source workbook at:

`data/CPP-260721清洁版数据.xlsx`

Expected SHA-256:

`2c4e4f510154ab74ebfd1d9d17b2be7d88b98d3d51f2007a8f029cb39d5861aa`

Expected size: 5,379,606 bytes. The workbook must contain sheets named `V1` and `V2`. These worksheet names are source-file identifiers only; manuscript reporting uses professional cohort terminology.

## Environment

Tested with Python 3.14.6. Install pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run

From this archive's root:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

To specify another Python executable:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Python "C:\path\to\python.exe"
```

Scripts run in this order:

1. `44_cpp_rhgh_integrated_analysis.py`: locked cohort construction and primary/sensitivity analyses.
2. `49_audit_hospital_icc.py`: independent maximum-likelihood ICC audit.
3. `50_audit_growth_methods.py`: growth-model specification and small-sample inference checks.
4. `60_gnrha_secondary_analysis.py`: exploratory effectiveness analyses by documented concomitant GnRHa use.
5. `67_audit_gnrha_record_classification.py`: audit of recorded GnRHa classification.
6. `53_nature_figures_audited.py`: main and supplementary audited figures.
7. `63_plot_gnrha_effectiveness.py`: Supplementary Figure S3.
8. `69_redraw_figure1_professional_labels.py`: final Figure 1 labels.

Outputs are written under `analysis/output/v4.0/cpp_rhgh_integrated_20260910/` relative to this archive root. The scripts preserve the manuscript's locked bone-age handling and treatment-information definitions.

## Reproducibility scope

The code reproduces analyses from the authorized workbook. It does not redistribute raw data, medical-record images, or direct identifiers. Minor graphical differences can occur if substitute fonts are used. Review all outputs before scientific use.
