"""Diagnostic growth-model checks; does not overwrite the primary analysis."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t
from statsmodels.genmod.cov_struct import Exchangeable, Independence

path=Path(__file__).with_name('44_cpp_rhgh_integrated_analysis.py')
spec=importlib.util.spec_from_file_location('original',path)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
v1=m.prepare_v1(pd.read_excel(m.WORKBOOK,sheet_name='V1'))
v2=m.prepare_v2(pd.read_excel(m.WORKBOOK,sheet_name='V2'))
b,d=m.add_analysis_columns(v1,m.prepare_longitudinal(v1,v2))
p=d[d.primary_eligible].copy()
p['days180']=p.followup_days-180
terms='long_acting + age + male + height_std + bmi_std + bone_age_advancement + followup_days + concurrent_gnrha + year_centered + log_hospital_volume + C(region)'
rows=[]
def fit(label,data,formula,cov=None,bias=False,term='long_acting',unit='cm/year'):
    model=sm.GEE.from_formula(formula,groups='hospital_code',data=data,family=sm.families.Gaussian(),cov_struct=cov if cov is not None else Exchangeable())
    result=model.fit(maxiter=200,cov_type='bias_reduced' if bias else 'robust')
    ncluster=data.hospital_code.nunique()
    crit=t.ppf(.975,ncluster-1) if bias else 1.959963984540054
    est=float(result.params[term]); se=float(result.bse[term])
    dep=result.cov_struct.dep_params
    row=dict(analysis=label,estimate=est,se=se,ci_low=est-crit*se,ci_high=est+crit*se,n=int(result.nobs),hospitals=int(ncluster),unit=unit,converged=bool(result.converged),working_correlation=None if dep is None else float(np.asarray(dep)))
    rows.append(row); print(json.dumps(row),flush=True)
    return result
fit('Original GEE reproduced',p,'height_velocity_primary ~ '+terms)
fit('Bias-reduced GEE, t reference',p,'height_velocity_primary ~ '+terms,bias=True)
fit('Independent working correlation',p,'height_velocity_primary ~ '+terms,cov=Independence(),bias=True)
nonlinear=terms.replace(' + age +',' + bs(age, df=4, degree=3) +').replace(' + bone_age_advancement +',' + bs(bone_age_advancement, df=4, degree=3) +').replace(' + year_centered +',' + C(visit_year) +')
fit('Nonlinear age and bone-age advancement, categorical year',p,'height_velocity_primary ~ '+nonlinear,bias=True)
ancova=terms.replace('long_acting','long_acting * days180').replace(' + followup_days','')
fit('Follow-up height ANCOVA, contrast at day 180',p,'height_fu ~ '+ancova,bias=True,unit='cm at day 180')
measured=d[d.height_date_bl_valid & d.height_date_fu_valid & d.measurement_days.between(120,240) & d.height_velocity_measurement.between(0,20)].copy()
fit('Measurement-date analysis, matched time covariate',measured,'height_velocity_measurement ~ '+terms.replace('followup_days','measurement_days'),bias=True)
counts=p.groupby('hospital_code').long_acting.nunique()
fe=p[p.hospital_code.isin(counts[counts==2].index)].copy()
formula='height_velocity_primary ~ '+nonlinear.replace(' + log_hospital_volume','').replace(' + C(region)','')+' + C(hospital_code)'
result=sm.OLS.from_formula(formula,fe).fit(cov_type='cluster',cov_kwds={'groups':fe.hospital_code},use_t=True)
ci=result.conf_int().loc['long_acting']; rows.append(dict(analysis='Hospital fixed effects plus nonlinear adjustment, cluster t',estimate=result.params.long_acting,se=result.bse.long_acting,ci_low=ci.iloc[0],ci_high=ci.iloc[1],n=int(result.nobs),hospitals=int(fe.hospital_code.nunique()),unit='cm/year'))
print(json.dumps(rows[-1]),flush=True)
sizes=p.groupby('hospital_code').size()
obs_hosp=set(p.hospital_code)
diag=dict(primary_n=len(p),hospital_size_median=float(sizes.median()),hospital_size_max=int(sizes.max()),hospitals_without_primary_outcome=int(b.loc[~b.hospital_code.isin(obs_hosp),'hospital_code'].nunique()),baseline_patients_in_those_hospitals=int((~b.hospital_code.isin(obs_hosp)).sum()),followup_days_by_group=p.groupby('long_acting').followup_days.agg(['mean','min','max']).to_dict(),excluded_within_window_by_group=d[d.window_primary & d.outcome_plausibility_query].groupby('long_acting').size().to_dict())
pd.DataFrame(rows).to_csv(m.TABLES/'growth_method_audit.csv',index=False)
(m.OUT/'growth_method_audit_diagnostics.json').write_text(json.dumps(diag,indent=2),encoding='utf-8')
print(json.dumps(diag),flush=True)
