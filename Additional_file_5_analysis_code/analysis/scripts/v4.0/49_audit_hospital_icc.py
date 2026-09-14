"""Independent maximum-likelihood logistic GLMM audit by adaptive quadrature."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import patsy
from scipy.special import roots_hermitenorm, expit, logsumexp
from scipy.optimize import minimize, brentq
from scipy.stats import chi2
from statsmodels.tools.numdiff import approx_hess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('original', HERE / '44_cpp_rhgh_integrated_analysis.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
v1 = m.prepare_v1(pd.read_excel(m.WORKBOOK, sheet_name='V1'))
v2 = m.prepare_v2(pd.read_excel(m.WORKBOOK, sheet_name='V2'))
b, _ = m.add_analysis_columns(v1, m.prepare_longitudinal(v1, v2))
cols = ['long_acting','hospital_code','age','male','height_std','bmi_std','bone_age_advancement','concurrent_gnrha','year_centered','log_hospital_volume','region']
b = b[cols].replace([np.inf,-np.inf],np.nan).dropna().sort_values('hospital_code')
formula = 'long_acting ~ age + male + height_std + bmi_std + bone_age_advancement + concurrent_gnrha + year_centered + log_hospital_volume + C(region)'
y, X = patsy.dmatrices(formula,b,return_type='dataframe')
# Drop unused categorical columns, then centre/scale continuous design for optimization.
X = X.loc[:,(X != 0).any()]
x = X.to_numpy(); y = y.to_numpy().ravel()
for j in range(1,x.shape[1]):
    x[:,j] = (x[:,j]-x[:,j].mean())/x[:,j].std()
g = pd.factorize(b.hospital_code)[0]; ng = int(g.max()+1)
starts = np.r_[0,np.flatnonzero(np.diff(g))+1]

def objective(theta, nodes, design):
    beta, logsd = theta[:-1], theta[-1]
    variance = np.exp(2*logsd)
    eta = design @ beta
    mode = np.zeros(ng)
    for _ in range(60):
        p = expit(eta+mode[g])
        score = np.add.reduceat(y-p,starts)-mode/variance
        precision = np.add.reduceat(p*(1-p),starts)+1/variance
        step = np.clip(score/precision,-3,3)
        mode += step
        if np.max(np.abs(step)) < 1e-11: break
    p = expit(eta+mode[g])
    precision = np.add.reduceat(p*(1-p),starts)+1/variance
    scale = 1/np.sqrt(precision)
    z,w = roots_hermitenorm(nodes)
    u = mode[:,None]+scale[:,None]*z
    e = eta[:,None]+u[g]
    ll = np.add.reduceat(y[:,None]*e-np.logaddexp(0,e),starts,axis=0)
    terms = ll-u*u/(2*variance)+z*z/2+np.log(w)[None,:]
    integrated = logsumexp(terms,axis=1)+np.log(scale)-logsd-.5*np.log(2*np.pi)
    return -integrated.sum()

out=[]
for label,design in [('Null',x[:,:1]),('Adjusted',x)]:
    theta = np.r_[np.zeros(design.shape[1]),.8]
    for nodes in [9,21,41]:
        result = minimize(objective,theta,args=(nodes,design),method='BFGS',options={'gtol':2e-5,'maxiter':600})
        theta=result.x
        gradient = float(np.max(np.abs(result.jac)))
        if gradient > .002: raise RuntimeError(str(result))
        hess=approx_hess(theta,lambda t: objective(t,nodes,design))
        eig=np.linalg.eigvalsh(hess)
        if eig.min() <= 0: raise RuntimeError('Nonpositive Hessian')
        se=np.sqrt(np.linalg.inv(hess)[-1,-1])
        transform=lambda a: float(np.exp(2*a)/(np.exp(2*a)+np.pi**2/3))
        row=dict(model=label,n=len(b),hospitals=ng,nodes=nodes,log_sd=float(theta[-1]),hospital_sd=float(np.exp(theta[-1])),icc=transform(theta[-1]),ci_low=transform(theta[-1]-1.96*se),ci_high=transform(theta[-1]+1.96*se),log_sd_se=float(se),mor=float(np.exp(.6744897501960817*np.sqrt(2)*np.exp(theta[-1]))),nll=float(result.fun),max_gradient=gradient,min_hessian_eigenvalue=float(eig.min()),interval='95% Wald confidence interval on log-SD transformed to ICC',method='Maximum likelihood, adaptive Gauss-Hermite quadrature')
        out.append(row)
        print(json.dumps(row),flush=True)
pd.DataFrame(out).to_csv(m.TABLES/'hospital_icc_independent_audit.csv',index=False)
best=theta.copy()
minimum=objective(best,41,x)
def profile(logsd):
    fit=minimize(lambda beta: objective(np.r_[beta,logsd],41,x),best[:-1],method='BFGS',options={'gtol':5e-5,'maxiter':600})
    if np.max(np.abs(fit.jac)) > .002: raise RuntimeError('Profile optimization failed')
    return 2*(fit.fun-minimum)-chi2.ppf(.95,1)
low=brentq(profile,best[-1]-.5,best[-1],xtol=1e-6)
high=brentq(profile,best[-1],best[-1]+.5,xtol=1e-6)
profile_result=dict(out[-1],ci_low=transform(low),ci_high=transform(high),interval='95% profile-likelihood confidence interval',proportional_variance_reduction=1-np.exp(2*(best[-1]-out[2]['log_sd'])))
(m.OUT/'hospital_icc_profile_audit.json').write_text(json.dumps(profile_result,indent=2),encoding='utf-8')
print(json.dumps(profile_result),flush=True)
