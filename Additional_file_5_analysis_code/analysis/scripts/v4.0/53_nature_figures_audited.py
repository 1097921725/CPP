"""Publication figures from audited aggregate results; Python-only rendering."""
from pathlib import Path
import json
import hashlib
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parents[3]
PORTABLE=(Path(__file__).resolve().parent/'source_data').is_dir()
DATA=Path(__file__).resolve().parent if PORTABLE else ROOT/'analysis/output/v4.0/cpp_rhgh_integrated_20260910'
OUT=DATA if PORTABLE else DATA/'nature_redesign_audited_20260911'
SRC=OUT/'source_data'; SRC.mkdir(parents=True,exist_ok=True)
BLUE='#24658B'; CORAL='#CB624D'; TEAL='#519D9A'; INK='#233647'; GREY='#8B97A3'; LIGHT='#EAF0F4'; GOLD='#B38845'
COLORS=[CORAL,BLUE,TEAL]
FORMS=['Long-acting cartridge','Short-acting aqueous','Short-acting powder']
SHORT=['Long-acting','Short-acting\naqueous','Short-acting\npowder']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7.2,'axes.labelsize':7.5,'xtick.labelsize':6.8,'ytick.labelsize':7,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'lines.linewidth':1.1,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False,'savefig.facecolor':'white','axes.facecolor':'white','figure.facecolor':'white','svg.hashsalt':'cpp-audited-20260911'})
rng=np.random.default_rng(20260911)
paths=[]; qa=[]
def load(name,folder='tables'):
    path=SRC/name if PORTABLE else DATA/folder/name
    if path.resolve() != (SRC/name).resolve(): shutil.copy2(path,SRC/name)
    return pd.read_csv(path)
flow=load('table0_cohort_flow.csv')
hosp=load('hospital_formulation_rates.csv','figure_source_data')
rates=load('followup_rates_by_formulation.csv').set_index('formulation').loc[FORMS]
raw=load('growth_response_distribution.csv','figure_source_data')
audit=load('growth_method_audit.csv'); original=load('table7_sensitivity_analyses.csv')
selection=load('table2_formulation_selection_gee.csv').set_index('term')
baseline=load('table1_v1_baseline_characteristics.csv')
follow=load('table3_followup_selection.csv')
icctab=load('hospital_icc_independent_audit.csv')
jsonbase=SRC if PORTABLE else DATA
icc=json.loads((jsonbase/'hospital_icc_profile_audit.json').read_text())
summary=json.loads((jsonbase/'analysis_summary.json').read_text())
if not PORTABLE:
    for name in ['hospital_icc_profile_audit.json','analysis_summary.json']: shutil.copy2(DATA/name,SRC/name)
assert len(raw)==735 and hosp.patients.sum()==3606 and hosp.long_acting_n.sum()==1490
assert abs(icc['icc']-.5269662765)<1e-8

def fig(height): return plt.figure(figsize=(170/25.4,height/25.4))
def panel(ax,letter,label=None):
    box=ax.get_position()
    ax.figure.text(box.x0-.035,box.y1+.055*box.height,letter,fontsize=10,fontweight='bold',va='bottom')
    if label: ax.text(0,1.06,label,transform=ax.transAxes,fontsize=8.2,fontweight='bold',va='bottom')
def clean(ax,grid='y'):
    ax.set_axisbelow(True); ax.grid(axis=grid,color=LIGHT,lw=.55)
    ax.tick_params(length=3,width=.65)
def save(f,name):
    f.canvas.draw()
    # Final-size text and file checks; visual inspection remains required.
    sizes=[t.get_fontsize() for t in f.findobj(matplotlib.text.Text) if t.get_text() and t.get_visible()]
    qa.append({'figure':name,'width_mm':170,'height_mm':round(f.get_figheight()*25.4,1),'min_text_pt':min(sizes)})
    for ext in ['pdf','svg','png','tiff']:
        opts={'dpi':600 if ext=='tiff' else 220}
        if ext=='tiff': opts['pil_kwargs']={'compression':'tiff_lzw'}
        path=OUT/f'{name}.{ext}'; f.savefig(path,**opts)
        if path.stat().st_size>=10_000_000: raise ValueError(f'File exceeds 10 MB: {name}.{ext}')
    pdfbook.savefig(f); paths.append(OUT/f'{name}.png'); plt.close(f)

def forest(ax,rows,labels,colors=None,xlim=(-2.1,.4),xlabel='Adjusted difference in annualized height velocity (cm/year)',numbers=True):
    n=len(rows); ax.set_ylim(n-.45,-.8); ax.set_xlim(*xlim)
    ax.axvline(0,color=GREY,ls=(0,(3,3)),lw=.8)
    for i,(_,r) in enumerate(rows.iterrows()):
        c=colors[i] if colors else BLUE
        if i%2==0: ax.axhspan(i-.42,i+.42,color=LIGHT,alpha=.6,zorder=0)
        ax.plot([r.ci_low,r.ci_high],[i,i],color=c,lw=1.7,solid_capstyle='round')
        ax.scatter(r.estimate,i,s=32,color=c,edgecolors='white',linewidths=.55,zorder=3)
        if numbers: ax.text(1.035,i,f'{r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]',transform=ax.get_yaxis_transform(),va='center',fontsize=6.7)
    ax.set_yticks(range(n),labels); ax.tick_params(axis='y',length=0,pad=7)
    ax.spines['left'].set_visible(False); ax.spines['bottom'].set_color(GREY)
    ax.set_xlabel(xlabel,labelpad=7)
    if numbers: ax.text(1.035,1.02,'Estimate [95% CI]',transform=ax.transAxes,fontsize=6.7,color=GREY)

with PdfPages(OUT/'All_Figures.pdf') as pdfbook:
    # Figure 1: a compact, auditable flow with real denominators.
    f=fig(125); a=f.add_axes([.07,.17,.56,.74]); a.axis('off'); panel(a,'a','Cohort construction')
    blocks=[('V1 records','8,907'),('Explicit CPP; eligible baseline','3,606'),('Linked V2 assessment','770'),('120–240-day interval','749'),('Analyzable growth outcome','735')]
    ys=[.88,.68,.48,.28,.08]
    for i,((label,n),y) in enumerate(zip(blocks,ys)):
        a.add_patch(FancyBboxPatch((.03,y-.07),.58,.145,boxstyle='round,pad=.008,rounding_size=.016',fc=BLUE if i in [1,4] else LIGHT,ec='none'))
        a.text(.06,y+.022,label,fontsize=6.8,color='white' if i in [1,4] else INK,va='center')
        a.text(.06,y-.029,n,fontsize=15,fontweight='bold',color='white' if i in [1,4] else BLUE,va='center')
        if i<4: a.annotate('',xy=(.31,ys[i+1]+.082),xytext=(.31,y-.08),arrowprops={'arrowstyle':'->','color':GREY,'lw':.9})
    for y,text in [(.77,'5,299 other diagnoses\n2 outside age range'),(.57,'2,836 without\nlinked V2'),(.37,'21 outside\ntime window'),(.17,'14 unresolved\nvelocity flags')]:
        a.plot([.31,.65],[y,y],color=GREY,lw=.65); a.text(.67,y,text,va='center',fontsize=6.4)
    a.text(.03,-.075,'V2 source: 1,685 records; 770 linked to the eligible V1 cohort.',fontsize=6.1,color=GREY)
    b=f.add_axes([.76,.21,.21,.65]); panel(b,'b','Formulation mix')
    for x,col in enumerate(['baseline_n','observed_primary_n']):
        vals=rates[col].to_numpy(); bottom=0
        for c,n in zip(COLORS,vals):
            h=n/vals.sum()*100; b.bar(x,h,bottom=bottom,width=.6,color=c,edgecolor='white',lw=.7); b.text(x,bottom+h/2,f'{h:.1f}%',ha='center',va='center',color='white',fontsize=6.6,fontweight='bold'); bottom+=h
        b.text(x,104,f'n={vals.sum():,}',ha='center',fontsize=6.5)
    b.set_ylim(0,112); b.set_yticks([0,50,100]); b.set_ylabel('Patients (%)'); b.set_xticks([0,1],['V1','Follow-up']); clean(b)
    for i,(label,c) in enumerate(zip(['Long-acting','Short-acting aqueous','Short-acting powder'],COLORS)):
        f.text(.10+i*.29,.035,'●  '+label,color=c,fontsize=7)
    save(f,'Figure1_cohort_and_composition')

    # Figure 2: observed hospital variation is the dominant panel.
    f=fig(150); a=f.add_axes([.10,.49,.85,.43]); panel(a,'a','Hospital-level long-acting use')
    h=hosp.sort_values(['long_acting_percent','patients']).reset_index(drop=True)
    x=np.arange(1,len(h)+1); pct=h.long_acting_percent.to_numpy()
    a.vlines(x,0,pct,color=BLUE,alpha=.20,lw=.6)
    a.scatter(x,pct,s=9+h.patients.to_numpy()*.23,c=pct,cmap=matplotlib.colors.LinearSegmentedColormap.from_list('hospital',[BLUE,TEAL,CORAL]),edgecolors='white',lw=.35,zorder=3)
    a.axhline(1490/3606*100,color=INK,lw=.9,ls=(0,(3,3))); a.text(4,45,'Overall 41.3%',fontsize=6.8,bbox={'facecolor':'white','edgecolor':'none','pad':1})
    a.set(xlim=(0,220),ylim=(-3,107),xlabel='Hospitals ordered by observed long-acting proportion',ylabel='Long-acting use (%)'); a.set_yticks([0,25,50,75,100]); clean(a)
    for n in [10,100,250]: a.scatter([],[],s=9+n*.23,color=GREY,label=str(n))
    a.legend(title='Patients per hospital',loc='upper left',bbox_to_anchor=(.025,1.03),ncol=3,fontsize=6,title_fontsize=6.2,handletextpad=.3,columnspacing=.8)
    b=f.add_axes([.10,.12,.32,.24]); panel(b,'b','Hospitals with ≥20 patients')
    hh=h[h.patients>=20]; b.hist(hh.long_acting_percent,bins=np.linspace(0,100,11),color=BLUE,edgecolor='white',lw=.8)
    b.set(xlim=(0,100),xlabel='Long-acting use (%)',ylabel='Hospitals'); b.text(.95,.9,f'n={len(hh)} hospitals',transform=b.transAxes,ha='right',fontsize=6.5); clean(b)
    c=f.add_axes([.65,.12,.29,.24]); panel(c,'c','Latent-scale hospital ICC')
    null=icctab[(icctab.model=='Null')&(icctab.nodes==41)].iloc[0]
    for y,r,color,label in [(1,null,GREY,'Null'),(0,icc,CORAL,'Adjusted')]:
        c.plot([r['ci_low'],r['ci_high']],[y,y],color=color,lw=2); c.scatter(r['icc'],y,s=42,color=color,zorder=3); c.text(r['icc'],y+.22,f"{r['icc']:.3f}",ha='center',color=color,fontweight='bold',fontsize=8)
    c.set(xlim=(0,1),ylim=(-.55,1.7),xlabel='ICC (95% interval)'); c.set_yticks([0,1],['Adjusted','Null']); c.spines['left'].set_visible(False); c.tick_params(axis='y',length=0); c.text(.98,.05,'Adjusted MOR 6.21',transform=c.transAxes,ha='right',fontsize=7,color=CORAL); clean(c,'x')
    save(f,'Figure2_hospital_variation')

    # Figure 3: selection correlates alongside measured baseline imbalance.
    f=fig(145); a=f.add_axes([.29,.16,.29,.73]); panel(a,'a','Formulation correlates')
    terms=['age','male','height_std','bmi_std','bone_age_advancement','concurrent_gnrha','year_centered','log_hospital_volume','C(region)[T.Central]','C(region)[T.North]','C(region)[T.Northeast]','C(region)[T.South]','C(region)[T.West]']
    labels=['Age (per year)','Male sex','Height (per SD)','BMI (per SD)','Bone-age advancement','Concurrent GnRHa','Calendar year','Log hospital sample size','Central vs East','North vs East','Northeast vs East','South vs East','West vs East']
    a.axvline(1,color=GREY,ls='--',lw=.8)
    for i,(term,label) in enumerate(zip(terms,labels)):
        r=selection.loc[term]; c=CORAL if term=='year_centered' else BLUE
        if i%2==0:a.axhspan(i-.4,i+.4,color=LIGHT,alpha=.65)
        a.plot([r.or_ci_low,r.or_ci_high],[i,i],color=c,lw=1.4); a.scatter(r.odds_ratio,i,s=22,color=c,zorder=3)
    a.set_xscale('log'); a.set_xlim(.25,16); a.set_xticks([.25,.5,1,2,4,8,16],['0.25','0.5','1','2','4','8','16']); a.set_ylim(12.7,-.7); a.set_yticks(range(13),labels); a.set_xlabel('Adjusted odds ratio (95% CI)'); a.tick_params(axis='y',length=0); a.spines['left'].set_visible(False)
    b=f.add_axes([.74,.25,.21,.58]); panel(b,'b','Baseline imbalance')
    bb=baseline.iloc[::-1]; labels2=['Hospital size','Visit year','GnRHa','BA − CA','BMI','Height','Male','Age']
    for i,v in enumerate(bb.SMD): b.plot([0,v],[i,i],color=GREY,lw=1); b.scatter(v,i,s=31,color=CORAL if abs(v)>=.5 else BLUE,zorder=3)
    b.axvline(0,color=INK,lw=.7); b.axvspan(-.1,.1,color=LIGHT); b.set(xlim=(-1,1),ylim=(-.7,7.7),xlabel='Signed SMD'); b.set_yticks(range(8),labels2); b.tick_params(axis='y',length=0); b.spines['left'].set_visible(False)
    f.text(.30,.055,'Positive SMD: higher in long-acting users.  Shading: |SMD| <0.10.',fontsize=6.4,color=GREY)
    save(f,'Figure3_formulation_selection')

    # Figure 4: true distributions, no synthetic means or pseudo-trajectories.
    f=fig(155); a=f.add_axes([.12,.49,.82,.43]); panel(a,'a','Observed annualized height velocity')
    for i,(form,c) in enumerate(zip(FORMS,COLORS)):
        vals=raw.loc[raw.formulation==form,'height_velocity_primary'].to_numpy(); yy=np.linspace(vals.min(),vals.max(),220); density=gaussian_kde(vals)(yy); width=density/density.max()*.30
        a.fill_betweenx(yy,i+.04,i+.04+width,color=c,alpha=.32,lw=0)
        xx=i-rng.uniform(.05,.27,len(vals)); a.scatter(xx,vals,s=5,color=c,alpha=.40,lw=0,rasterized=True)
        q1,med,q3=np.quantile(vals,[.25,.5,.75]); a.plot([i,i],[q1,q3],color=c,lw=4,solid_capstyle='round'); a.scatter(i,med,s=19,c='white',edgecolors=c,lw=1,zorder=5)
        a.text(i,20.8,f'n={len(vals)}',ha='center',fontsize=7,color=c)
    a.set(xlim=(-.5,2.6),ylim=(-.35,22),ylabel='Height velocity (cm/year)'); a.set_xticks(range(3),SHORT); a.set_yticks([0,5,10,15,20]); clean(a)
    b=f.add_axes([.30,.13,.27,.22]); panel(b,'b','Annualized contrast')
    forest(b,audit.iloc[:2],['Original GEE','Finite-sample\ncorrection'],[GREY,CORAL],xlim=(-1.9,.15),numbers=False,xlabel='Adjusted difference (cm/year)')
    c=f.add_axes([.76,.13,.20,.22]); panel(c,'c','Day-180 height')
    r=audit.iloc[4]; c.axvline(0,color=GREY,ls='--',lw=.8); c.plot([r.ci_low,r.ci_high],[0,0],color=CORAL,lw=2); c.scatter(r.estimate,0,s=45,color=CORAL,zorder=3)
    c.text(.5,.83,f'{r.estimate:.3f} cm',transform=c.transAxes,ha='center',fontsize=9,fontweight='bold',color=CORAL); c.set(xlim=(-1,.2),ylim=(-.6,.6),xlabel='Adjusted difference (cm)'); c.set_yticks([]); c.spines['left'].set_visible(False); c.set_xticks([-1,-.5,0])
    f.text(.12,.035,'Negative contrasts indicate lower observed growth in long-acting users; associations are not causal.',fontsize=6.2,color=GREY)
    save(f,'Figure4_growth_distribution_and_effects')

    # Figure 5: robustness and the selection boundary belong on the same page.
    f=fig(165); a=f.add_axes([.30,.57,.39,.33]); panel(a,'a','Methodological checks')
    rows=audit.iloc[[1,2,3,5,6]]
    forest(a,rows,['Finite-sample correction','Independent correlation','Nonlinear adjustment','Measurement-date aligned','Within-hospital + nonlinear'],[CORAL,BLUE,BLUE,BLUE,TEAL])
    b=f.add_axes([.21,.13,.27,.27]); panel(b,'b','Outcome ascertainment')
    for i,(form,c) in enumerate(zip(FORMS,COLORS)):
        r=rates.loc[form]; v=r.observed_primary_percent
        b.barh(i,100,color=LIGHT,height=.52); b.barh(i,v,color=c,height=.52); b.text(v+2,i,f'{v:.1f}%',va='center',fontsize=6.7,color=c,fontweight='bold')
    b.set(xlim=(0,100),ylim=(2.6,-.65),xlabel='Baseline patients (%)'); b.set_yticks(range(3),SHORT); b.tick_params(axis='y',length=0); b.spines['left'].set_visible(False); b.set_xticks([0,50,100])
    c=f.add_axes([.76,.13,.20,.27]); panel(c,'c','Follow-up selection')
    vals=follow.SMD.to_numpy(); lab=['Long-acting','Age','Male','Height','BMI','BA − CA','GnRHa','Visit year']
    for i,v in enumerate(vals): c.plot([0,v],[i,i],color=GREY,lw=.7); c.scatter(v,i,s=22,color=CORAL if abs(v)>=.4 else BLUE,zorder=3)
    c.axvline(0,color=INK,lw=.7); c.axvspan(-.1,.1,color=LIGHT); c.set(xlim=(-1,.6),ylim=(7.6,-.6),xlabel='Signed SMD'); c.set_yticks(range(8),lab); c.tick_params(axis='y',length=0); c.spines['left'].set_visible(False)
    f.text(.12,.025,'140 of 217 hospitals contributed no analyzable primary outcome (1,042 baseline patients).',fontsize=6.5,color=INK)
    save(f,'Figure5_robustness_and_followup_selection')

    f=fig(105); a=f.add_axes([.36,.20,.33,.65]); panel(a,'a','Original sensitivity analyses')
    forest(a,original,['Original GEE','150–210-day window','Stable formulation + adherence','Validated measurement dates','Treatment overlap weights','Follow-up weights','Truncated follow-up weights','Hospital fixed effects'],[CORAL]+[BLUE]*6+[TEAL],xlim=(-1.9,.15))
    f.text(.12,.045,'Original intervals; these analyses target different populations. Updated inference appears in Figure 5.',fontsize=6.1,color=GREY)
    save(f,'SupplementaryFigureS1_original_sensitivity')
    ps=load('treatment_overlap_distribution.csv','figure_source_data'); bal=load('treatment_overlap_balance.csv')
    f=fig(105); a=f.add_axes([.11,.21,.34,.61]); panel(a,'a','Treatment overlap')
    for code,c,label in [(0,BLUE,'Short-acting'),(1,CORAL,'Long-acting')]:
        vals=ps.loc[ps.long_acting==code,'treatment_ps']; a.hist(vals,bins=np.linspace(0,1,21),density=True,histtype='stepfilled',alpha=.23,color=c,label=label); a.hist(vals,bins=np.linspace(0,1,21),density=True,histtype='step',color=c,lw=1)
    a.set(xlim=(0,1),xlabel='Estimated probability of long-acting use',ylabel='Density'); a.legend(fontsize=6.5); clean(a)
    b=f.add_axes([.69,.21,.26,.61]); panel(b,'b','Covariate balance')
    for i,r in bal.iterrows():
        before=abs(r.smd_unweighted); after=abs(r.smd_overlap_weighted); b.plot([before,after],[i,i],color=GREY,lw=.8); b.scatter(before,i,color=GREY,s=21); b.scatter(after,i,color=TEAL,s=25,zorder=3)
    b.axvline(.1,color=INK,ls='--',lw=.7); b.set(xlim=(0,1),ylim=(len(bal)-.5,-.6),xlabel='Absolute SMD'); b.set_yticks(range(len(bal)),['Age','Male','Height','BMI','BA − CA','GnRHa','Visit year','Hospital size']); b.tick_params(axis='y',length=0); b.spines['left'].set_visible(False)
    f.text(.67,.09,'● Before',color=GREY,fontsize=7); f.text(.83,.09,'● Weighted',color=TEAL,fontsize=7)
    save(f,'SupplementaryFigureS2_overlap_diagnostics')

# A preview board generated entirely in Python, separate from submission images.
thumbs=[]
for path in paths:
    im=Image.open(path).convert('RGB'); im.thumbnail((800,800))
    card=Image.new('RGB',(830,850),'white'); card.paste(im,((830-im.width)//2,35)); ImageDraw.Draw(card).text((16,12),path.stem,fill=INK); thumbs.append(card)
board=Image.new('RGB',(1660,850*((len(thumbs)+1)//2)),LIGHT)
for i,im in enumerate(thumbs):board.paste(im,((i%2)*830,(i//2)*850))
board.save(OUT/'Figure_overview.png')
pd.DataFrame(qa).to_csv(OUT/'export_QA.csv',index=False)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.iterdir()}
(OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
if Path(__file__).resolve() != (OUT/'reproduce_figures.py').resolve(): shutil.copy2(Path(__file__),OUT/'reproduce_figures.py')
print('Completed:',len(paths),'figures. Output:',OUT)
