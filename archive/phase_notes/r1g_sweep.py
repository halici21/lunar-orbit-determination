from __future__ import annotations
import contextlib, io, json, math, runpy, sys
from pathlib import Path
import numpy as np

ROOT=Path(r'C:/Users/erayh/Documents/Python/Grad')
FEATURE=ROOT/'python_port_phase17r0'
sys.path.insert(0,str(FEATURE))
SCRIPT=ROOT/'od_covariance_campaign/06_scripts/phase17r_final_synthesis.py'
with contextlib.redirect_stdout(io.StringIO()):
    d=runpy.run_path(str(SCRIPT))
C=d['C']; ET0=d['ET0']; T_ORBIT=d['T_ORBIT']; X0=d['X0']; earth_at=d['earth_at']; sun_at=d['sun_at']; TIGHT=d['TIGHT']; J2=d['J2_MOON_UNNORMALIZED']
from lunar_od.dynamics import propagate_state_with_k_sensitivity
from lunar_od.srp import SRPOptions
from lunar_od.two_way_range import generate_two_way_range_measurements, two_way_range_nominal_and_initial_jacobian
from lunar_od.estimators import _two_way_range_k_srp_column, _two_way_range_weight_diagonal, estimate_two_way_range_bls_lm
from lunar_od.visibility import VisibilityConfig, analyze_visibility_gap_with_transforms, sample_j2000_to_itrf93_transforms
KTRUTH=.01; KWRONG=.015; KINIT=.012; CAD=60.; RTOL=1e-12; ATOL=1e-13
OUT=[]
def make_case(arc, station_names):
    # +0.7 orbit holdout; deterministic grid and same truth.
    t_est=np.arange(0.,arc*T_ORBIT+0.5*CAD,CAD); t_hold=np.arange((arc*T_ORBIT)+CAD,(arc+.7)*T_ORBIT+0.5*CAD,CAD)
    t_all=np.concatenate([t_est,t_hold]); truth=propagate_state_with_k_sensitivity(t_all,X0,C.MU,0.,0.,earth_at,sun_at,srp=SRPOptions(k_srp_m2_per_kg=KTRUTH),rtol=RTOL,atol=ATOL,j2_moon=J2); x=truth[:,:6]
    xf=sample_j2000_to_itrf93_transforms(ET0,t_all); vc=VisibilityConfig(r_moon_mean_m=1737400.,earth_rotation_rad_s=7.292115e-5,epoch_utc=d['spice'].et2utc(ET0,'ISOC',3),min_elevation_deg=10.)
    _,_,vis,_=analyze_visibility_gap_with_transforms(t_all,x,C.STATIONS, C.get_earth_pos, xf,0.,vc); vis=np.asarray(vis,dtype=bool).reshape(t_all.size,len(C.STATIONS))
    idx=[i for i,s in enumerate(C.STATIONS) if s.name in station_names]; vm=vis[:,idx]; sts=tuple(C.STATIONS[i] for i in idx)
    obs,geo=generate_two_way_range_measurements(t_est,x[:len(t_est)],sts,vm[:len(t_est)],C.get_earth_pos,C.get_earth_vel,ET0,noise=False,rng=None,config=TIGHT)
    # Build data-only [x0,K] information.
    nom=truth[:len(t_est)]; _,Hx=two_way_range_nominal_and_initial_jacobian(obs,geo,nom[:,:42]); Hk=_two_way_range_k_srp_column(obs,nom,Hx); H=np.hstack([Hx,Hk[:,None]]); w=_two_way_range_weight_diagonal(obs,geo); scale=np.diag([1e6]*3+[1e3]*3+[KTRUTH]); hs=H@scale; info=hs.T@(w[:,None]*hs); sv=np.linalg.svd(info,compute_uv=False); v=np.linalg.svd(info)[2][-1]; raw=np.asarray(H).T@(w[:,None]*H); ixx=raw[:6,:6]; ix=raw[:6,6]; cond=float(raw[6,6]-ix@np.linalg.pinv(ixx)@ix)
    result={'arc_orbits':arc,'stations':','.join(station_names),'obs':int(obs.shape[0]),'rank':int(np.linalg.matrix_rank(info)),'sv':sv.tolist(),'cond_scaled':float(sv[0]/sv[-1]),'weak_k':float(v[-1]),'ikk_raw':float(raw[6,6]),'conditional_k_info':cond,'obs_data':obs,'geo':geo,'t_est':t_est,'t_hold':t_hold,'truth':truth,'sts':sts,'vis':vis,'H':H,'w':w}
    return result
def run_bls(c):
    xguess=c['truth'][0,:6]+np.array([5.,-3.,2.,0.,0.,0.]); pc=np.diag([10.**2]*3+[.01**2]*3)
    vals={}
    for label,k,solve in [('correct',KTRUTH,False),('wrong',KWRONG,False),('solve',KTRUTH,True)]:
        kw=dict(max_iter=35,rtol=RTOL,atol=ATOL,j2_moon=J2,srp=SRPOptions(k_srp_m2_per_kg=k),prior_covariance=pc,return_posterior=True)
        if solve: kw.update(solve_for_k_srp=True,k_srp_initial=KINIT,k_srp_prior_sigma=.005)
        x,stop,st=estimate_two_way_range_bls_lm(c['t_est'],c['obs_data'],xguess.copy(),c['geo'],C.MU,0.,0.,earth_at,sun_at,**kw)
        # Continue with fixed parameter for holdout prediction.
        pred=propagate_state_with_k_sensitivity(np.concatenate([c['t_est'],c['t_hold']]),x,C.MU,0.,0.,earth_at,sun_at,srp=SRPOptions(k_srp_m2_per_kg=(st.k_srp_estimate if solve else k)),rtol=RTOL,atol=ATOL,j2_moon=J2)[:,:6]
        true_hold=c['truth'][len(c['t_est']):,:6]; err=pred[len(c['t_est']):]-true_hold
        vals[label]={'k':float(st.k_srp_estimate if solve else k),'sigma_k':float(np.sqrt(st.posterior_covariance[6,6])) if solve and st.posterior_covariance.shape[0]>6 else None,'final_hold_m':float(np.linalg.norm(err[-1,:3])),'rms_hold_m':float(np.sqrt(np.mean(np.sum(err[:,:3]**2,axis=1)))),'stop':str(stop)}
    return vals
for arc,sts in [(1.3,['Canberra DSN']),(2.0,['Canberra DSN']),(3.0,['Canberra DSN']),(5.0,['Canberra DSN']),(2.0,['Goldstone DSN','Madrid DSN','Canberra DSN']),(3.0,['Goldstone DSN','Madrid DSN','Canberra DSN'])]:
    c=make_case(arc,sts); row={k:v for k,v in c.items() if k not in {'obs_data','geo','t_est','t_hold','truth','sts','vis','H','w'}}
    if arc in (1.3,2.0,3.0,5.0): row['bls']=run_bls(c)
    OUT.append(row); print(json.dumps(row,sort_keys=True))
(ROOT/'r1g_sweep_results.json').write_text(json.dumps(OUT,indent=2))
