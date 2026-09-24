import contextlib,io,runpy,sys,numpy as np
sys.path.insert(0,r'C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0')
p=r'C:/Users/erayh/Documents/Python/Grad/od_covariance_campaign/06_scripts/phase17r_srukf_qualification.py'
with contextlib.redirect_stdout(io.StringIO()): d=runpy.run_path(p)
from lunar_od.radiometrics import two_way_counted_doppler_initial_state_jacobian
from lunar_od.filters import _range_rate_measurement_from_state
obs=d['obs']; pg=d['pass_geo']; xt=d['x_truth']; nom=d['nom']; mu=d['_MU_MOON']; epos=d['get_earth_pos']; spos=d['get_sun_pos']; dk=1e-4
Hs=[]; sig=[]
for row in obs:
    ti=int(row[6])-1
    J=two_way_counted_doppler_initial_state_jacobian(float(row[0]),pg.stations[int(row[5])-1],pg.t_s,nom[:,:42],pg.earth_pos_mci_m,pg.earth_vel_mci_mps,pg.x_j2000_to_itrf93,pg.range_rate_physics,et0_s=pg.et0_s)
    sp=xt[ti]+dk*nom[ti,42:48]; sm=xt[ti]-dk*nom[ti,42:48]
    hp=_range_rate_measurement_from_state(sp,row,pg,mu,0.,0.,epos,spos,1e-12,1e-13)[1]
    hm=_range_rate_measurement_from_state(sm,row,pg,mu,0.,0.,epos,spos,1e-12,1e-13)[1]
    Hs.append(np.r_[J,(hp-hm)/(2*dk)])
    sig.append(pg.stations[int(row[5])-1].sigma_range_rate_mps)
H=np.asarray(Hs); w=1/np.asarray(sig)**2; S=np.diag([1e6]*3+[1e3]*3+[.01]); HS=H@S; I=HS.T@(w[:,None]*HS); sv=np.linalg.svd(I,compute_uv=False); v=np.linalg.svd(I)[2][-1]; raw=H.T@(w[:,None]*H); ix=raw[:6,6]; cond=float(raw[6,6]-ix@np.linalg.pinv(raw[:6,:6])@ix)
print('obs',len(obs)); print('rank',np.linalg.matrix_rank(I)); print('sv',sv); print('cond',sv[0]/sv[-1]); print('weak_k',v[-1]); print('ikk',raw[6,6]); print('conditional',cond); print('h_k_range',np.min(np.abs(H[:,-1])),np.max(np.abs(H[:,-1])))
# Range + counted-Doppler information on the same synthetic 3-station arc.
from lunar_od.two_way_range import generate_two_way_range_measurements, TwoWayRangeConfig, two_way_range_nominal_and_initial_jacobian
from lunar_od.estimators import _two_way_range_k_srp_column, _two_way_range_weight_diagonal
from lunar_od.measurements import PassGeometry
from lunar_od.config import Station
sts=tuple(pg.stations)
t_pass=d['t_pass']
vis=np.ones((len(t_pass),len(sts)),dtype=bool)
zero_vel=lambda tt: np.zeros((np.atleast_1d(np.asarray(tt)).size,3))
robs,rgeo=generate_two_way_range_measurements(t_pass,xt,sts,vis,epos,zero_vel,pg.et0_s,noise=False,rng=None,config=TwoWayRangeConfig(tolerance_s=1e-13,equation_tolerance_s=1e-14,max_iter=200))
_,rhx=two_way_range_nominal_and_initial_jacobian(robs,rgeo,nom[:,:42]); rhk=_two_way_range_k_srp_column(robs,nom,rhx); Hr=np.hstack([rhx,rhk[:,None]]); wr=_two_way_range_weight_diagonal(robs,rgeo)
Hc=np.vstack([Hr,H]); wc=np.concatenate([wr,w]); Hcs=Hc@S; Ic=Hcs.T@(wc[:,None]*Hcs); svc=np.linalg.svd(Ic,compute_uv=False); vc=np.linalg.svd(Ic)[2][-1]; rawc=Hc.T@(wc[:,None]*Hc); ixc=rawc[:6,6]; cc=float(rawc[6,6]-ixc@np.linalg.pinv(rawc[:6,:6])@ixc)
print('combined_obs',len(robs),len(obs)); print('combined_rank',np.linalg.matrix_rank(Ic)); print('combined_sv',svc); print('combined_cond',svc[0]/svc[-1]); print('combined_weak_k',vc[-1]); print('combined_conditional',cc)
