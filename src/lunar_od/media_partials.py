"""Derivatives of the existing media value model, in metres and SI seconds.

LOS derivatives include its normalisation. Explicit epoch derivatives hold
the inertial LOS fixed; event-motion terms belong to the light-time chain.
No atmosphere, calibration or ionosphere model is added here.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class MediaDelayPartials:
    group_los_gradient: np.ndarray
    phase_los_gradient: np.ndarray
    group_time_rate: float
    phase_time_rate: float
    method: str = "analytic_media_model"


def _fraction_partials(a, b, c, q):
    """Niell continued fraction: value, d/d sin(el), d/d[a,b,c]."""
    n1=1.+c; n2=1.+b/n1; num=1.+a/n2
    d1=q+c; d2=q+b/d1; den=q+a/d2
    dn=np.array([1./n2,-a/(n2*n2*n1),a*b/(n2*n2*n1*n1)])
    dd=np.array([1./d2,-a/(d2*d2*d1),a*b/(d2*d2*d1*d1)])
    dq=1.-a/(d2*d2)*(1.-b/(d1*d1))
    return num/den, -num*dq/(den*den), (dn*den-num*dd)/(den*den)


def model_partials(model, site, et, los, uplink):
    from . import media as M
    import spiceypy as spice
    v=np.asarray(los,dtype=float).reshape(3);n=float(np.linalg.norm(v))
    if not np.isfinite(n) or n<=0.:raise ValueError("Media derivatives require a finite nonzero LOS.")
    transform=np.asarray(spice.sxform("ITRF93","J2000",float(et)))
    up=transform[:3,:3]@site.up_ecef;up_rate=transform[3:6,:3]@site.up_ecef
    unit=v/n;q=float(np.clip(unit@up,-1.,1.))
    dq_los=(up-q*unit)/n;dq_time=float(unit@up_rate)
    trop_q=0.;trop_explicit=0.
    if model.troposphere:
        zhd=model.zhd_m if model.zhd_m is not None else M.zhd_iers(M.standard_pressure_hpa(site.h_m),site.lat_rad,site.h_m)
        if model.mapping=="BLACK_EISNER":
            mq=-1.001*q/(.002001+q*q)**1.5
            trop_q=(zhd+model.zwd_m)*mq
        else:
            lat=min(max(abs(math.degrees(site.lat_rad)),15.),75.)
            doy=model.day_of_year(et);dmin=M._DOY_MIN+(183. if site.lat_rad<0 else 0.)
            theta=2*math.pi*(doy-dmin)/M._YEAR_D
            avg=[float(np.interp(lat,M._LAT_DEG,a)) for a in (M._AH_AVG,M._BH_AVG,M._CH_AVG)]
            amp=np.array([float(np.interp(lat,M._LAT_DEG,a)) for a in (M._AH_AMP,M._BH_AMP,M._CH_AMP)])
            a,b,c=np.asarray(avg)-amp*math.cos(theta)
            _,mhq,mhc=_fraction_partials(a,b,c,q)
            _,htq,_=_fraction_partials(*M._HT,q)
            mhq+=(-1./(q*q)-htq)*max(0.,site.h_m)/1000.
            wet=[float(np.interp(lat,M._LAT_DEG,a)) for a in (M._AW,M._BW,M._CW)]
            _,mwq,_=_fraction_partials(*wet,q)
            trop_q=zhd*mhq+model.zwd_m*mwq
            # et2utc's full time relation and the same day-of-year value are
            # used. Shrink only to stay on one smooth calendar branch.
            h=10.
            for _ in range(24):
                plus=model.day_of_year(et+h);minus=model.day_of_year(et-h)
                if abs(plus-minus)<1.:break
                h*=.5
            else:raise ValueError("Media derivative crosses a non-differentiable calendar boundary.")
            doy_rate=(plus-minus)/((et+h)-(et-h))
            coefficient_rate=amp*math.sin(theta)*(2*math.pi/M._YEAR_D)*doy_rate
            trop_explicit=zhd*float(mhc@coefficient_rate)
    ion_q=0.;ion_explicit=0.
    if model.vtec_tecu is not None:
        if model.vtec_tecu=="DEFAULT_CLIMATOLOGY":
            lt=((float(et)-69.184+43200.)/3600.+math.degrees(site.lon_rad)/15.)%24.
            theta=2*math.pi*(lt-14.)/24.;cosine=math.cos(theta)
            if abs(cosine)<64*np.finfo(float).eps:
                raise ValueError("Media climatology is not differentiable at its day/night cusp.")
            vtec=M.default_vtec_tecu(lt)
            rate=-25.*math.sin(theta)*2*math.pi/86400. if cosine>0 else 0.
        else:vtec=float(model.vtec_tecu);rate=0.
        alpha=model.earth_radius_m/(model.earth_radius_m+model.layer_height_m)
        mapping=(1.-alpha*alpha*(1.-q*q))**-.5
        mapping_q=-alpha*alpha*q*mapping**3
        factor=M.IONO_K/(model.uplink_hz if uplink else model.downlink_hz)**2
        ion_q=factor*vtec*mapping_q;ion_explicit=factor*rate*mapping
    group_q=trop_q+ion_q;phase_q=trop_q-ion_q
    return MediaDelayPartials(group_q*dq_los,phase_q*dq_los,
        group_q*dq_time+trop_explicit+ion_explicit,
        phase_q*dq_time+trop_explicit-ion_explicit)


def leg_partials(provider, site, et, los, uplink):
    """Native analytic derivatives, or a same-function provider reference.

    External legacy providers exposing only leg_delays remain supported. Their
    reference derivatives difference the local delay, never a large range or
    a propagated orbit. The method is explicit in the returned provenance.
    """
    native=getattr(provider,"leg_delay_partials",None)
    if native is not None:return native(site,et,los,uplink)
    los=np.asarray(los,float);h=max(1.,math.sqrt(np.finfo(float).eps)*float(np.linalg.norm(los)))
    gradients=np.zeros((2,3))
    for j in range(3):
        dx=np.zeros(3);dx[j]=h
        plus=np.asarray(provider.leg_delays(site,et,los+dx,uplink)[:2])
        minus=np.asarray(provider.leg_delays(site,et,los-dx,uplink)[:2])
        gradients[:,j]=(plus-minus)/(2*h)
    h=max(.1,8*abs(float(np.spacing(et))))
    plus=np.asarray(provider.leg_delays(site,et+h,los,uplink)[:2])
    minus=np.asarray(provider.leg_delays(site,et-h,los,uplink)[:2])
    rates=(plus-minus)/((et+h)-(et-h))
    return MediaDelayPartials(gradients[0],gradients[1],float(rates[0]),float(rates[1]),
                              method="external_provider_local_finite_difference")
