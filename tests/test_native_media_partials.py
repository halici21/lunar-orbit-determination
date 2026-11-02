"""Independent finite differences of actual NIELL/BLACK_EISNER values."""
import math
import numpy as np
import pytest
from lunar_od.media import MediaModel,MediaSite
from lunar_od.spice_loader import load_spice_kernels,resolve_kernel_dir
from pathlib import Path


@pytest.fixture(scope='module')
def geometry():
    import spiceypy as spice
    load_spice_kernels();spice.furnsh(str(Path(resolve_kernel_dir())/'earth_latest_high_prec.bpc'))
    et=float(spice.str2et('2009-09-17T06:00:00'))
    site=MediaSite(math.radians(35.3),math.radians(-116.81),970.)
    rot=np.asarray(spice.pxform('ITRF93','J2000',et))
    north=np.array([-math.sin(site.lat_rad)*math.cos(site.lon_rad),
                    -math.sin(site.lat_rad)*math.sin(site.lon_rad),math.cos(site.lat_rad)])
    yield et,site,rot,north
    load_spice_kernels()


@pytest.mark.parametrize('mapping',['NIELL','BLACK_EISNER'])
@pytest.mark.parametrize('vtec',[None,12.,'DEFAULT_CLIMATOLOGY'])
@pytest.mark.parametrize('uplink',[False,True])
@pytest.mark.parametrize('elevation',[5.,30.,90.])
def test_native_partial_matches_two_fd_scales(geometry,mapping,vtec,uplink,elevation):
    et,site,rot,north=geometry;el=math.radians(elevation)
    los=rot@(site.up_ecef*math.sin(el)+north*math.cos(el))*3.84e8
    model=MediaModel(mapping=mapping,vtec_tecu=vtec,uplink_hz=2.09e9,downlink_hz=2.271125e9)
    analytic=model.leg_delay_partials(site,et,los,uplink)
    for h in [100.,1000.]:
        numerical=np.zeros((2,3))
        for j in range(3):
            dx=np.zeros(3);dx[j]=h
            numerical[:,j]=(np.asarray(model.leg_delays(site,et,los+dx,uplink)[:2])
                            -np.asarray(model.leg_delays(site,et,los-dx,uplink)[:2]))/(2*h)
        np.testing.assert_allclose([analytic.group_los_gradient,analytic.phase_los_gradient],
                                   numerical,rtol=2e-6,atol=2e-13)
    for h in [.1,1.]:
        numerical=(np.asarray(model.leg_delays(site,et+h,los,uplink)[:2])
                   -np.asarray(model.leg_delays(site,et-h,los,uplink)[:2]))/((et+h)-(et-h))
        np.testing.assert_allclose([analytic.group_time_rate,analytic.phase_time_rate],numerical,
                                   rtol=2e-6,atol=5e-8)
