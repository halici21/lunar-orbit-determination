"""Contact completeness cannot depend on the numerical origin of time."""
import numpy as np
import pytest
from tests import test_shadow_grazing_regression as G


@pytest.mark.parametrize('origin',[0.,1e6,1e8,1e9])
def test_short_contact_pair_survives_epoch_translation(origin):
    sun,q=G.geometry(1.)
    span=np.array([G.MID-180,G.MID+180])
    reference=np.array([G.brentq(q,G.MID-100,G.MID),G.brentq(q,G.MID,G.MID+100)])
    contacts=G.shadow_contact_times(span+origin,G.state(span[0]),G.MU_MOON_M3S2,0.,0.,
        lambda t:np.array([384400e3,0,0]),lambda t:sun,
        srp=G.SRPOptions(k_srp_m2_per_kg=0.),rtol=1e-12,atol=1e-13)
    assert len(contacts)==2
    np.testing.assert_allclose(np.asarray(contacts)-origin,reference,rtol=0,atol=1e-3)
