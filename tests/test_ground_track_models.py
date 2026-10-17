"""Ground-track labels must distinguish uncomputed visibility from measured masks."""
from unittest.mock import patch
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSettings
from controllers import ground_track_controller as gt


def samples(mask=None):
    return dict(lons=np.array([10.,11.,12.,13.]), lats=np.array([-30.,-10.,10.,30.]),
                t_s=np.array([0.,240.,480.,720.]),
                vis_mask=np.zeros((0,4),dtype=bool) if mask is None else np.asarray([mask],dtype=bool),
                station_names=[] if mask is None else ['Madrid DSN'], duration_days=1.,
                epoch_utc='Keplerian (analytical)' if mask is None else '2027-03-02 23:32:37')


def test_analytical_plot_does_not_present_uncomputed_visibility_as_zero():
    app=QApplication.instance() or QApplication([])
    data=samples()
    originals={key:value.tobytes() for key,value in data.items() if isinstance(value,np.ndarray)}
    figure=gt._build_static_figure(data)
    ax=figure.axes[0]
    assert 'not computed' in ax.get_title()
    assert '0%' not in ax.get_title()
    assert 'No coverage' not in [label.get_text() for label in ax.get_legend().get_texts()]
    assert 'Visibility not computed' in [label.get_text() for label in ax.get_legend().get_texts()]
    assert originals=={key:data[key].tobytes() for key in originals}


@pytest.mark.parametrize('mask,percent', [([False]*4, '0%'), ([True,False,True,False], '50%')])
def test_sampled_spice_plot_reports_sample_percentage_including_real_zero(mask,percent):
    app=QApplication.instance() or QApplication([])
    data=samples(mask)
    figure=gt._build_static_figure(data); ax=figure.axes[0]
    assert f'Visible samples: {percent}' in ax.get_title()
    labels=[label.get_text() for label in ax.get_legend().get_texts()]
    assert 'No sample visibility' in labels and 'Visibility not computed' not in labels
    assert np.array_equal(data['vis_mask'],[mask])


@pytest.fixture
def page(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    settings=QSettings(str(tmp_path/'ground-track.ini'),QSettings.IniFormat)
    monkeypatch.setattr('PyQt5.QtCore.QSettings',lambda *a,**kw:settings)
    widget=gt.GroundTrackController()
    widget._orbit_designer._preview_timer.stop()
    widget.resize(1250,850); widget.show(); app.processEvents()
    yield widget,app
    widget._anim_timer.stop(); widget._orbit_designer._preview_timer.stop()
    widget.close(); app.processEvents()


def test_designer_result_and_playback_identify_missing_epoch_and_visibility(page,tmp_path):
    widget,app=page
    designer=widget._orbit_designer
    designer._dur_spin.setValue(.1)
    designer._generate_track()
    note=widget._model_note.text()
    assert 'iki cisim' in note and 'UTC epoku yok' in note
    assert 'sabit hızlı' in note and 'hesaplanmadı' in note
    assert widget._tab_widget.currentIndex()==0
    assert widget._data['vis_mask'].shape[0]==0
    assert '0.1 days' in widget._fig.axes[0].get_title()
    widget._anim_step()
    assert 'not computed' in widget._anim_label.get_text()
    assert 'no coverage' not in widget._anim_label.get_text()
    target=tmp_path/'analytical.csv'
    with patch.object(gt.QFileDialog,'getSaveFileName',return_value=(str(target),'CSV')):
        widget._export_csv()
    assert target.read_text().splitlines()[0]=='time_s,time_days,lon_deg,lat_deg,network_visible'
    assert widget._data['epoch_utc']=='Keplerian (analytical)'


def test_spice_result_note_describes_actual_example_epoch_and_sample_scope(page):
    widget,app=page
    data=samples([False]*4)
    widget._on_data(data)
    note=widget._model_note.text()
    assert 'SPICE' in note and 'örnek başlangıç' in note
    assert data['epoch_utc'] in note and 'UTC' in note
    assert 'IAU_MOON' in note and '4 dk' in note
    assert 'kayıtlı' in widget._scope_note.text()
    assert widget._data is data
    assert 'samples' in widget.statusLabel.text()
    widget._anim_step()
    assert 'No sample visibility' in widget._anim_label.get_text()
    assert 'not computed' not in widget._anim_label.get_text()
