"""Large workspace optimizations retain fresh archives and exact selected states."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import math
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QColor
from services import run_archive
from widgets.tracking_plots import StoredOrbit, ACCENT
from controllers.run_archive_controller import RunArchiveController
from tests.test_tracking_console import make_run, wait_until


@pytest.fixture
def app():
    yield QApplication.instance() or QApplication([])


def result(count=12001):
    return SimpleNamespace(
        states=[dict(t_s=i*10, x_m=2200000*math.cos(i*.001), y_m=2200000*math.sin(i*.001), z_m=0) for i in range(count)],
        updates=[], metadata={"frame": "test"})


def test_orbit_cached_render_matches_fresh_after_view_changes(app):
    orbit = StoredOrbit()
    orbit.resize(700, 500)
    orbit.set_result(result())
    orbit.grab()
    original = orbit._base_image.cacheKey()
    orbit.selected_t = 31
    selected = orbit.grab().toImage()
    assert orbit._base_image.cacheKey() == original
    for attribute, value in (("yaw", 1.2), ("pitch", .9), ("zoom", 1.5)):
        setattr(orbit, attribute, value)
        cached = orbit.grab().toImage()
        assert orbit._base_image.cacheKey() != original
        original = orbit._base_image.cacheKey()
        orbit._base_key = None
        assert cached == orbit.grab().toImage()
    orbit.resize(800, 550)
    resized = orbit.grab().toImage()
    orbit._base_key = None
    assert resized == orbit.grab().toImage()
    orbit.set_result(None)
    orbit.grab()
    assert orbit._project is None
    orbit.set_result(result(5))
    loaded = orbit.grab().toImage()
    orbit._base_key = None
    assert loaded == orbit.grab().toImage()
    orbit.close()


def test_selection_uses_raw_state_even_when_line_is_decimated(app):
    orbit = StoredOrbit()
    orbit.resize(700, 500)
    source = result()
    # Index 3 is skipped by the displayed line's stride of 2.
    source.states[3]["z_m"] = 900000
    orbit.set_result(source)
    orbit.selected_t = 31
    image = orbit.grab().toImage()
    point = orbit._project(orbit._points[3])
    assert image.pixelColor(round(point.x()), round(point.y())) == QColor(ACCENT)
    assert len(orbit._points) == len(source.states) == 12001
    orbit.selected_t = None
    cleared = orbit.grab().toImage()
    orbit._base_key = None
    assert cleared == orbit.grab().toImage()
    assert cleared != image
    orbit.close()


def test_archive_cache_reuses_counts_but_refreshes_manifest_and_changed_csv(tmp_path):
    path = make_run(tmp_path)
    cache = {}
    real_count = run_archive._count_csv
    with patch.object(run_archive, "_count_csv", wraps=real_count) as count:
        first = run_archive.scan_run_archive(tmp_path, count_cache=cache)
        assert run_archive.scan_run_archive(tmp_path, count_cache=cache) == first
        assert count.call_count == 1
        manifest = path.with_name("test_result_manifest.json")
        manifest.write_text('{"schema":"unsupported"}')
        assert run_archive.scan_run_archive(tmp_path, count_cache=cache)[0].status == "invalid"
        assert count.call_count == 1
        manifest.unlink()
        with path.open("a") as handle:
            handle.write("4,3,50,1,60,.02,3,1,True\n")
        assert run_archive.scan_run_archive(tmp_path, count_cache=cache)[0].records == 4
        assert count.call_count == 2
        with path.open("a") as handle:
            handle.write("broken,row\n")
        assert run_archive.scan_run_archive(tmp_path, count_cache=cache)[0].status == "invalid"
        path.unlink()
        assert not run_archive.scan_run_archive(tmp_path, count_cache=cache)


def test_archive_cached_scan_still_cancels(tmp_path):
    make_run(tmp_path)
    cache = {}
    run_archive.scan_run_archive(tmp_path, count_cache=cache)
    with pytest.raises(run_archive.LoadCancelled):
        run_archive.scan_run_archive(tmp_path, cancel=lambda: True, count_cache=cache)


def test_archive_navigation_reuses_count_and_refresh_button_rescans(app, tmp_path):
    make_run(tmp_path)
    page = RunArchiveController()
    real_count = run_archive._count_csv
    try:
        with patch.object(run_archive, "_count_csv", wraps=real_count) as count:
            page.set_folder(tmp_path)
            wait_until(app, lambda: not page._jobs)
            assert count.call_count == 1
            page.on_shown()
            wait_until(app, lambda: not page._jobs)
            assert count.call_count == 1
            page.refresh_button.click()
            wait_until(app, lambda: not page._jobs)
            assert count.call_count == 2
    finally:
        page.shutdown()
        page.close()
