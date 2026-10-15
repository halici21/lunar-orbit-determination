"""Record a real Qt window and real tutorial QProcess; no fabricated UI frames."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"))
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "desktop_app")]

import imageio_ffmpeg
import numpy as np
from PyQt5.QtCore import QSettings, Qt, QCoreApplication, QEvent
from PyQt5.QtGui import QFont, QImage
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtTest import QTest


def record(output):
    output.mkdir(parents=True, exist_ok=False)
    qt = QApplication.instance() or QApplication([])
    qt.setFont(QFont("Segoe UI", 10))
    from desktop_app.main import _load_stylesheet
    _load_stylesheet(qt)
    import app as desktop
    settings = QSettings(str(output / "recording-settings.ini"), QSettings.IniFormat)
    desktop.QSettings = lambda *args, **kwargs: settings
    window = desktop.MainWindow()
    window.resize(1440, 900)
    caption = QLabel("Lunar OD · gerçek uygulama / sentetik eğitim")
    caption.setStyleSheet("font-size:12px; color:#012855; padding:0 4px; background:#edf3f8;")
    window.statusBar.insertPermanentWidget(0, caption, 1)
    window.show()
    qt.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    size = window.grab().size()
    video = output / "desktop-walkthrough.mp4"
    writer = imageio_ffmpeg.write_frames(str(video), (size.width(), size.height()), fps=8,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=1,
        output_params=["-crf", "23", "-movflags", "+faststart"])
    writer.send(None)
    started = time.monotonic()
    actions = set()
    result_path = None
    menu = window.menuBar().actions()[-1].menu()
    try:
        for frame in range(60 * 8):
            elapsed = frame / 8
            if elapsed >= 2 and "menu" not in actions:
                menu.popup(window.menuBar().mapToGlobal(window.menuBar().actionGeometry(window.menuBar().actions()[-1]).bottomLeft()))
                actions.add("menu")
                caption.setText("1 / Örnek menüsü: Python veya SPICE indirmeden eğitimi başlat")
            if elapsed >= 5 and "run" not in actions:
                menu.hide()
                menu.actions()[0].trigger()
                actions.add("run")
                caption.setText("2 / Gerçek alt süreç: aynı model, kayıtlı seed, SR-UKF güncellemeleri")
            monitor = window._pages.get("run_monitor")
            if monitor and monitor._result_path is not None:
                result_path = monitor._result_path
            if elapsed >= 12 and result_path and "results" not in actions:
                window.open_in_result_console(result_path)
                caption.setText("3 / Sonuçlar: konum ve hız farkları, normalize NIS ve kaynak kayıtları")
                actions.add("results")
            console = window._pages.get("console")
            if elapsed >= 18 and console and console.result and "selection" not in actions:
                console.time_slider.setValue(64)
                caption.setText("4 / Zaman seçimi: gerçek 124 güncellemeden bir kaydı incele")
                actions.add("selection")
            if elapsed >= 26 and console and console.result and "details" not in actions:
                window.open_result_details()
                caption.setText("5 / Koşu ayrıntıları: birimler, sentetik çerçeve, seed ve dosya kimlikleri")
                actions.add("details")
            if elapsed >= 35 and result_path and "compare" not in actions:
                other = result_path.with_name("tutorial_seed_41001_ukf_updates.csv")
                window.compare_runs(result_path, other)
                caption.setText("6 / İki gerçek koşu: aynı model ve takvim, farklı gürültü tohumu")
                actions.add("compare")
            comparison = window._pages.get("comparison")
            if elapsed >= 46 and comparison and comparison.comparison and "differences" not in actions:
                # Use the actual page's tabs, with the stored differences visible.
                from PyQt5.QtWidgets import QTabWidget
                tabs = comparison.findChildren(QTabWidget)
                if tabs:
                    for index in range(tabs[0].count()):
                        if "fark" in tabs[0].tabText(index).lower():
                            tabs[0].setCurrentIndex(index)
                            break
                caption.setText("7 / Ayar farkları ve yorum sınırları: eğitim, operasyonel doğrulama değildir")
                actions.add("differences")
            if elapsed >= 54 and result_path and "end" not in actions:
                window.open_in_result_console(result_path)
                caption.setText("Tekrar üretilebilir: Örnek menüsü → çalıştır → sonuçları incele → karşılaştır")
                actions.add("end")
            qt.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            image = window.grab().toImage().convertToFormat(QImage.Format_RGB888)
            bits = image.bits()
            bits.setsize(image.height() * image.bytesPerLine())
            pixels = np.frombuffer(bits, dtype=np.uint8).reshape(image.height(), image.bytesPerLine())
            pixels = pixels[:, :image.width() * 3].reshape(image.height(), image.width(), 3)
            writer.send(np.ascontiguousarray(pixels))
            if frame in (1, 20 * 8, 29 * 8, 40 * 8, 49 * 8):
                window.grab().save(str(output / f"frame-{frame:03d}.png"))
            deadline = started + (frame + 1) / 8
            while time.monotonic() < deadline:
                qt.processEvents()
                time.sleep(.002)
        required = {"menu", "run", "results", "selection", "details", "compare", "differences", "end"}
        if actions != required or not comparison or not comparison.comparison:
            raise RuntimeError(f"Incomplete recorded workflow: {actions}")
    finally:
        writer.close()
        window.close()
        qt.processEvents()
    report = {"schema": "lunar-od-desktop-recording-v1", "duration_s": 60, "fps": 8,
        "capture": "Actual Qt widget rendering, offscreen platform; annotations are the recording footer",
        "execution": "Real QProcess and scientific estimators, no mocked result values",
        "result_sources": [str(result_path), str(other)],
        "steps": sorted(actions), "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest()}
    from examples.quickstart_od import source_identity
    report["source"] = source_identity()
    report["source"]["source_sha256"].update({name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("tools/record_desktop_walkthrough.py", "desktop_app/app.py",
                     "examples/desktop_tutorial.py", "examples/benchmark_estimators.py")})
    report["result_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (result_path, other)}
    (output / "recording.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {video}: all workflow stages verified", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/desktop-recording"))
    record(parser.parse_args().output.resolve())
