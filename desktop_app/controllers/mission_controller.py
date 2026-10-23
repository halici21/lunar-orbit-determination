"""Results overview tab: recorded orbit playback, station map, and sampled contacts.

Embedded as the first tab of the Sonuçlar page; the page header owns opening,
exporting and details, so this widget carries no navigation buttons of its own.
"""
from __future__ import annotations
import bisect
import time

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
                            QSlider, QComboBox, QSplitter, QSizePolicy)
from services.mission_data import station_catalog, orbit_summary
from services.result_summary import result_status
from widgets.mission_plots import StationWorldMap, MissionGantt
from widgets.tracking_plots import StoredOrbit
from styles.theme import C


def caption(text, parent=None):
    label = QLabel(text, parent)
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    return label


class MissionController(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self._times = []
        self._position = 0
        self._target_t = 0.
        self._last_tick = None
        self._mu = None
        self._intervals = {}
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._tick)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 12, 0, 8)
        root.setSpacing(10)
        self.run_label = caption("Henüz koşu açılmadı · haritada istasyon kataloğu gösteriliyor")
        self.run_label.setObjectName("detailHeading")
        root.addWidget(self.run_label)
        self.run_status = caption("Arşivden bir koşu açın; yörünge, istasyonlar ve ölçüm zamanları burada birlikte oynatılır.")
        self.run_status.setAccessibleName("Yüklü koşunun işlem ve kayıtlı yay başarı durumu")
        self.run_status.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:12px;")
        self.run_status.setToolTip("Yay başarısı kaynak CSV’deki operational_success alanıdır. Bu alanlar yakınsama veya "
                                  "bilimsel doğruluk onayı değildir; referans yörünge de başarı kanıtı değildir.")
        root.addWidget(self.run_status)

        self.plots = plots = QSplitter(Qt.Horizontal)
        plots.setChildrenCollapsible(False)
        orbit_page = QWidget()
        orbit_layout = QVBoxLayout(orbit_page)
        orbit_layout.setContentsMargins(0, 0, 12, 0)
        orbit_layout.setSpacing(4)
        orbit_layout.addWidget(caption("AY MERKEZLİ YÖRÜNGE"))
        self.orbit_caption = caption("Kayıtlı x/y/z durumları · sürükleyerek döndür, tekerlekle yakınlaştır")
        self.orbit_caption.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:11px;")
        orbit_layout.addWidget(self.orbit_caption)
        self.orbit = StoredOrbit()
        self.orbit.setMinimumHeight(260)
        orbit_layout.addWidget(self.orbit, 1)
        self.orbit_info = caption("Bir koşu açıldığında kayıtlı yörünge ve seçili epokun yükseklik, eğim ve periyot bilgileri burada görünür.")
        self.orbit_info.setMinimumHeight(42)
        orbit_layout.addWidget(self.orbit_info)
        map_page = QWidget()
        map_layout = QVBoxLayout(map_page)
        map_layout.setContentsMargins(12, 0, 0, 0)
        map_layout.setSpacing(4)
        map_layout.addWidget(caption("DÜNYADAKİ YER İSTASYONLARI"))
        self.map_caption = caption("İçi boş: katalog / seçili ağ · dolu: seçili zamanda ölçüm kaydı")
        self.map_caption.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:11px;")
        map_layout.addWidget(self.map_caption)
        station_row = QHBoxLayout()
        station_label = QLabel("İstasyon:")
        self.station_picker = QComboBox()
        self.station_picker.setAccessibleName("Harita ve Gantt için istasyon seçimi")
        self.station_picker.setMinimumContentsLength(14)
        self.station_picker.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.station_picker.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.station_picker.setToolTip("Listeden veya haritadan seçin; Gantt satır başlıklarına da tıklayabilirsiniz.")
        station_label.setBuddy(self.station_picker)
        station_row.addWidget(station_label)
        station_row.addWidget(self.station_picker, 1)
        map_layout.addLayout(station_row)
        self.world_map = StationWorldMap(station_catalog())
        self.world_map.stationSelected.connect(self._select_station)
        self._fill_station_picker()
        self.station_picker.currentIndexChanged.connect(self._picker_changed)
        map_layout.addWidget(self.world_map, 1)
        self.station_info = caption("Noktaya tıkla: istasyon adı ve koordinatları. DSN / ITU / ESA ve diğer ağlar katalogdan gelir.")
        self.station_info.setMinimumHeight(42)
        map_layout.addWidget(self.station_info)
        plots.addWidget(orbit_page)
        plots.addWidget(map_page)
        plots.setSizes([600, 450])
        root.addWidget(plots, 3)

        row = QHBoxLayout()
        row.addWidget(caption("İSTASYON ÖLÇÜM ZAMANLARI · GANTT"), 1)
        self.visibility_button = QPushButton("Görünürlük sayfası")
        self.visibility_button.setToolTip("Seçtiğiniz istasyonlar için örnek yörüngeyle ayrı bir temas planı hesaplayın.")
        self.visibility_button.clicked.connect(lambda: self._navigate("visibility"))
        row.addWidget(self.visibility_button)
        root.addLayout(row)
        self.gantt = MissionGantt()
        self.gantt.timeSelected.connect(self.select_time)
        self.gantt.stationSelected.connect(self._select_station)
        root.addWidget(self.gantt, 1)
        row = QHBoxLayout()
        self.play_button = QPushButton("▶ Oynat")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._play)
        self.reset_button = QPushButton("Başa dön")
        self.reset_button.clicked.connect(self.reset)
        self.speed_combo = QComboBox()
        for label, speed in (("1 sa / sn", 3600.), ("10 sa / sn", 36000.), ("1 gün / sn", 86400.)):
            self.speed_combo.addItem(label, speed)
        self.speed_combo.setToolTip("Bir duvar saati saniyesinde ilerletilen simülasyon süresi; hesaplama hızını değiştirmez.")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setAccessibleName("Kaydedilmiş yörünge zamanı")
        self.slider.valueChanged.connect(self._slider_changed)
        self.time_label = QLabel("T+ —")
        for widget in (self.play_button, self.reset_button, self.speed_combo):
            row.addWidget(widget)
        row.addWidget(self.slider, 1)
        row.addWidget(self.time_label)
        root.addLayout(row)
        self.help_button = QPushButton("Bu görünüm ne gösteriyor?  ›")
        self.help_button.setCheckable(True)
        self.help_button.setStyleSheet(f"border:0; text-align:left; padding:2px 0; color:{C.TEXT_MUTED};")
        self.help_text = caption("Gantt, koşuda kaydedilen ölçüm örneklerini gösterir; kesintisiz radyo teması veya çözüm başarısı değildir. "
                                 "Oynatma yalnız kayıtlı durumları izler.\n"
                                 "Yükseklik: kayıtlı konumun Ay merkezine uzaklığı eksi 1737,4 km. Eğim: yörünge düzleminin kayıt çerçevesinin XY düzlemine açısı; Ay ekvatoru değildir. "
                                 "Yarı büyük eksen (a): anlık iki cisim elipsinin boyutu. Dışmerkezlik (e): 0 daire, 0–1 elips. Periyot: aynı iki cisim modelindeki tur süresi. "
                                 "Truth, senaryonun referans hareketidir; ○ işaretleri kestiricinin yay sonu/başlangıç durumlarıdır. Bunlar gerçek uzay aracı telemetrisi değildir.")
        self.help_text.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:12px;")
        self.help_button.toggled.connect(self.help_text.setVisible)
        root.addWidget(self.help_button, alignment=Qt.AlignLeft)
        root.addWidget(self.help_text)
        self.help_text.hide()
        self._set_controls(False)

    def resizeEvent(self, event):
        # The outer page scroll keeps both full-size plots reachable on laptops.
        orientation = Qt.Vertical if self.width() < 920 else Qt.Horizontal
        if self.plots.orientation() != orientation:
            self.plots.setOrientation(orientation)
            vertical = orientation == Qt.Vertical
            self.plots.widget(0).layout().setContentsMargins(0, 0, 0 if vertical else 12, 0)
            self.plots.widget(1).layout().setContentsMargins(0 if vertical else 12, 0, 0, 0)
            self.plots.setSizes([360, 360] if vertical else [600, 450])
        super().resizeEvent(event)

    def _fill_station_picker(self):
        self.station_picker.blockSignals(True)
        self.station_picker.clear()
        self.station_picker.addItem("Tüm istasyonlar", None)
        for station in self.world_map.stations:
            self.station_picker.addItem(station["name"], station["name"])
        self.station_picker.blockSignals(False)

    def _picker_changed(self, index):
        self._select_station(self.station_picker.itemData(index))

    def _navigate(self, page):
        if hasattr(self.window(), "navigate"):
            self.window().navigate(page)

    def _set_controls(self, enabled):
        for widget in (self.play_button, self.reset_button, self.speed_combo, self.slider):
            widget.setEnabled(enabled)

    def set_result(self, result, details=None):
        self.play_button.setChecked(False)
        self.result = result
        details = details or {}
        payload = details
        for key in ("force", "truth", "canonical_payload", "lunar_point_mass"):
            payload = payload.get(key, {}) if isinstance(payload, dict) else {}
        self._mu = payload.get("gravitational_parameter_m3_s2") if isinstance(payload, dict) else None
        self.run_label.setText(f"{result.name} · {result.metadata.get('estimator_type', 'kayıt')} · {len(result.states):,} durum örneği")
        self.run_label.setToolTip(result.sources[0]["path"])
        self.run_status.setText(result_status(result, details) +
                                " · Yakınsama: kaynakta ayrı bir doğrulama alanı yok.")
        self.orbit_caption.setText(f"{result.metadata.get('origin', 'Bilinmiyor')} merkezli · {result.metadata.get('frame', 'Bilinmiyor')} · "
                                  f"{'referans (truth)' if result.metadata.get('state_role') == 'truth' else 'kayıtlı durum'} · sürükle / yakınlaştır")
        self.orbit.selected_t = None
        self.orbit.set_result(result)
        self.gantt.set_result(result)
        geometry = result.metadata.get("mission_geometry")
        self.world_map.set_stations(geometry["stations"] if geometry else station_catalog())
        self._fill_station_picker()
        self.map_caption.setText("Seçili koşunun istasyonları · dolu nokta: ölçüm aralığında" if geometry else
                                 "Katalog konumları · bu koşunun istasyon eşlemesi kaydedilmemiş")
        self._intervals = {}
        if geometry:
            for station in geometry["stations"]:
                windows = sorted((w["start_s"], w["end_s"]) for w in geometry["windows"] if w["station_id"] == station["station_id"])
                merged = []
                for start, end in windows:
                    if merged and start <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                    else:
                        merged.append((start, end))
                self._intervals[station["name"]] = ([w[0] for w in merged], [w[1] for w in merged])
        self._times = [r["t_s"] for r in result.states]
        self._set_controls(len(self._times) > 1)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(self._times)-1))
        self.slider.blockSignals(False)
        self.station_info.setText("Noktaya tıklayarak istasyon adını ve koordinatlarını inceleyin.")
        self._select_index(0)
        self._refresh_station_info()

    def _select_station(self, name):
        if name is not None and not any(s["name"] == name for s in self.world_map.stations):
            return
        self.world_map.selected = name
        self.world_map.update()
        self.gantt.selected_station = name
        self.gantt.update()
        self.station_picker.blockSignals(True)
        self.station_picker.setCurrentIndex(max(0, self.station_picker.findData(name)))
        self.station_picker.blockSignals(False)
        self._refresh_station_info()

    def _refresh_station_info(self):
        name = self.world_map.selected
        station = next((s for s in self.world_map.stations if s["name"] == name), None)
        geometry = self.result.metadata.get("mission_geometry") if self.result else None
        if not self.result:
            status = "Katalog konumu · ölçüm zamanlarını görmek için bir koşu açın."
        elif not geometry:
            status = "Bu koşuda istasyon ölçüm zamanları kaydedilmemiş."
        elif not self._times:
            status = "Durum geçmişi yok · seçili epok için ölçüm durumu gösterilemiyor."
        elif station:
            status = "Seçili epokta ölçüm kaydı " + ("var." if name in self.world_map.active else "yok.")
        else:
            active = sorted(self.world_map.active)
            names = ", ".join(active[:3]) + (f" (+{len(active)-3})" if len(active) > 3 else "")
            status = f"Seçili epokta ölçüm kaydı: {names}." if active else "Seçili epokta hiçbir istasyonun ölçüm kaydı yok."
        coordinates = f"{name} · {station['lat_deg']:.3f}° enlem · {station['lon_deg']:.3f}° boylam\n" if station else ""
        self.station_info.setText(coordinates + status)
        self.station_info.setToolTip("\n".join(sorted(self.world_map.active)))

    def _select_index(self, index):
        if not self._times:
            self.orbit_info.setText("Bu sonuçta durum geçmişi yok; yörünge ve animasyon için x/y/z ile hız kayıtları gerekir.")
            self.time_label.setText("T+ —")
            return
        self._position = min(max(0, index), len(self._times)-1)
        t_s = self._times[self._position]
        self.slider.blockSignals(True)
        self.slider.setValue(self._position)
        self.slider.blockSignals(False)
        self.orbit.selected_t = t_s
        self.orbit.update()
        self.gantt.cursor = t_s
        self.gantt.update()
        active = set()
        for name, (starts, ends) in self._intervals.items():
            index = bisect.bisect_right(starts, t_s)-1
            if index >= 0 and t_s <= ends[index]:
                active.add(name)
        self.world_map.active = active
        self.world_map.update()
        self._refresh_station_info()
        self.time_label.setText(f"T+ {t_s/3600:.2f} sa · {self._position+1}/{len(self._times)}")
        values = orbit_summary(self.result.states[self._position], mu=self._mu)
        def fmt(key, precision=1):
            return f"{values[key]:.{precision}f}" if values[key] is not None else "bilinmiyor"
        eccentricity = values["e"]
        shape = ("biçim bilinmiyor" if eccentricity is None else
                 "daireye yakın" if eccentricity < .01 else "eliptik" if eccentricity < 1 else "açık yörünge")
        self.orbit_info.setText(f"{shape.capitalize()} · yükseklik {fmt('altitude_km')} km · eğim {fmt('inclination_deg')}°\n"
                                f"Yarı büyük eksen {fmt('a_km')} km · e {fmt('e', 4)} · periyot {fmt('period_h', 2)} sa")
        self.orbit_info.setToolTip("Eğim kayıt çerçevesinin XY düzlemine göredir. a, e ve periyot kayıtlı kuvvet manifestindeki Ay GM’siyle "
                                  "hesaplanan anlık iki cisim tanımlayıcılarıdır; pertürbasyonlu hareketin tam tekrarını garanti etmez.")

    def _slider_changed(self, index):
        self.play_button.setChecked(False)
        self._select_index(index)

    def select_time(self, t_s):
        self.play_button.setChecked(False)
        if self._times:
            index = bisect.bisect_left(self._times, t_s)
            choices = [n for n in (index-1, index) if 0 <= n < len(self._times)]
            self._select_index(min(choices, key=lambda n: abs(self._times[n]-t_s)))

    def _play(self, checked):
        self.play_button.setText("❚❚ Duraklat" if checked else "▶ Oynat")
        if checked and len(self._times) > 1:
            if self._position == len(self._times)-1:
                self._select_index(0)
            self._target_t = self._times[self._position]
            self._last_tick = time.monotonic()
            self._timer.start()
        else:
            self._timer.stop()

    def _tick(self):
        now = time.monotonic()
        self._target_t += (now-self._last_tick)*self.speed_combo.currentData()
        self._last_tick = now
        index = min(len(self._times)-1, max(0, bisect.bisect_right(self._times, self._target_t)-1))
        if index != self._position:
            self._select_index(index)
        if self._target_t >= self._times[-1]:
            self.play_button.setChecked(False)

    def reset(self):
        self.play_button.setChecked(False)
        self._select_index(0)
        self.orbit.reset()

    def hideEvent(self, event):
        self.play_button.setChecked(False)
        super().hideEvent(event)

    def shutdown(self):
        self.play_button.setChecked(False)
        self._timer.stop()
