#!/usr/bin/env python3
"""Lunar OD native desktop application.

This is a real desktop GUI: no browser, no HTTP server, no webview wrapper.
Run from the repository root or from ``python_port``:

    python python_port/lunar_od_app.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
WORLD_MAP_PATH = PROJECT_ROOT / "dunya-haritasi-dilsiz.png"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lunar_od.config import position_only_stations, range_rate_stations
from lunar_od.scenario_config import scenario_config_from_mapping


COLORS = {
    "bg": "#080c18",
    "panel": "#0f1932",
    "panel_2": "#0b1427",
    "sidebar": "#080e1e",
    "card": "#101d37",
    "ink": "#e8edf5",
    "muted": "#8090aa",
    "line": "#24344f",
    "blue": "#4facfe",
    "cyan": "#00d4ff",
    "green": "#43e97b",
    "yellow": "#f6d365",
    "red": "#f5576c",
    "orange": "#f97316",
    "purple": "#a855f7",
}

FAMILY_COLORS = {
    "ITU": "#4facfe",
    "DSN": "#f97316",
    "KGS": "#a855f7",
    "ESA": "#43e97b",
    "RUS": "#f5576c",
    "ISRO": "#eab308",
    "Other": "#a0b4d0",
}

NETWORKS = {
    "single": ("Canberra DSN",),
    "multi": ("Goldstone DSN", "Madrid DSN", "Canberra DSN", "ITU Ayazaga"),
}


@dataclass(frozen=True)
class StationView:
    name: str
    family: str
    lat_deg: float
    lon_deg: float
    alt_m: float
    sigma_range_m: float
    sigma_angle_deg: float
    sigma_range_rate_mps: float | None

    @property
    def supports_range_rate(self) -> bool:
        return self.sigma_range_rate_mps is not None


def station_family(name: str) -> str:
    for family in ("ITU", "DSN", "KGS", "ESA", "RUS", "ISRO"):
        if family in name:
            return family
    return "Other"


def load_station_views() -> list[StationView]:
    by_name: dict[str, StationView] = {}
    for station in position_only_stations():
        by_name[station.name] = StationView(
            name=station.name,
            family=station_family(station.name),
            lat_deg=station.lat_deg,
            lon_deg=station.lon_deg,
            alt_m=station.alt_m,
            sigma_range_m=station.sigma_range_m,
            sigma_angle_deg=math.degrees(station.sigma_angle_rad),
            sigma_range_rate_mps=None,
        )
    for station in range_rate_stations():
        by_name[station.name] = StationView(
            name=station.name,
            family=station_family(station.name),
            lat_deg=station.lat_deg,
            lon_deg=station.lon_deg,
            alt_m=station.alt_m,
            sigma_range_m=station.sigma_range_m,
            sigma_angle_deg=math.degrees(station.sigma_angle_rad),
            sigma_range_rate_mps=station.sigma_range_rate_mps,
        )
    return sorted(by_name.values(), key=lambda s: (s.family, s.name))


def lon_to_x(lon: float, width: int) -> float:
    return ((lon + 180.0) / 360.0) * width


def lat_to_y(lat: float, height: int) -> float:
    return ((90.0 - lat) / 180.0) * height


def angular_distance_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    cos_d = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dlon)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_d))))


class LunarODDesktopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Lunar OD - Native Desktop")
        self.geometry("1320x860")
        self.minsize(1040, 680)
        self.configure(bg=COLORS["bg"])

        self.stations = load_station_views()
        self.selected_network = tk.StringVar(value="multi")
        self.sim_frame = tk.IntVar(value=0)
        self.sim_frame_count = 96
        self.sim_playing = False
        self._syncing_sim_scale = False
        self.analysis_frame = tk.IntVar(value=0)
        self.analysis_playing = False
        self._syncing_analysis_scale = False
        self.analysis_plot = tk.StringVar(value="ground_track")
        self.analysis_mode = tk.StringVar(value="static")
        self.measurement_type = tk.StringVar(value="range_rate")
        self.start_mode = tk.StringVar(value="cold")
        self.estimator_type = tk.StringVar(value="srif")
        self.scenario_network = tk.StringVar(value="multi")
        self.scenario_name = tk.StringVar(value="hot_range_rate_multi")
        self.duration_h = tk.DoubleVar(value=4.0)
        self.sample_step_s = tk.DoubleVar(value=240.0)
        self.max_iter = tk.IntVar(value=10)
        self.rtol = tk.StringVar(value="1e-8")
        self.atol = tk.StringVar(value="1e-10")
        self.noise_enabled = tk.BooleanVar(value=True)
        self.bias_mode = tk.StringVar(value="none")
        self.range_rate_physics = tk.StringVar(value="geometric_instantaneous")
        self.count_interval_s = tk.StringVar(value="60.0")
        self.uplink_frequency_hz = tk.StringVar(value="7200000000.0")
        self.turnaround_ratio = tk.StringVar(value=str(880.0 / 749.0))
        self.two_way_local_state_model = tk.StringVar(value="ode")
        self.station_clock_offset_s = tk.StringVar(value="0.0")
        self.station_clock_drift = tk.StringVar(value="0.0")
        self.clock_reference_time_s = tk.StringVar(value="0.0")
        self.transponder_delay_s = tk.StringVar(value="0.0")
        self.ukf_alpha = tk.StringVar(value="0.35")
        self.ukf_beta = tk.StringVar(value="2.0")
        self.ukf_kappa = tk.StringVar(value="0.0")
        self.ukf_covariance_inflation = tk.StringVar(value="1.0")
        self.ukf_adaptive_measurement_noise = tk.BooleanVar(value=False)
        self.ukf_nis_gate = tk.StringVar(value="")
        self.ukf_component_nis_gate = tk.StringVar(value="")
        self.ukf_component_gate_mode = tk.StringVar(value="marginal")
        self.ukf_robust_measurement_update = tk.BooleanVar(value=False)
        self.ukf_robust_loss = tk.StringVar(value="student_t")
        self.ukf_robust_student_t_dof = tk.StringVar(value="5.0")
        self.ukf_robust_huber_threshold = tk.StringVar(value="3.0")
        self.ukf_robust_min_component_weight = tk.StringVar(value="0.05")
        self.ukf_covariance_form = tk.StringVar(value="standard")
        self.ukf_process_noise_model = tk.StringVar(value="discrete")
        self.ukf_acceleration_psd = tk.StringVar(value="")
        self.ukf_adaptive_process_noise = tk.BooleanVar(value=False)
        self.ukf_auto_bias_constraints = tk.BooleanVar(value=False)
        self.ukf_bias_freeze_relative_information = tk.StringVar(value="1e-12")
        self.ukf_bias_regularize_relative_information = tk.StringVar(value="1e-5")
        self.ukf_bias_regularization_std = tk.StringVar(value="1.0")
        self.output_dir = tk.StringVar(value="python_port/results")
        self.world_map_image: tk.PhotoImage | None = None
        self.map_image_cache: dict[int, tk.PhotoImage] = {}
        self.nav_items: dict[str, tk.Label] = {}
        self.segmented_renderers: list[Any] = []
        self.load_world_map_asset()

        self._build_style()
        self._build_shell()
        self._build_dashboard_tab()
        self._build_simulation_tab()
        self._build_analysis_tab()
        self._build_stations_tab()
        self._build_scenario_tab()
        self._build_results_tab()
        self.refresh_dashboard()
        self.draw_simulation()
        self.draw_analysis()
        self.draw_station_map()
        self.refresh_scenario_preview()
        self.refresh_results()

    def load_world_map_asset(self) -> None:
        if not WORLD_MAP_PATH.is_file():
            return
        try:
            self.world_map_image = tk.PhotoImage(file=str(WORLD_MAP_PATH))
        except tk.TclError:
            self.world_map_image = None

    def _build_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=COLORS["bg"], foreground=COLORS["ink"], fieldbackground=COLORS["panel"])
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["ink"])
        style.configure("Muted.TLabel", background=COLORS["bg"], foreground=COLORS["muted"])
        style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["ink"])
        style.configure("MutedPanel.TLabel", background=COLORS["panel"], foreground=COLORS["muted"])
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), background=COLORS["bg"])
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"), foreground=COLORS["blue"], background=COLORS["panel"])
        style.configure("TNotebook", background=COLORS["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 10), background=COLORS["panel_2"], foreground=COLORS["muted"])
        style.map("TNotebook.Tab", background=[("selected", COLORS["panel"])], foreground=[("selected", COLORS["cyan"])])
        style.layout("Tabless.TNotebook.Tab", [])
        style.configure("Tabless.TNotebook", background=COLORS["bg"], borderwidth=0)
        style.configure("TButton", padding=(12, 8), background=COLORS["panel_2"], foreground=COLORS["ink"], bordercolor=COLORS["line"])
        style.map("TButton", background=[("active", "#172844")], foreground=[("active", COLORS["cyan"])])
        style.configure("Accent.TButton", background="#17405f", foreground=COLORS["ink"])
        style.configure("Treeview", background=COLORS["panel_2"], foreground=COLORS["ink"], fieldbackground=COLORS["panel_2"], rowheight=28)
        style.configure("Treeview.Heading", background=COLORS["panel"], foreground=COLORS["blue"], font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", "#1d4770")])
        style.configure("TEntry", fieldbackground=COLORS["panel_2"], foreground=COLORS["ink"], insertcolor=COLORS["ink"])
        style.configure("TCombobox", fieldbackground=COLORS["panel_2"], foreground=COLORS["ink"], arrowcolor=COLORS["cyan"])
        style.configure("Horizontal.TScale", background=COLORS["panel"], troughcolor=COLORS["panel_2"])
        style.configure("TCheckbutton", background=COLORS["panel"], foreground=COLORS["ink"])
        style.map("TCheckbutton", background=[("active", COLORS["panel"])], foreground=[("active", COLORS["cyan"])])
        style.configure("TRadiobutton", background=COLORS["panel"], foreground=COLORS["ink"], indicatorcolor=COLORS["panel_2"])
        style.map("TRadiobutton", background=[("active", COLORS["panel"])], foreground=[("active", COLORS["cyan"])])

    def _build_shell(self) -> None:
        self.app_shell = tk.Frame(self, bg=COLORS["bg"])
        self.app_shell.pack(fill=tk.BOTH, expand=True)

        self.sidebar = tk.Frame(self.app_shell, bg="#080e1e", width=240, highlightthickness=1, highlightbackground="#1a2842")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        brand = tk.Frame(self.sidebar, bg="#080e1e", height=70)
        brand.pack(fill=tk.X)
        brand.pack_propagate(False)
        tk.Label(brand, text="Lunar OD", bg="#080e1e", fg=COLORS["cyan"], font=("Segoe UI", 18, "bold")).pack(anchor=tk.W, padx=20, pady=(18, 0))
        tk.Label(brand, text="Research Dashboard", bg="#080e1e", fg=COLORS["muted"], font=("Segoe UI", 9)).pack(anchor=tk.W, padx=20)
        tk.Frame(self.sidebar, bg="#17243b", height=1).pack(fill=tk.X)

        nav = tk.Frame(self.sidebar, bg="#080e1e")
        nav.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.nav_specs = [
            ("dashboard", "Dashboard", "Mission overview"),
            ("simulation", "Simulation", "Frame playback"),
            ("analysis", "Ground Track / Visibility", "Static or animated plots"),
            ("stations", "Stations", "Tracking network"),
            ("scenario", "Scenario Builder", "Config and validation"),
            ("results", "Results Browser", "CSV and artifacts"),
        ]
        for page_id, label, hint in self.nav_specs:
            self._add_nav_item(nav, page_id, label, hint)

        footer = tk.Frame(self.sidebar, bg="#080e1e", height=54)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        tk.Frame(self.sidebar, bg="#17243b", height=1).pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(footer, text="Native app online", bg="#080e1e", fg=COLORS["green"], font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=20, pady=(12, 0))
        tk.Label(footer, text="No web server / no browser", bg="#080e1e", fg=COLORS["muted"], font=("Segoe UI", 8)).pack(anchor=tk.W, padx=20)

        self.content_shell = tk.Frame(self.app_shell, bg=COLORS["bg"])
        self.content_shell.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.page_header = tk.Frame(self.content_shell, bg=COLORS["bg"])
        self.page_header.pack(fill=tk.X, padx=28, pady=(24, 8))
        self.page_title = tk.Label(self.page_header, text="", bg=COLORS["bg"], fg=COLORS["ink"], font=("Segoe UI", 22, "bold"))
        self.page_title.pack(anchor=tk.W)
        self.page_subtitle = tk.Label(self.page_header, text="", bg=COLORS["bg"], fg=COLORS["muted"], font=("Segoe UI", 10))
        self.page_subtitle.pack(anchor=tk.W, pady=(2, 0))

        self.notebook = ttk.Notebook(self.content_shell, style="Tabless.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(8, 20))
        self.dashboard_tab = ttk.Frame(self.notebook, padding=8)
        self.sim_tab = ttk.Frame(self.notebook, padding=8)
        self.analysis_tab = ttk.Frame(self.notebook, padding=8)
        self.stations_tab = ttk.Frame(self.notebook, padding=8)
        self.scenario_tab = ttk.Frame(self.notebook, padding=8)
        self.results_tab = ttk.Frame(self.notebook, padding=8)
        self.page_tabs = {
            "dashboard": self.dashboard_tab,
            "simulation": self.sim_tab,
            "analysis": self.analysis_tab,
            "stations": self.stations_tab,
            "scenario": self.scenario_tab,
            "results": self.results_tab,
        }
        for page_id, frame in self.page_tabs.items():
            self.notebook.add(frame, text=page_id)
        self.navigate("dashboard")

    def _add_nav_item(self, parent: tk.Widget, page_id: str, label: str, hint: str) -> None:
        item = tk.Label(
            parent,
            text=f"{label}\n{hint}",
            bg="#080e1e",
            fg=COLORS["muted"],
            justify=tk.LEFT,
            anchor=tk.W,
            padx=20,
            pady=10,
            font=("Segoe UI", 10),
            cursor="hand2",
        )
        item.pack(fill=tk.X, pady=1)
        item.bind("<Button-1>", lambda _event, pid=page_id: self.navigate(pid))
        item.bind("<Enter>", lambda _event, widget=item: widget.configure(bg="#0d1a31", fg=COLORS["ink"]))
        item.bind("<Leave>", lambda _event, pid=page_id, widget=item: self._style_nav_item(pid, widget))
        self.nav_items[page_id] = item

    def navigate(self, page_id: str) -> None:
        if page_id not in self.page_tabs:
            return
        self.notebook.select(self.page_tabs[page_id])
        titles = {
            "dashboard": ("Mission Dashboard", "Lunar Orbit Determination - research overview"),
            "simulation": ("Frame Simulation", "Step through station visibility and estimator start modes"),
            "analysis": ("Ground Track / Visibility", "Static plots or animated analysis playback"),
            "stations": ("Stations & Networks", "Ground station locations and measurement capabilities"),
            "scenario": ("Scenario Builder", "Configure and export OD scenario JSON"),
            "results": ("Results Browser", "Explore generated plots, CSV tables, and artifacts"),
        }
        title, subtitle = titles[page_id]
        self.page_title.configure(text=title)
        self.page_subtitle.configure(text=subtitle)
        for pid, widget in self.nav_items.items():
            self._style_nav_item(pid, widget, active=(pid == page_id))
        if page_id == "analysis":
            self.draw_analysis()
        elif page_id == "simulation":
            self.draw_simulation()
        elif page_id == "stations":
            self.draw_station_map()

    def _style_nav_item(self, page_id: str, widget: tk.Label, active: bool | None = None) -> None:
        if active is None:
            active = bool(self.nav_items.get(page_id) and str(self.notebook.select()) == str(self.page_tabs.get(page_id)))
        if active:
            widget.configure(bg="#0f2542", fg=COLORS["cyan"], font=("Segoe UI", 10, "bold"))
        else:
            widget.configure(bg="#080e1e", fg=COLORS["muted"], font=("Segoe UI", 10))

    def panel(self, parent: tk.Widget, padding: int | tuple[int, ...] = 14) -> tk.Frame:
        frame = tk.Frame(
            parent,
            bg=COLORS["panel"],
            padx=padding if isinstance(padding, int) else padding[0],
            pady=padding if isinstance(padding, int) else padding[0],
            highlightthickness=1,
            highlightbackground="#1d3150",
            highlightcolor=COLORS["blue"],
        )
        return frame

    def _build_dashboard_tab(self) -> None:
        hero = self.panel(self.dashboard_tab, padding=10)
        hero.pack(fill=tk.X)
        self.dashboard_hero = tk.Canvas(hero, height=260, bg=COLORS["panel_2"], highlightthickness=0)
        self.dashboard_hero.pack(fill=tk.X)

        top = ttk.Frame(self.dashboard_tab)
        top.pack(fill=tk.X)
        self.dashboard_cards = ttk.Frame(top)
        self.dashboard_cards.pack(fill=tk.X, pady=(16, 0))
        self.recent_panel = self.panel(self.dashboard_tab)
        self.recent_panel.pack(fill=tk.BOTH, expand=True, pady=(16, 0))
        ttk.Label(self.recent_panel, text="Recent result files", style="Section.TLabel").pack(anchor=tk.W)
        self.recent_list = tk.Listbox(
            self.recent_panel,
            bg=COLORS["panel_2"],
            fg=COLORS["ink"],
            selectbackground="#1d4770",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=COLORS["line"],
            font=("Consolas", 10),
        )
        self.recent_list.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def _card(self, parent: tk.Widget, title: str, value: str, subtitle: str, color: str = COLORS["blue"]) -> None:
        frame = self.panel(parent, padding=16)
        frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 12))
        tk.Frame(frame, bg=color, height=3).pack(fill=tk.X, pady=(0, 10))
        ttk.Label(frame, text=title.upper(), style="MutedPanel.TLabel", font=("Segoe UI", 8, "bold")).pack(anchor=tk.W)
        ttk.Label(frame, text=value, style="Panel.TLabel", font=("Segoe UI", 26, "bold"), foreground=color).pack(anchor=tk.W)
        ttk.Label(frame, text=subtitle, style="MutedPanel.TLabel").pack(anchor=tk.W)

    def refresh_dashboard(self) -> None:
        for child in self.dashboard_cards.winfo_children():
            child.destroy()
        result_files = self._result_files()
        rr_count = sum(1 for s in self.stations if s.supports_range_rate)
        self.draw_dashboard_hero(result_files, rr_count)
        self._card(self.dashboard_cards, "Stations", str(len(self.stations)), f"{rr_count} support range-rate")
        self._card(self.dashboard_cards, "Results", str(len(result_files)), "PNG/CSV artifacts in results")
        self._card(self.dashboard_cards, "Network", self.scenario_network.get(), "current scenario network", COLORS["green"])
        self._card(self.dashboard_cards, "Mode", self.measurement_type.get(), f"{self.start_mode.get()} start", COLORS["yellow"])

        self.recent_list.delete(0, tk.END)
        for path in result_files[:40]:
            size_kb = path.stat().st_size / 1024
            self.recent_list.insert(tk.END, f"{path.name:<68} {size_kb:>8.1f} KB")

    def draw_dashboard_hero(self, result_files: list[Path], rr_count: int) -> None:
        if not hasattr(self, "dashboard_hero"):
            return
        canvas = self.dashboard_hero
        canvas.delete("all")
        width = max(canvas.winfo_width(), 920)
        height = 260
        canvas.create_rectangle(0, 0, width, height, fill=COLORS["panel_2"], outline="")
        for x, y, r, color in (
            (width * 0.08, 44, 1, "#50627d"),
            (width * 0.18, 190, 1, "#50627d"),
            (width * 0.44, 62, 2, "#6a7d99"),
            (width * 0.72, 38, 1, "#50627d"),
            (width * 0.90, 170, 2, "#6a7d99"),
        ):
            canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="")
        canvas.create_text(26, 28, text="Mission Operations View", fill=COLORS["ink"], anchor=tk.W, font=("Segoe UI", 18, "bold"))
        canvas.create_text(
            26,
            54,
            text="Ground network, synthetic visibility, scenario controls, and result artifacts in one native surface.",
            fill=COLORS["muted"],
            anchor=tk.W,
            font=("Segoe UI", 10),
        )
        map_x, map_y, map_w, map_h = self.draw_map_background(canvas, 28, 84, int(width * 0.48), 138)
        data = self.get_frame_data(self.sim_frame.get())
        track_points: list[float] = []
        for i in range(0, self.sim_frame_count, 2):
            frame_data = self.get_frame_data(i)
            track_points.extend((map_x + lon_to_x(frame_data["craft_lon"], map_w), map_y + lat_to_y(frame_data["craft_lat"], map_h)))
        if len(track_points) >= 4:
            canvas.create_line(*track_points, fill=COLORS["cyan"], width=2, smooth=True)
        craft_x = map_x + lon_to_x(data["craft_lon"], map_w)
        craft_y = map_y + lat_to_y(data["craft_lat"], map_h)
        canvas.create_oval(craft_x - 5, craft_y - 5, craft_x + 5, craft_y + 5, fill=COLORS["cyan"], outline="")

        stat_x = int(width * 0.58)
        self._hero_metric(canvas, stat_x, 92, "Artifacts", str(len(result_files)), COLORS["blue"])
        self._hero_metric(canvas, stat_x + 150, 92, "RR Stations", str(rr_count), COLORS["green"])
        self._hero_metric(canvas, stat_x, 172, "Network", self.scenario_network.get(), COLORS["yellow"])
        self._hero_metric(canvas, stat_x + 150, 172, "Start", self.start_mode.get(), COLORS["purple"])
        canvas.create_line(stat_x - 24, 88, stat_x - 24, 222, fill="#1d3150", width=1)

    def _hero_metric(self, canvas: tk.Canvas, x: int, y: int, label: str, value: str, color: str) -> None:
        canvas.create_rectangle(x, y, x + 128, y + 54, fill=COLORS["panel"], outline="#1d3150")
        canvas.create_rectangle(x, y, x + 128, y + 3, fill=color, outline=color)
        canvas.create_text(x + 12, y + 20, text=label.upper(), fill=COLORS["muted"], anchor=tk.W, font=("Segoe UI", 7, "bold"))
        canvas.create_text(x + 12, y + 40, text=value, fill=COLORS["ink"], anchor=tk.W, font=("Segoe UI", 15, "bold"))

    def segmented(
        self,
        parent: tk.Widget,
        title: str,
        variable: tk.StringVar,
        options: tuple[tuple[str, str], ...],
        callback: Any,
    ) -> None:
        ttk.Label(parent, text=title, style="Section.TLabel").pack(anchor=tk.W)
        row = tk.Frame(parent, bg=COLORS["panel"])
        row.pack(fill=tk.X, pady=(10, 0))
        buttons: list[tuple[str, tk.Label]] = []

        def render() -> None:
            for value, widget in buttons:
                active = variable.get() == value
                widget.configure(
                    bg="#0f2542" if active else COLORS["panel_2"],
                    fg=COLORS["cyan"] if active else COLORS["muted"],
                    font=("Segoe UI", 9, "bold" if active else "normal"),
                    highlightbackground=COLORS["blue"] if active else COLORS["line"],
                )

        def choose(value: str) -> None:
            variable.set(value)
            render()
            callback()

        for index, (label, value) in enumerate(options):
            chip = tk.Label(
                row,
                text=label,
                bg=COLORS["panel_2"],
                fg=COLORS["muted"],
                padx=12,
                pady=9,
                cursor="hand2",
                highlightthickness=1,
                highlightbackground=COLORS["line"],
                font=("Segoe UI", 9),
            )
            chip.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0 if index == 0 else 6, 0))
            chip.bind("<Button-1>", lambda _event, v=value: choose(v))
            buttons.append((value, chip))
        render()
        self.segmented_renderers.append(render)

    def refresh_segmented_controls(self) -> None:
        for render in self.segmented_renderers:
            render()

    def _build_simulation_tab(self) -> None:
        left = self.panel(self.sim_tab, padding=12)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(self.sim_tab)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(16, 0))

        ttk.Label(left, text="Frame-by-frame tracking simulation", style="Section.TLabel").pack(anchor=tk.W)
        self.sim_canvas = tk.Canvas(left, width=860, height=560, bg=COLORS["panel_2"], highlightthickness=0)
        self.sim_canvas.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        playback = self.panel(right)
        playback.pack(fill=tk.X)
        ttk.Label(playback, text="Playback", style="Section.TLabel").pack(anchor=tk.W)
        row = ttk.Frame(playback, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=(10, 4))
        ttk.Button(row, text="<", command=lambda: self.step_sim(-1)).pack(side=tk.LEFT, padx=(0, 6))
        self.play_button = ttk.Button(row, text="Play", command=self.toggle_play, style="Accent.TButton")
        self.play_button.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text=">", command=lambda: self.step_sim(1)).pack(side=tk.LEFT)
        self.frame_scale = ttk.Scale(
            playback,
            from_=0,
            to=self.sim_frame_count - 1,
            orient=tk.HORIZONTAL,
            command=self.on_sim_scale,
        )
        self.frame_scale.pack(fill=tk.X, pady=(8, 0))

        mode_panel = self.panel(right)
        mode_panel.pack(fill=tk.X, pady=(14, 0))
        ttk.Label(mode_panel, text="Modes", style="Section.TLabel").pack(anchor=tk.W)
        self._combo(mode_panel, "Measurement", self.measurement_type, ("position", "range_rate"), self.on_sim_mode)
        self._combo(mode_panel, "Start", self.start_mode, ("cold", "hot", "formal", "sqrt_formal"), self.on_sim_mode)
        self._combo(mode_panel, "Network", self.scenario_network, ("single", "multi"), self.on_sim_mode)

        self.telemetry_panel = self.panel(right)
        self.telemetry_panel.pack(fill=tk.X, pady=(14, 0))
        ttk.Label(self.telemetry_panel, text="Frame telemetry", style="Section.TLabel").pack(anchor=tk.W)
        self.telemetry_labels: dict[str, ttk.Label] = {}
        for key in ("Frame", "Epoch", "Range", "Range-rate", "Visible", "Best station"):
            line = ttk.Frame(self.telemetry_panel, style="Panel.TFrame")
            line.pack(fill=tk.X, pady=4)
            ttk.Label(line, text=key, style="MutedPanel.TLabel").pack(side=tk.LEFT)
            value = ttk.Label(line, text="-", style="Panel.TLabel", font=("Consolas", 10, "bold"))
            value.pack(side=tk.RIGHT)
            self.telemetry_labels[key] = value

    def _combo(
        self,
        parent: tk.Widget,
        label: str,
        variable: tk.StringVar,
        values: tuple[str, ...],
        callback: Any | None = None,
    ) -> ttk.Combobox:
        ttk.Label(parent, text=label, style="MutedPanel.TLabel").pack(anchor=tk.W, pady=(10, 2))
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.pack(fill=tk.X)
        if callback is not None:
            combo.bind("<<ComboboxSelected>>", lambda _event: callback())
        return combo

    def get_frame_data(self, frame: int | None = None) -> dict[str, Any]:
        if frame is None:
            frame = self.sim_frame.get()
        t = frame / max(1, self.sim_frame_count - 1)
        craft_lon = -175.0 + 350.0 * t
        craft_lat = 24.0 * math.sin(t * math.tau * 2.35 - 0.6)
        phase = t * math.tau
        start = self.start_mode.get()
        base_range = {"cold": 386400.0, "hot": 384250.0, "formal": 385120.0, "sqrt_formal": 385000.0}.get(start, 386400.0)
        range_km = base_range + 1450.0 * math.sin(phase * 1.25)
        rr_scale = 920.0 if self.measurement_type.get() == "range_rate" else 360.0
        range_rate = rr_scale * math.cos(phase * 1.25)
        network_names = set(NETWORKS.get(self.scenario_network.get(), NETWORKS["multi"]))
        network_stations = [s for s in self.stations if s.name in network_names] or self.stations
        candidates = []
        for station in network_stations:
            distance = angular_distance_deg(station.lat_deg, station.lon_deg, craft_lat, craft_lon)
            has_mode = self.measurement_type.get() != "range_rate" or station.supports_range_rate
            candidates.append((station, distance, has_mode and distance < 78.0))
        candidates.sort(key=lambda item: item[1])
        visible = [item for item in candidates if item[2]][:4]
        return {
            "t": t,
            "craft_lon": craft_lon,
            "craft_lat": craft_lat,
            "range_km": range_km,
            "range_rate": range_rate,
            "candidates": candidates,
            "visible": visible,
        }

    def draw_simulation(self) -> None:
        canvas = self.sim_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 860)
        height = max(canvas.winfo_height(), 560)
        frame = self.sim_frame.get()
        data = self.get_frame_data(frame)
        map_x, map_y = 28, 58
        map_w, map_h = int(width * 0.66), int(height * 0.55)
        canvas.create_rectangle(0, 0, width, height, fill=COLORS["panel_2"], outline="")
        canvas.create_text(28, 24, text="Lunar tracking playback", fill=COLORS["ink"], anchor=tk.W, font=("Segoe UI", 15, "bold"))
        canvas.create_text(
            28,
            44,
            text=f"Frame {frame + 1:03d}/{self.sim_frame_count}  |  {self.measurement_type.get()}  |  {self.start_mode.get()} start",
            fill=COLORS["muted"],
            anchor=tk.W,
            font=("Segoe UI", 9),
        )

        map_x, map_y, map_w, map_h = self.draw_map_background(canvas, map_x, map_y, map_w, map_h)
        track_points = []
        for i in range(80):
            tt = i / 79
            lon = -175 + 350 * tt
            lat = 24 * math.sin(tt * math.tau * 2.35 - 0.6)
            track_points.extend((map_x + lon_to_x(lon, map_w), map_y + lat_to_y(lat, map_h)))
        canvas.create_line(*track_points, fill="#2b8bca", width=2, smooth=True)

        sx = map_x + lon_to_x(data["craft_lon"], map_w)
        sy = map_y + lat_to_y(data["craft_lat"], map_h)
        for station, _distance, _is_visible in data["visible"]:
            px = map_x + lon_to_x(station.lon_deg, map_w)
            py = map_y + lat_to_y(station.lat_deg, map_h)
            canvas.create_line(px, py, sx, sy, fill=FAMILY_COLORS[station.family], width=2, dash=(7, 6))
        for station in self.stations:
            active = any(item[0].name == station.name for item in data["visible"])
            color = FAMILY_COLORS[station.family]
            px = map_x + lon_to_x(station.lon_deg, map_w)
            py = map_y + lat_to_y(station.lat_deg, map_h)
            radius = 5 if active else 3
            canvas.create_oval(px - radius, py - radius, px + radius, py + radius, fill=color, outline="")

        offset = {"cold": 32, "hot": 12, "formal": 20, "sqrt_formal": 18}.get(self.start_mode.get(), 32)
        est_x = sx + offset * math.cos(data["t"] * math.tau * 2.5)
        est_y = sy + offset * math.sin(data["t"] * math.tau * 2.0)
        canvas.create_oval(est_x - 7, est_y - 7, est_x + 7, est_y + 7, outline=COLORS["yellow"], width=2, dash=(3, 3))
        canvas.create_oval(sx - 12, sy - 12, sx + 12, sy + 12, fill="#123a55", outline=COLORS["cyan"], width=2)
        canvas.create_oval(sx - 4, sy - 4, sx + 4, sy + 4, fill=COLORS["cyan"], outline="")

        moon_cx, moon_cy = int(width * 0.80), int(height * 0.36)
        canvas.create_text(moon_cx - 95, moon_cy - 116, text="Lunar phase frame", fill=COLORS["muted"], anchor=tk.W, font=("Segoe UI", 9))
        canvas.create_oval(moon_cx - 70, moon_cy - 70, moon_cx + 70, moon_cy + 70, fill="#8d9bb1", outline="#cbd5e1", width=2)
        canvas.create_oval(moon_cx - 28, moon_cy - 34, moon_cx - 8, moon_cy - 14, fill="#6b768c", outline="")
        canvas.create_oval(moon_cx + 20, moon_cy + 22, moon_cx + 48, moon_cy + 50, fill="#6b768c", outline="")
        canvas.create_oval(moon_cx - 120, moon_cy - 82, moon_cx + 120, moon_cy + 82, outline="#2f86c6", width=2)
        orb_x = moon_cx + math.cos(data["t"] * math.tau) * 120
        orb_y = moon_cy + math.sin(data["t"] * math.tau) * 82
        canvas.create_line(moon_cx, moon_cy, orb_x, orb_y, fill="#1d637f")
        canvas.create_oval(orb_x - 7, orb_y - 7, orb_x + 7, orb_y + 7, fill=COLORS["cyan"], outline="")

        self.telemetry_labels["Frame"].configure(text=f"{frame + 1:03d}/{self.sim_frame_count}")
        self.telemetry_labels["Epoch"].configure(text=f"T+{data['t'] * self.duration_h.get():.2f} h")
        self.telemetry_labels["Range"].configure(text=f"{data['range_km']:.0f} km")
        self.telemetry_labels["Range-rate"].configure(text=f"{data['range_rate']:.2f} m/s")
        self.telemetry_labels["Visible"].configure(text=str(len(data["visible"])))
        best = data["candidates"][0][0].name if data["candidates"] else "-"
        self.telemetry_labels["Best station"].configure(text=best)

    def draw_map_background(self, canvas: tk.Canvas, x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
        if self.world_map_image is not None:
            image = self.scaled_world_map(w, h)
            img_w = image.width()
            img_h = image.height()
            draw_x = int(x + (w - img_w) / 2)
            draw_y = int(y + (h - img_h) / 2)
            canvas.create_rectangle(draw_x - 1, draw_y - 1, draw_x + img_w + 1, draw_y + img_h + 1, fill="#0d1d3c", outline=COLORS["line"], width=1)
            canvas.create_image(draw_x, draw_y, image=image, anchor=tk.NW)
            for lon in range(-150, 181, 30):
                px = draw_x + lon_to_x(lon, img_w)
                canvas.create_line(px, draw_y, px, draw_y + img_h, fill="#2a5d73")
            for lat in range(-60, 61, 20):
                py = draw_y + lat_to_y(lat, img_h)
                canvas.create_line(draw_x, py, draw_x + img_w, py, fill="#2a5d73")
            canvas.create_line(draw_x, draw_y + img_h / 2, draw_x + img_w, draw_y + img_h / 2, fill="#2f5477", dash=(8, 6))
            return draw_x, draw_y, img_w, img_h

        canvas.create_rectangle(x, y, x + w, y + h, fill="#0d1d3c", outline=COLORS["line"], width=1)
        for lon in range(-150, 181, 30):
            px = x + lon_to_x(lon, w)
            canvas.create_line(px, y, px, y + h, fill="#1d2e48")
        for lat in range(-60, 61, 20):
            py = y + lat_to_y(lat, h)
            canvas.create_line(x, py, x + w, py, fill="#1d2e48")
        canvas.create_line(x, y + h / 2, x + w, y + h / 2, fill="#2f5477", dash=(8, 6))
        land = "#1a2b43"
        outline = "#334762"
        shapes = [
            [(0.07, 0.24), (0.16, 0.10), (0.27, 0.18), (0.30, 0.33), (0.20, 0.43), (0.09, 0.36)],
            [(0.23, 0.53), (0.30, 0.55), (0.33, 0.73), (0.25, 0.90), (0.20, 0.74)],
            [(0.43, 0.22), (0.54, 0.14), (0.62, 0.30), (0.56, 0.42), (0.45, 0.37)],
            [(0.48, 0.42), (0.58, 0.39), (0.62, 0.60), (0.55, 0.80), (0.48, 0.66)],
            [(0.57, 0.18), (0.76, 0.13), (0.91, 0.31), (0.80, 0.45), (0.62, 0.37)],
            [(0.78, 0.66), (0.89, 0.62), (0.94, 0.75), (0.84, 0.86), (0.76, 0.76)],
        ]
        for shape in shapes:
            points = []
            for px, py in shape:
                points.extend((x + px * w, y + py * h))
            canvas.create_polygon(*points, fill=land, outline=outline, smooth=True)
        return x, y, w, h

    def scaled_world_map(self, target_w: int, target_h: int) -> tk.PhotoImage:
        assert self.world_map_image is not None
        src_w = self.world_map_image.width()
        src_h = self.world_map_image.height()
        scale = max(src_w / max(1, target_w), src_h / max(1, target_h))
        subsample = max(1, int(math.ceil(scale)))
        if target_h >= 340 and subsample > 2:
            subsample = 2
        if subsample not in self.map_image_cache:
            self.map_image_cache[subsample] = self.world_map_image.subsample(subsample, subsample)
        return self.map_image_cache[subsample]

    def set_sim_frame(self, frame: int) -> None:
        frame = max(0, min(self.sim_frame_count - 1, frame))
        if frame != self.sim_frame.get():
            self.sim_frame.set(frame)
        if hasattr(self, "frame_scale") and not self._syncing_sim_scale:
            self._syncing_sim_scale = True
            try:
                self.frame_scale.set(frame)
            finally:
                self._syncing_sim_scale = False
        self.draw_simulation()

    def on_sim_scale(self, value: str) -> None:
        if self._syncing_sim_scale:
            return
        frame = max(0, min(self.sim_frame_count - 1, int(float(value))))
        if frame != self.sim_frame.get():
            self.sim_frame.set(frame)
        self.draw_simulation()

    def step_sim(self, delta: int) -> None:
        self.set_sim_frame((self.sim_frame.get() + delta) % self.sim_frame_count)

    def toggle_play(self) -> None:
        self.sim_playing = not self.sim_playing
        self.play_button.configure(text="Pause" if self.sim_playing else "Play")
        if self.sim_playing:
            self.after(380, self._play_tick)

    def _play_tick(self) -> None:
        if not self.sim_playing:
            return
        self.step_sim(1)
        self.after(380, self._play_tick)

    def on_sim_mode(self) -> None:
        self.draw_simulation()
        if hasattr(self, "analysis_canvas"):
            self.draw_analysis()
        self.refresh_scenario_preview()
        self.refresh_dashboard()

    def _build_analysis_tab(self) -> None:
        left = self.panel(self.analysis_tab, padding=12)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(self.analysis_tab)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(16, 0))

        ttk.Label(left, text="Ground-track and visibility analysis", style="Section.TLabel").pack(anchor=tk.W)
        self.analysis_canvas = tk.Canvas(left, width=880, height=610, bg=COLORS["panel_2"], highlightthickness=0)
        self.analysis_canvas.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        plot_panel = self.panel(right)
        plot_panel.pack(fill=tk.X)
        self.segmented(
            plot_panel,
            "Plot",
            self.analysis_plot,
            (("Ground track", "ground_track"), ("Visibility", "visibility")),
            self.draw_analysis,
        )

        mode_panel = self.panel(right)
        mode_panel.pack(fill=tk.X, pady=(14, 0))
        self.segmented(
            mode_panel,
            "Mode",
            self.analysis_mode,
            (("Static", "static"), ("Animated", "animated")),
            self.on_analysis_mode,
        )

        playback = self.panel(right)
        playback.pack(fill=tk.X, pady=(14, 0))
        ttk.Label(playback, text="Animation frame", style="Section.TLabel").pack(anchor=tk.W)
        row = ttk.Frame(playback, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=(10, 4))
        ttk.Button(row, text="<", command=lambda: self.step_analysis(-1)).pack(side=tk.LEFT, padx=(0, 6))
        self.analysis_play_button = ttk.Button(row, text="Play", command=self.toggle_analysis_play, style="Accent.TButton")
        self.analysis_play_button.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text=">", command=lambda: self.step_analysis(1)).pack(side=tk.LEFT)
        self.analysis_scale = ttk.Scale(
            playback,
            from_=0,
            to=self.sim_frame_count - 1,
            orient=tk.HORIZONTAL,
            command=self.on_analysis_scale,
        )
        self.analysis_scale.pack(fill=tk.X, pady=(8, 0))

        summary = self.panel(right)
        summary.pack(fill=tk.X, pady=(14, 0))
        ttk.Label(summary, text="Analysis summary", style="Section.TLabel").pack(anchor=tk.W)
        self.analysis_summary_labels: dict[str, ttk.Label] = {}
        for key in ("Frame", "Network", "Visible stations", "Coverage", "Best station"):
            line = ttk.Frame(summary, style="Panel.TFrame")
            line.pack(fill=tk.X, pady=4)
            ttk.Label(line, text=key, style="MutedPanel.TLabel").pack(side=tk.LEFT)
            value = ttk.Label(line, text="-", style="Panel.TLabel", font=("Consolas", 10, "bold"))
            value.pack(side=tk.RIGHT)
            self.analysis_summary_labels[key] = value

    def selected_analysis_stations(self) -> list[StationView]:
        names = set(NETWORKS.get(self.scenario_network.get(), NETWORKS["multi"]))
        return [station for station in self.stations if station.name in names] or self.stations

    def station_visible_at_frame(self, station: StationView, frame: int) -> bool:
        data = self.get_frame_data(frame)
        has_mode = self.measurement_type.get() != "range_rate" or station.supports_range_rate
        distance = angular_distance_deg(station.lat_deg, station.lon_deg, data["craft_lat"], data["craft_lon"])
        return has_mode and distance < 78.0

    def frame_visible_stations(self, frame: int) -> list[StationView]:
        return [station for station in self.selected_analysis_stations() if self.station_visible_at_frame(station, frame)]

    def draw_analysis(self) -> None:
        if not hasattr(self, "analysis_canvas"):
            return
        if self.analysis_plot.get() == "visibility":
            self.draw_visibility_analysis()
        else:
            self.draw_ground_track_analysis()

    def draw_ground_track_analysis(self) -> None:
        canvas = self.analysis_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 880)
        height = max(canvas.winfo_height(), 610)
        frame = self.analysis_frame.get()
        animated = self.analysis_mode.get() == "animated"
        canvas.create_rectangle(0, 0, width, height, fill=COLORS["panel_2"], outline="")
        title = "Animated ground track" if animated else "Static ground track"
        canvas.create_text(28, 24, text=title, fill=COLORS["ink"], anchor=tk.W, font=("Segoe UI", 15, "bold"))
        canvas.create_text(
            28,
            45,
            text=f"{self.scenario_network.get()} network | {self.measurement_type.get()} | {self.start_mode.get()} start",
            fill=COLORS["muted"],
            anchor=tk.W,
            font=("Segoe UI", 9),
        )
        map_x, map_y, map_w, map_h = self.draw_map_background(canvas, 28, 66, width - 56, int(height * 0.62))
        max_index = frame if animated else self.sim_frame_count - 1
        track_points: list[float] = []
        for i in range(max_index + 1):
            data = self.get_frame_data(i)
            track_points.extend((map_x + lon_to_x(data["craft_lon"], map_w), map_y + lat_to_y(data["craft_lat"], map_h)))
        if len(track_points) >= 4:
            canvas.create_line(*track_points, fill="#0b5f8e", width=5, smooth=True)
            canvas.create_line(*track_points, fill=COLORS["cyan"], width=2, smooth=True)
        for i in range(0, max_index + 1, 8):
            data = self.get_frame_data(i)
            px = map_x + lon_to_x(data["craft_lon"], map_w)
            py = map_y + lat_to_y(data["craft_lat"], map_h)
            canvas.create_oval(px - 2, py - 2, px + 2, py + 2, fill=COLORS["cyan"], outline="")

        current = self.get_frame_data(frame if animated else self.sim_frame_count - 1)
        cx = map_x + lon_to_x(current["craft_lon"], map_w)
        cy = map_y + lat_to_y(current["craft_lat"], map_h)
        visible = self.frame_visible_stations(frame)
        for station in self.selected_analysis_stations():
            sx = map_x + lon_to_x(station.lon_deg, map_w)
            sy = map_y + lat_to_y(station.lat_deg, map_h)
            is_visible = station in visible
            color = FAMILY_COLORS[station.family]
            radius = 7 if is_visible else 4
            if is_visible and animated:
                canvas.create_line(sx, sy, cx, cy, fill=color, width=2, dash=(7, 6))
            canvas.create_oval(sx - radius, sy - radius, sx + radius, sy + radius, fill=color if is_visible else "#52627a", outline="")
            canvas.create_text(sx + 10, sy - 10, text=station.name, fill=COLORS["ink"] if is_visible else COLORS["muted"], anchor=tk.W, font=("Segoe UI", 8))
        canvas.create_oval(cx - 12, cy - 12, cx + 12, cy + 12, fill="#123a55", outline=COLORS["cyan"], width=2)
        canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=COLORS["cyan"], outline="")
        self.draw_ground_track_mini_profile(canvas, 28, map_y + map_h + 28, width - 56, height - (map_y + map_h + 48), max_index, animated)
        self.update_analysis_summary(frame, visible)

    def draw_ground_track_mini_profile(self, canvas: tk.Canvas, x: int, y: int, w: int, h: int, max_index: int, animated: bool) -> None:
        if h < 80:
            return
        canvas.create_text(x, y, text="Latitude / longitude profile", fill=COLORS["muted"], anchor=tk.W, font=("Segoe UI", 9, "bold"))
        plot_y = y + 18
        plot_h = h - 26
        canvas.create_rectangle(x, plot_y, x + w, plot_y + plot_h, fill=COLORS["panel"], outline=COLORS["line"])
        for frac in (0.25, 0.5, 0.75):
            px = x + w * frac
            canvas.create_line(px, plot_y, px, plot_y + plot_h, fill="#1d2e48")
        points_lat: list[float] = []
        points_lon: list[float] = []
        for i in range(max_index + 1):
            data = self.get_frame_data(i)
            px = x + (i / max(1, self.sim_frame_count - 1)) * w
            lat_y = plot_y + plot_h * (1 - ((data["craft_lat"] + 40) / 80))
            lon_y = plot_y + plot_h * (1 - ((data["craft_lon"] + 180) / 360))
            points_lat.extend((px, lat_y))
            points_lon.extend((px, lon_y))
        if len(points_lat) >= 4:
            canvas.create_line(*points_lon, fill=COLORS["yellow"], width=2, smooth=True)
            canvas.create_line(*points_lat, fill=COLORS["cyan"], width=2, smooth=True)
        if animated:
            cursor_x = x + (max_index / max(1, self.sim_frame_count - 1)) * w
            canvas.create_line(cursor_x, plot_y, cursor_x, plot_y + plot_h, fill=COLORS["red"], width=2)
        canvas.create_text(x + 12, plot_y + 14, text="lat", fill=COLORS["cyan"], anchor=tk.W, font=("Segoe UI", 8, "bold"))
        canvas.create_text(x + 48, plot_y + 14, text="lon", fill=COLORS["yellow"], anchor=tk.W, font=("Segoe UI", 8, "bold"))

    def draw_visibility_analysis(self) -> None:
        canvas = self.analysis_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 880)
        height = max(canvas.winfo_height(), 610)
        frame = self.analysis_frame.get()
        animated = self.analysis_mode.get() == "animated"
        stations = self.selected_analysis_stations()
        canvas.create_rectangle(0, 0, width, height, fill=COLORS["panel_2"], outline="")
        title = "Animated visibility analysis" if animated else "Static visibility analysis"
        canvas.create_text(28, 24, text=title, fill=COLORS["ink"], anchor=tk.W, font=("Segoe UI", 15, "bold"))
        canvas.create_text(
            28,
            45,
            text=f"Green bars show station visibility. Range-rate mode hides stations without RR support.",
            fill=COLORS["muted"],
            anchor=tk.W,
            font=("Segoe UI", 9),
        )
        x0, y0 = 170, 92
        plot_w = width - x0 - 42
        row_h = min(42, max(26, int((height - 205) / max(1, len(stations)))))
        plot_h = row_h * len(stations)
        canvas.create_rectangle(x0, y0, x0 + plot_w, y0 + plot_h, fill=COLORS["panel"], outline=COLORS["line"])
        for tick in range(0, self.sim_frame_count, 12):
            px = x0 + (tick / (self.sim_frame_count - 1)) * plot_w
            canvas.create_line(px, y0, px, y0 + plot_h, fill="#1d2e48")
            canvas.create_text(px, y0 + plot_h + 12, text=f"{tick}", fill=COLORS["muted"], font=("Segoe UI", 8))
        visible_counts = [0] * self.sim_frame_count
        for row, station in enumerate(stations):
            y = y0 + row * row_h
            canvas.create_text(28, y + row_h / 2, text=station.name, fill=COLORS["ink"], anchor=tk.W, font=("Segoe UI", 9))
            canvas.create_line(x0, y, x0 + plot_w, y, fill="#1d2e48")
            last_x: float | None = None
            seg_start: float | None = None
            for i in range(self.sim_frame_count):
                visible = self.station_visible_at_frame(station, i)
                if visible:
                    visible_counts[i] += 1
                    px = x0 + (i / (self.sim_frame_count - 1)) * plot_w
                    if seg_start is None:
                        seg_start = px
                    last_x = px
                elif seg_start is not None and last_x is not None:
                    canvas.create_rectangle(seg_start, y + 8, last_x + 4, y + row_h - 8, fill=FAMILY_COLORS[station.family], outline="")
                    seg_start = None
                    last_x = None
            if seg_start is not None and last_x is not None:
                canvas.create_rectangle(seg_start, y + 8, last_x + 4, y + row_h - 8, fill=FAMILY_COLORS[station.family], outline="")

        coverage_y = y0 + plot_h + 48
        coverage_h = min(90, height - coverage_y - 28)
        canvas.create_text(28, coverage_y - 14, text="Network visible station count", fill=COLORS["muted"], anchor=tk.W, font=("Segoe UI", 9, "bold"))
        canvas.create_rectangle(x0, coverage_y, x0 + plot_w, coverage_y + coverage_h, fill=COLORS["panel"], outline=COLORS["line"])
        max_count = max(1, max(visible_counts))
        count_points: list[float] = []
        for i, count in enumerate(visible_counts):
            px = x0 + (i / (self.sim_frame_count - 1)) * plot_w
            py = coverage_y + coverage_h * (1 - count / max_count)
            count_points.extend((px, py))
        if len(count_points) >= 4:
            canvas.create_line(*count_points, fill=COLORS["cyan"], width=2, smooth=True)
        if animated:
            cursor_x = x0 + (frame / (self.sim_frame_count - 1)) * plot_w
            canvas.create_line(cursor_x, y0, cursor_x, coverage_y + coverage_h, fill=COLORS["red"], width=2)
            for row, station in enumerate(stations):
                if self.station_visible_at_frame(station, frame):
                    y = y0 + row * row_h
                    canvas.create_rectangle(x0 - 9, y + 9, x0 - 3, y + row_h - 9, fill=COLORS["green"], outline="")
        visible = self.frame_visible_stations(frame)
        self.update_analysis_summary(frame, visible, visible_counts)

    def update_analysis_summary(self, frame: int, visible: list[StationView], visible_counts: list[int] | None = None) -> None:
        if not hasattr(self, "analysis_summary_labels"):
            return
        if visible_counts is None:
            visible_counts = [len(self.frame_visible_stations(i)) for i in range(self.sim_frame_count)]
        coverage = sum(1 for count in visible_counts if count > 0) / max(1, len(visible_counts))
        best = visible[0].name if visible else "-"
        self.analysis_summary_labels["Frame"].configure(text=f"{frame + 1:03d}/{self.sim_frame_count}")
        self.analysis_summary_labels["Network"].configure(text=self.scenario_network.get())
        self.analysis_summary_labels["Visible stations"].configure(text=str(len(visible)))
        self.analysis_summary_labels["Coverage"].configure(text=f"{coverage * 100:.1f}%")
        self.analysis_summary_labels["Best station"].configure(text=best)

    def on_analysis_mode(self) -> None:
        if self.analysis_mode.get() == "static" and self.analysis_playing:
            self.toggle_analysis_play()
        self.draw_analysis()

    def set_analysis_frame(self, frame: int) -> None:
        frame = max(0, min(self.sim_frame_count - 1, frame))
        if frame != self.analysis_frame.get():
            self.analysis_frame.set(frame)
        if hasattr(self, "analysis_scale") and not self._syncing_analysis_scale:
            self._syncing_analysis_scale = True
            try:
                self.analysis_scale.set(frame)
            finally:
                self._syncing_analysis_scale = False
        self.draw_analysis()

    def on_analysis_scale(self, value: str) -> None:
        if self._syncing_analysis_scale:
            return
        frame = max(0, min(self.sim_frame_count - 1, int(float(value))))
        if frame != self.analysis_frame.get():
            self.analysis_frame.set(frame)
        self.draw_analysis()

    def step_analysis(self, delta: int) -> None:
        self.set_analysis_frame((self.analysis_frame.get() + delta) % self.sim_frame_count)

    def toggle_analysis_play(self) -> None:
        if self.analysis_mode.get() != "animated":
            self.analysis_mode.set("animated")
            self.refresh_segmented_controls()
        self.analysis_playing = not self.analysis_playing
        self.analysis_play_button.configure(text="Pause" if self.analysis_playing else "Play")
        if self.analysis_playing:
            self.after(420, self._analysis_play_tick)
        self.draw_analysis()

    def _analysis_play_tick(self) -> None:
        if not self.analysis_playing:
            return
        self.step_analysis(1)
        self.after(420, self._analysis_play_tick)

    def _build_stations_tab(self) -> None:
        left = self.panel(self.stations_tab, padding=12)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(self.stations_tab)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(16, 0))
        ttk.Label(left, text="Stations and networks", style="Section.TLabel").pack(anchor=tk.W)
        self.station_canvas = tk.Canvas(left, width=850, height=470, bg=COLORS["panel_2"], highlightthickness=0)
        self.station_canvas.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        controls = self.panel(right)
        controls.pack(fill=tk.X)
        self.segmented(
            controls,
            "Network focus",
            self.selected_network,
            (("All", "all"), ("Single", "single"), ("Multi", "multi")),
            self.refresh_station_tab,
        )
        self.station_summary = ttk.Label(controls, text="", style="MutedPanel.TLabel", wraplength=320)
        self.station_summary.pack(fill=tk.X, pady=(12, 0))

        table_panel = self.panel(right)
        table_panel.pack(fill=tk.BOTH, expand=True, pady=(14, 0))
        ttk.Label(table_panel, text="Station catalog", style="Section.TLabel").pack(anchor=tk.W)
        columns = ("name", "family", "lat", "lon", "rr")
        self.station_tree = ttk.Treeview(table_panel, columns=columns, show="headings", height=15)
        for col, label, width in (
            ("name", "Station", 150),
            ("family", "Family", 70),
            ("lat", "Lat", 72),
            ("lon", "Lon", 72),
            ("rr", "RR", 52),
        ):
            self.station_tree.heading(col, text=label)
            self.station_tree.column(col, width=width, anchor=tk.CENTER)
        self.station_tree.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def refresh_station_tab(self) -> None:
        self.draw_station_map()
        self.populate_station_table()

    def draw_station_map(self) -> None:
        canvas = self.station_canvas
        canvas.delete("all")
        w = max(canvas.winfo_width(), 850)
        h = max(canvas.winfo_height(), 470)
        margin = 28
        map_w = w - margin * 2
        map_h = h - margin * 2 - 40
        canvas.create_rectangle(0, 0, w, h, fill=COLORS["panel_2"], outline="")
        canvas.create_text(margin, 18, text="Tracking network map", fill=COLORS["ink"], anchor=tk.W, font=("Segoe UI", 14, "bold"))
        map_x, map_y, map_w, map_h = self.draw_map_background(canvas, margin, 48, map_w, map_h)

        focus = self.selected_network.get()
        active_names = set()
        if focus in NETWORKS:
            active_names = set(NETWORKS[focus])
        visible_stations = [s for s in self.stations if not active_names or s.name in active_names]
        for i, station in enumerate(visible_stations):
            if i == len(visible_stations) - 1 or len(visible_stations) < 2:
                continue
            nxt = visible_stations[i + 1]
            x1 = map_x + lon_to_x(station.lon_deg, map_w)
            y1 = map_y + lat_to_y(station.lat_deg, map_h)
            x2 = map_x + lon_to_x(nxt.lon_deg, map_w)
            y2 = map_y + lat_to_y(nxt.lat_deg, map_h)
            canvas.create_line(x1, y1, x2, y2, fill="#2f86c6", width=2, dash=(7, 7))

        for station in self.stations:
            active = not active_names or station.name in active_names
            color = FAMILY_COLORS[station.family]
            x = map_x + lon_to_x(station.lon_deg, map_w)
            y = map_y + lat_to_y(station.lat_deg, map_h)
            radius = 7 if active else 4
            fill = color if active else "#52627a"
            if station.supports_range_rate and active:
                canvas.create_oval(x - 13, y - 13, x + 13, y + 13, outline=color, width=1, dash=(3, 3))
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline="")
            if active:
                canvas.create_text(x + 10, y - 10, text=station.name, fill=COLORS["ink"], anchor=tk.W, font=("Segoe UI", 8))

        legend_x = margin
        legend_y = h - 24
        for family, color in FAMILY_COLORS.items():
            if family == "Other":
                continue
            canvas.create_oval(legend_x, legend_y - 5, legend_x + 10, legend_y + 5, fill=color, outline="")
            canvas.create_text(legend_x + 16, legend_y, text=family, fill=COLORS["muted"], anchor=tk.W, font=("Segoe UI", 8))
            legend_x += 78
        rr_count = sum(1 for s in visible_stations if s.supports_range_rate)
        self.station_summary.configure(
            text=f"Shown: {len(visible_stations)} station(s)\nRange-rate capable: {rr_count}\nFocus: {focus}"
        )
        self.populate_station_table()

    def populate_station_table(self) -> None:
        if not hasattr(self, "station_tree"):
            return
        self.station_tree.delete(*self.station_tree.get_children())
        focus = self.selected_network.get()
        active_names = set(NETWORKS.get(focus, ()))
        rows = [s for s in self.stations if not active_names or s.name in active_names]
        for station in rows:
            item = self.station_tree.insert(
                "",
                tk.END,
                values=(
                    station.name,
                    station.family,
                    f"{station.lat_deg:.2f}",
                    f"{station.lon_deg:.2f}",
                    "yes" if station.supports_range_rate else "no",
                ),
            )

    def _build_scenario_tab(self) -> None:
        left = self.panel(self.scenario_tab, padding=14)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(self.scenario_tab)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(16, 0))
        ttk.Label(left, text="Scenario builder", style="Section.TLabel").pack(anchor=tk.W)

        presets = ttk.Frame(left, style="Panel.TFrame")
        presets.pack(fill=tk.X, pady=(10, 8))
        for text, key in (
            ("Cold + Range", "cold_range"),
            ("Hot + RR", "hot_rr"),
            ("Formal + RR", "formal_rr"),
            ("UKF + RR", "ukf_rr"),
            ("Two-way UKF", "twoway_ukf"),
            ("Sqrt + Bias", "sqrt_bias"),
        ):
            ttk.Button(presets, text=text, command=lambda k=key: self.apply_preset(k)).pack(side=tk.LEFT, padx=(0, 8))

        self._entry(left, "Scenario name", self.scenario_name)
        self._combo(left, "Measurement", self.measurement_type, ("position", "range_rate"), self.refresh_scenario_preview)
        self._combo(left, "Estimator", self.estimator_type, ("bls_lm", "srif", "ukf"), self.refresh_scenario_preview)
        self._combo(left, "Start mode", self.start_mode, ("cold", "hot", "formal", "sqrt_formal"), self.refresh_scenario_preview)
        self._combo(left, "Network", self.scenario_network, ("single", "multi"), self.refresh_scenario_preview)

        grid = ttk.Frame(left, style="Panel.TFrame")
        grid.pack(fill=tk.X, pady=(12, 0))
        self._entry(grid, "Duration h", self.duration_h, row=0, col=0)
        self._entry(grid, "Sample step s", self.sample_step_s, row=0, col=1)
        self._entry(grid, "Max iter", self.max_iter, row=1, col=0)
        self._entry(grid, "rtol", self.rtol, row=1, col=1)
        self._entry(grid, "atol", self.atol, row=2, col=0)
        bias_holder = ttk.Frame(grid, style="Panel.TFrame")
        bias_holder.grid(row=3, column=0, sticky="ew", padx=(0, 8), pady=(0, 8))
        ttk.Label(bias_holder, text="Bias mode", style="MutedPanel.TLabel").pack(anchor=tk.W, pady=(0, 2))
        bias_combo = ttk.Combobox(
            bias_holder,
            textvariable=self.bias_mode,
            values=("none", "global", "station_angles", "station_full"),
            state="readonly",
        )
        bias_combo.pack(fill=tk.X)
        bias_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_scenario_preview())
        ttk.Checkbutton(grid, text="Noise enabled", variable=self.noise_enabled, command=self.refresh_scenario_preview).grid(row=3, column=1, sticky="w")
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        radio_grid = ttk.LabelFrame(left, text="Radiometrics")
        radio_grid.pack(fill=tk.X, pady=(12, 0))
        physics_holder = ttk.Frame(radio_grid)
        physics_holder.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(physics_holder, text="Range-rate physics", style="MutedPanel.TLabel").pack(anchor=tk.W, pady=(0, 2))
        physics_combo = ttk.Combobox(
            physics_holder,
            textvariable=self.range_rate_physics,
            values=("geometric_instantaneous", "two_way_counted_doppler"),
            state="readonly",
        )
        physics_combo.pack(fill=tk.X)
        physics_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_scenario_preview())
        self._entry(radio_grid, "Count interval s", self.count_interval_s, row=0, col=1)
        self._entry(radio_grid, "Uplink Hz", self.uplink_frequency_hz, row=1, col=0)
        self._entry(radio_grid, "Turnaround ratio", self.turnaround_ratio, row=1, col=1)
        local_model_holder = ttk.Frame(radio_grid)
        local_model_holder.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(local_model_holder, text="Two-way local state model", style="MutedPanel.TLabel").pack(anchor=tk.W, pady=(0, 2))
        local_model_combo = ttk.Combobox(
            local_model_holder,
            textvariable=self.two_way_local_state_model,
            values=("ode", "taylor3"),
            state="readonly",
        )
        local_model_combo.pack(fill=tk.X)
        local_model_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_scenario_preview())
        self._entry(radio_grid, "Clock offset s", self.station_clock_offset_s, row=2, col=1)
        self._entry(radio_grid, "Clock drift", self.station_clock_drift, row=3, col=0)
        self._entry(radio_grid, "Clock reference s", self.clock_reference_time_s, row=3, col=1)
        self._entry(radio_grid, "Transponder delay s", self.transponder_delay_s, row=4, col=0)
        radio_grid.columnconfigure(0, weight=1)
        radio_grid.columnconfigure(1, weight=1)

        ukf_grid = ttk.LabelFrame(left, text="UKF tuning")
        ukf_grid.pack(fill=tk.X, pady=(12, 0))
        self._entry(ukf_grid, "Alpha", self.ukf_alpha, row=0, col=0)
        self._entry(ukf_grid, "Beta", self.ukf_beta, row=0, col=1)
        self._entry(ukf_grid, "Kappa", self.ukf_kappa, row=1, col=0)
        self._entry(ukf_grid, "Covariance inflation", self.ukf_covariance_inflation, row=1, col=1)
        self._entry(ukf_grid, "NIS gate (blank = off)", self.ukf_nis_gate, row=2, col=0)
        ttk.Checkbutton(
            ukf_grid,
            text="Adaptive measurement noise",
            variable=self.ukf_adaptive_measurement_noise,
            command=self.refresh_scenario_preview,
        ).grid(row=2, column=1, sticky="w")
        self._entry(ukf_grid, "Component NIS gate", self.ukf_component_nis_gate, row=3, col=0)
        covariance_holder = ttk.Frame(ukf_grid)
        covariance_holder.grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(0, 8))
        ttk.Label(covariance_holder, text="Covariance form", style="MutedPanel.TLabel").pack(anchor=tk.W, pady=(0, 2))
        covariance_combo = ttk.Combobox(
            covariance_holder,
            textvariable=self.ukf_covariance_form,
            values=("standard", "square_root"),
            state="readonly",
        )
        covariance_combo.pack(fill=tk.X)
        covariance_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_scenario_preview())
        gate_mode_holder = ttk.Frame(ukf_grid)
        gate_mode_holder.grid(row=5, column=1, sticky="ew", padx=(8, 0), pady=(0, 8))
        ttk.Label(gate_mode_holder, text="Component gate mode", style="MutedPanel.TLabel").pack(anchor=tk.W, pady=(0, 2))
        gate_mode_combo = ttk.Combobox(
            gate_mode_holder,
            textvariable=self.ukf_component_gate_mode,
            values=("marginal", "conditional"),
            state="readonly",
        )
        gate_mode_combo.pack(fill=tk.X)
        gate_mode_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_scenario_preview())
        process_holder = ttk.Frame(ukf_grid)
        process_holder.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(process_holder, text="Process noise model", style="MutedPanel.TLabel").pack(anchor=tk.W, pady=(0, 2))
        process_combo = ttk.Combobox(
            process_holder,
            textvariable=self.ukf_process_noise_model,
            values=("discrete", "continuous_white_acceleration"),
            state="readonly",
        )
        process_combo.pack(fill=tk.X)
        process_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_scenario_preview())
        self._entry(ukf_grid, "Acceleration PSD m2/s3", self.ukf_acceleration_psd, row=4, col=1)
        ttk.Checkbutton(
            ukf_grid,
            text="Adaptive process noise",
            variable=self.ukf_adaptive_process_noise,
            command=self.refresh_scenario_preview,
        ).grid(row=5, column=0, sticky="w")
        ukf_grid.columnconfigure(0, weight=1)
        ukf_grid.columnconfigure(1, weight=1)

        robust_grid = ttk.LabelFrame(left, text="UKF robust measurement update")
        robust_grid.pack(fill=tk.X, pady=(12, 0))
        ttk.Checkbutton(
            robust_grid,
            text="Enable Student-t/Huber covariance reweighting",
            variable=self.ukf_robust_measurement_update,
            command=self.refresh_scenario_preview,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        loss_holder = ttk.Frame(robust_grid)
        loss_holder.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(loss_holder, text="Robust loss", style="MutedPanel.TLabel").pack(anchor=tk.W, pady=(0, 2))
        loss_combo = ttk.Combobox(
            loss_holder,
            textvariable=self.ukf_robust_loss,
            values=("student_t", "huber"),
            state="readonly",
        )
        loss_combo.pack(fill=tk.X)
        loss_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_scenario_preview())
        self._entry(robust_grid, "Student-t dof", self.ukf_robust_student_t_dof, row=1, col=1)
        self._entry(robust_grid, "Huber threshold", self.ukf_robust_huber_threshold, row=2, col=0)
        self._entry(robust_grid, "Min component weight", self.ukf_robust_min_component_weight, row=2, col=1)
        robust_grid.columnconfigure(0, weight=1)
        robust_grid.columnconfigure(1, weight=1)

        bias_constraint_grid = ttk.LabelFrame(left, text="UKF bias observability constraints")
        bias_constraint_grid.pack(fill=tk.X, pady=(12, 0))
        ttk.Checkbutton(
            bias_constraint_grid,
            text="Auto freeze/regularize weak bias states",
            variable=self.ukf_auto_bias_constraints,
            command=self.refresh_scenario_preview,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self._entry(
            bias_constraint_grid,
            "Freeze relative info",
            self.ukf_bias_freeze_relative_information,
            row=1,
            col=0,
        )
        self._entry(
            bias_constraint_grid,
            "Regularize relative info",
            self.ukf_bias_regularize_relative_information,
            row=1,
            col=1,
        )
        self._entry(
            bias_constraint_grid,
            "Regularization std",
            self.ukf_bias_regularization_std,
            row=2,
            col=0,
        )
        bias_constraint_grid.columnconfigure(0, weight=1)
        bias_constraint_grid.columnconfigure(1, weight=1)
        self._entry(left, "Output dir", self.output_dir)

        actions = ttk.Frame(left, style="Panel.TFrame")
        actions.pack(fill=tk.X, pady=(14, 0))
        ttk.Button(actions, text="Validate", command=self.refresh_scenario_preview, style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(actions, text="Write normalized JSON", command=self.write_scenario_json).pack(side=tk.LEFT)

        preview = self.panel(right, padding=14)
        preview.pack(fill=tk.BOTH, expand=True)
        ttk.Label(preview, text="JSON preview and validation", style="Section.TLabel").pack(anchor=tk.W)
        self.validation_label = ttk.Label(preview, text="", style="MutedPanel.TLabel", wraplength=520)
        self.validation_label.pack(fill=tk.X, pady=(10, 8))
        self.json_text = tk.Text(
            preview,
            bg=COLORS["panel_2"],
            fg=COLORS["ink"],
            insertbackground=COLORS["ink"],
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=COLORS["line"],
            font=("Consolas", 10),
            wrap=tk.NONE,
        )
        self.json_text.pack(fill=tk.BOTH, expand=True)

    def _entry(
        self,
        parent: tk.Widget,
        label: str,
        variable: tk.Variable,
        row: int | None = None,
        col: int | None = None,
    ) -> ttk.Entry:
        holder = ttk.Frame(parent, style="Panel.TFrame")
        ttk.Label(holder, text=label, style="MutedPanel.TLabel").pack(anchor=tk.W, pady=(0, 2))
        entry = ttk.Entry(holder, textvariable=variable)
        entry.pack(fill=tk.X)
        entry.bind("<KeyRelease>", lambda _event: self.refresh_scenario_preview())
        if row is None or col is None:
            holder.pack(fill=tk.X, pady=(10, 0))
        else:
            holder.grid(row=row, column=col, sticky="ew", padx=(0 if col == 0 else 8, 8 if col == 0 else 0), pady=(0, 8))
        return entry

    def scenario_payload(self) -> dict[str, Any]:
        bias = self.bias_mode.get()
        nis_gate = self.ukf_nis_gate.get().strip()
        component_nis_gate = self.ukf_component_nis_gate.get().strip()
        acceleration_psd = self.ukf_acceleration_psd.get().strip()
        return {
            "name": self.scenario_name.get(),
            "measurement_type": self.measurement_type.get(),
            "estimator_type": self.estimator_type.get(),
            "start_mode": self.start_mode.get(),
            "network": self.scenario_network.get(),
            "duration_h": float(self.duration_h.get()),
            "sample_step_s": float(self.sample_step_s.get()),
            "max_iter": int(self.max_iter.get()),
            "rtol": float(self.rtol.get()),
            "atol": float(self.atol.get()),
            "noise": bool(self.noise_enabled.get()),
            "bias_mode": None if bias == "none" else bias,
            "range_rate_physics": self.range_rate_physics.get(),
            "count_interval_s": float(self.count_interval_s.get()),
            "uplink_frequency_hz": float(self.uplink_frequency_hz.get()),
            "turnaround_ratio": float(self.turnaround_ratio.get()),
            "two_way_local_state_model": self.two_way_local_state_model.get(),
            "station_clock_offset_s": float(self.station_clock_offset_s.get()),
            "station_clock_drift": float(self.station_clock_drift.get()),
            "clock_reference_time_s": float(self.clock_reference_time_s.get()),
            "transponder_delay_s": float(self.transponder_delay_s.get()),
            "ukf_alpha": float(self.ukf_alpha.get()),
            "ukf_beta": float(self.ukf_beta.get()),
            "ukf_kappa": float(self.ukf_kappa.get()),
            "ukf_covariance_inflation": float(self.ukf_covariance_inflation.get()),
            "ukf_adaptive_measurement_noise": bool(self.ukf_adaptive_measurement_noise.get()),
            "ukf_nis_gate": None if not nis_gate else float(nis_gate),
            "ukf_component_nis_gate": None if not component_nis_gate else float(component_nis_gate),
            "ukf_component_gate_mode": self.ukf_component_gate_mode.get(),
            "ukf_robust_measurement_update": bool(self.ukf_robust_measurement_update.get()),
            "ukf_robust_loss": self.ukf_robust_loss.get(),
            "ukf_robust_student_t_dof": float(self.ukf_robust_student_t_dof.get()),
            "ukf_robust_huber_threshold": float(self.ukf_robust_huber_threshold.get()),
            "ukf_robust_min_component_weight": float(self.ukf_robust_min_component_weight.get()),
            "ukf_covariance_form": self.ukf_covariance_form.get(),
            "ukf_process_noise_model": self.ukf_process_noise_model.get(),
            "ukf_acceleration_psd_m2_s3": None if not acceleration_psd else float(acceleration_psd),
            "ukf_adaptive_process_noise": bool(self.ukf_adaptive_process_noise.get()),
            "ukf_auto_bias_constraints": bool(self.ukf_auto_bias_constraints.get()),
            "ukf_bias_freeze_relative_information": float(self.ukf_bias_freeze_relative_information.get()),
            "ukf_bias_regularize_relative_information": float(
                self.ukf_bias_regularize_relative_information.get()
            ),
            "ukf_bias_regularization_std": float(self.ukf_bias_regularization_std.get()),
            "output_dir": self.output_dir.get(),
        }

    def refresh_scenario_preview(self) -> None:
        try:
            payload = self.scenario_payload()
            config = scenario_config_from_mapping(payload)
            text = json.dumps(config.to_dict(), indent=2)
            self.validation_label.configure(text="Valid scenario.", foreground=COLORS["green"])
        except Exception as exc:
            payload = {}
            try:
                payload = self.scenario_payload()
            except Exception:
                pass
            text = json.dumps(payload, indent=2, default=str)
            self.validation_label.configure(text=f"Validation issue: {exc}", foreground=COLORS["yellow"])
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert("1.0", text)
        self.draw_simulation()
        if hasattr(self, "analysis_canvas"):
            self.draw_analysis()
        self.refresh_dashboard()

    def apply_preset(self, key: str) -> None:
        presets = {
            "cold_range": ("cold_range_single", "position", "bls_lm", "cold", "single", "none"),
            "hot_rr": ("hot_range_rate_multi", "range_rate", "srif", "hot", "multi", "none"),
            "formal_rr": ("formal_rr_multi", "range_rate", "srif", "formal", "multi", "none"),
            "ukf_rr": ("ukf_range_rate_multi", "range_rate", "ukf", "hot", "multi", "none"),
            "twoway_ukf": ("twoway_ukf_rr_bias", "range_rate", "ukf", "hot", "multi", "station_full"),
            "sqrt_bias": ("sqrt_formal_rr_bias", "range_rate", "srif", "sqrt_formal", "multi", "station_full"),
        }
        name, measurement, estimator, start, network, bias = presets[key]
        self.scenario_name.set(name)
        self.measurement_type.set(measurement)
        self.estimator_type.set(estimator)
        self.start_mode.set(start)
        self.scenario_network.set(network)
        self.bias_mode.set(bias)
        if key == "twoway_ukf":
            self.range_rate_physics.set("two_way_counted_doppler")
            self.count_interval_s.set("30.0")
            self.two_way_local_state_model.set("taylor3")
            self.ukf_covariance_form.set("square_root")
            self.ukf_component_gate_mode.set("conditional")
            self.ukf_component_nis_gate.set("25.0")
            self.ukf_robust_measurement_update.set(True)
            self.ukf_robust_loss.set("student_t")
            self.ukf_auto_bias_constraints.set(True)
        elif measurement != "range_rate":
            self.range_rate_physics.set("geometric_instantaneous")
        self.refresh_scenario_preview()

    def write_scenario_json(self) -> None:
        try:
            config = scenario_config_from_mapping(self.scenario_payload())
        except Exception as exc:
            self.validation_label.configure(text=f"Cannot write invalid scenario: {exc}", foreground=COLORS["red"])
            return
        out_dir = PROJECT_ROOT / "python_port" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{config.name}.scenario.json"
        path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
        self.validation_label.configure(text=f"Wrote {path}", foreground=COLORS["green"])
        self.refresh_results()

    def _build_results_tab(self) -> None:
        left = self.panel(self.results_tab, padding=12)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = self.panel(self.results_tab, padding=12)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(16, 0))
        ttk.Label(left, text="Results browser", style="Section.TLabel").pack(anchor=tk.W)
        ttk.Button(left, text="Refresh", command=self.refresh_results).pack(anchor=tk.W, pady=(10, 8))
        columns = ("type", "size", "modified")
        self.results_tree = ttk.Treeview(left, columns=columns, show="tree headings")
        self.results_tree.heading("#0", text="File")
        self.results_tree.column("#0", width=360)
        for col, label, width in (("type", "Type", 70), ("size", "KB", 80), ("modified", "Modified", 150)):
            self.results_tree.heading(col, text=label)
            self.results_tree.column(col, width=width, anchor=tk.CENTER)
        self.results_tree.pack(fill=tk.BOTH, expand=True)
        self.results_tree.bind("<<TreeviewSelect>>", lambda _event: self.preview_result())

        ttk.Label(right, text="Preview", style="Section.TLabel").pack(anchor=tk.W)
        self.preview_text = tk.Text(
            right,
            bg=COLORS["panel_2"],
            fg=COLORS["ink"],
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=COLORS["line"],
            font=("Consolas", 10),
            wrap=tk.NONE,
        )
        self.preview_text.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def _result_files(self) -> list[Path]:
        if not RESULTS_DIR.exists():
            return []
        return sorted([p for p in RESULTS_DIR.iterdir() if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)

    def refresh_results(self) -> None:
        if not hasattr(self, "results_tree"):
            return
        self.results_tree.delete(*self.results_tree.get_children())
        for path in self._result_files():
            stat = path.stat()
            item = self.results_tree.insert(
                "",
                tk.END,
                text=path.name,
                values=(path.suffix.lstrip(".") or "file", f"{stat.st_size / 1024:.1f}", self._format_mtime(stat.st_mtime)),
            )
            self.results_tree.set(item, "type", path.suffix.lstrip(".") or "file")

    def preview_result(self) -> None:
        selection = self.results_tree.selection()
        if not selection:
            return
        name = self.results_tree.item(selection[0], "text")
        path = RESULTS_DIR / name
        self.preview_text.delete("1.0", tk.END)
        if path.suffix.lower() == ".csv":
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.reader(handle))[:80]
                widths = [0] * max((len(row) for row in rows), default=0)
                for row in rows:
                    for i, cell in enumerate(row):
                        widths[i] = max(widths[i], min(len(cell), 24))
                lines = []
                for row in rows:
                    lines.append("  ".join(cell[:24].ljust(widths[i]) for i, cell in enumerate(row)))
                self.preview_text.insert("1.0", "\n".join(lines))
            except Exception as exc:
                self.preview_text.insert("1.0", f"Could not read CSV: {exc}")
        elif path.suffix.lower() in {".json", ".txt"}:
            try:
                self.preview_text.insert("1.0", path.read_text(encoding="utf-8")[:12000])
            except Exception as exc:
                self.preview_text.insert("1.0", f"Could not read file: {exc}")
        else:
            self.preview_text.insert("1.0", f"{path.name}\n\nBinary/plot preview is not embedded yet.\nOpen from results folder to inspect the image.")

    @staticmethod
    def _format_mtime(timestamp: float) -> str:
        import datetime as _dt

        return _dt.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    os.chdir(SCRIPT_DIR)
    app = LunarODDesktopApp()
    app.mainloop()


if __name__ == "__main__":
    main()
