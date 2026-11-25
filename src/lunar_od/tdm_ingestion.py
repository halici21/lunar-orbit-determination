"""CCSDS TDM input for the observation-row layouts of :mod:`lunar_od.measurement_ingestion` (F6 WP6.1).

``measurement_ingestion`` is byte-frozen (``docs/development/path-migration.json``), so its TDM entry
point lives here and returns the same :class:`~lunar_od.measurement_ingestion.IngestedMeasurements`
as :func:`~lunar_od.measurement_ingestion.read_measurement_csv`. The KVN format itself is
:mod:`lunar_od.ccsds_tdm`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .measurement_ingestion import IngestedMeasurements
from .scenarios import MeasurementType

__all__ = ["TDM_POSITION_KEYWORDS", "TDM_RANGE_RATE_KEYWORDS", "TWO_WAY_CONVENTIONS",
           "ingested_to_tdm", "read_measurement_tdm"]


#: TDM data keywords each observation layout needs at one (station, epoch).
TDM_POSITION_KEYWORDS = ("RANGE", "ANGLE_1", "ANGLE_2")
TDM_RANGE_RATE_KEYWORDS = ("RANGE", "DOPPLER_INSTANTANEOUS", "ANGLE_1", "ANGLE_2")
TWO_WAY_CONVENTIONS = ("round_trip", "one_way_equivalent")


def read_measurement_tdm(
    source,
    measurement_type: MeasurementType,
    *,
    station_names: Sequence[str],
    et0_s: float,
    two_way_convention: str | None = None,
    site=None,
) -> IngestedMeasurements:
    """Read a CCSDS TDM (KVN) into the position or range-rate ObsData layout.

    ``source`` is a path or a parsed :class:`lunar_od.ccsds_tdm.TrackingDataMessage`. Rows are
    ``[t_s, range_m, az_rad, el_rad, station_id, time_index]`` (position) or
    ``[t_s, range_m, range_rate_mps, az_rad, el_rad, station_id, time_index]`` (range rate), with
    ``t_s`` the TDB seconds past ``et0_s`` (itself TDB seconds past J2000) of the receive time tag,
    ``station_id`` one-based into ``station_names`` and ``time_index`` one-based into the sorted
    unique epochs of the file (build the pass geometry on ``np.unique(obs_data[:, 0])``).

    Records of several segments (TDM producers often put range and angles in separate segments)
    are joined per (station, epoch); every joined epoch must carry exactly the keywords of the
    layout (:data:`TDM_POSITION_KEYWORDS` / :data:`TDM_RANGE_RATE_KEYWORDS`).

    Fails closed (``ValueError``/:class:`~lunar_od.ccsds_tdm.TDMFormatError`) on what the
    observation layout cannot represent: a data keyword the layout does not use, MODE other than
    SEQUENTIAL, a path that does not end at a station of ``station_names`` or that ends where it did
    not start (three-way), TIMETAG_REF = TRANSMIT, RANGE_UNITS other than km, a non-zero
    RANGE_MODULUS (ambiguous range), ANGLE_TYPE other than AZEL, non-zero TRANSMIT/RECEIVE delays,
    CORRECTION_* not applied, DATA_QUALITY = DEGRADED, more than one spacecraft, or an incomplete
    epoch. For a two-way path (``a,b,a``) the standard leaves to the ICD whether RANGE and Doppler
    are the round-trip value or half of it (503.0-B-2 3.5.2.3, 3.5.2.7): pass
    ``two_way_convention="round_trip"`` (halved here) or ``"one_way_equivalent"``; it is refused
    for one-way data and required for two-way data.
    """
    from .ccsds_tdm import TrackingDataMessage, read_tdm, tdm_epochs_to_tdb_seconds

    if measurement_type not in {"position", "range_rate"}:
        raise ValueError("measurement_type must be 'position' or 'range_rate'.")
    if two_way_convention is not None and two_way_convention not in TWO_WAY_CONVENTIONS:
        raise ValueError(f"two_way_convention must be one of {TWO_WAY_CONVENTIONS}.")
    station_names = tuple(station_names)
    if not station_names:
        raise ValueError("read_measurement_tdm needs station_names to map TDM participants.")
    tdm = source if isinstance(source, TrackingDataMessage) else read_tdm(source)
    wanted = TDM_POSITION_KEYWORDS if measurement_type == "position" else TDM_RANGE_RATE_KEYWORDS

    joined: dict[tuple[int, float], dict[str, float]] = {}
    spacecraft: set[str] = set()
    for number, segment in enumerate(tdm.segments, start=1):
        where = f"TDM segment {number}"
        station_id, target, two_way = _tdm_segment_geometry(segment, station_names, where)
        spacecraft.add(target)
        if two_way and two_way_convention is None:
            raise ValueError(f"{where}: two-way PATH {segment.get('PATH')}; pass two_way_convention "
                             f"('round_trip' or 'one_way_equivalent') as the ICD defines it.")
        if not two_way and two_way_convention is not None:
            raise ValueError(f"{where}: two_way_convention given but PATH {segment.get('PATH')} is one-way.")
        scale = 0.5 if two_way and two_way_convention == "round_trip" else 1.0
        keywords = {r.keyword for r in segment.records}
        unused = sorted(keywords - set(wanted))
        if unused:
            raise ValueError(f"{where}: data keyword(s) {', '.join(unused)} have no column in the "
                             f"{measurement_type} layout.")
        _tdm_segment_conventions(segment, keywords, where)
        times = tdm_epochs_to_tdb_seconds([r.epoch for r in segment.records], segment.time_system,
                                          et0_s, site)
        for record, t in zip(segment.records, times):
            value = record.value
            if record.keyword == "RANGE":
                value = value * 1000.0 * scale
            elif record.keyword == "DOPPLER_INSTANTANEOUS":
                value = value * 1000.0 * scale
            elif record.keyword == "ANGLE_1":
                if not 0.0 <= value < 360.0:
                    raise ValueError(f"{where}: azimuth {value} deg outside [0, 360).")
                value = np.deg2rad(value)
            elif record.keyword == "ANGLE_2":
                if not -90.0 <= value <= 90.0:
                    raise ValueError(f"{where}: elevation {value} deg outside [-90, 90].")
                value = np.deg2rad(value)
            slot = joined.setdefault((station_id, float(t)), {})
            if record.keyword in slot:
                raise ValueError(f"{where}: {record.keyword} for station {station_names[station_id - 1]!r} "
                                 f"at {record.epoch.format()} appears in more than one segment.")
            slot[record.keyword] = value
    if len(spacecraft) > 1:
        raise ValueError(f"TDM tracks more than one spacecraft: {', '.join(sorted(spacecraft))}.")

    width = 6 if measurement_type == "position" else 7
    if not joined:
        return IngestedMeasurements(measurement_type, np.zeros((0, width), dtype=float), station_names)
    epochs = sorted({t for _, t in joined})
    index = {t: i + 1 for i, t in enumerate(epochs)}
    rows = []
    for (station_id, t), values in sorted(joined.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        missing = [k for k in wanted if k not in values]
        if missing:
            raise ValueError(f"TDM epoch t = {t!r} s, station {station_names[station_id - 1]!r} lacks "
                             f"{', '.join(missing)} for the {measurement_type} layout.")
        if measurement_type == "position":
            rows.append([t, values["RANGE"], values["ANGLE_1"], values["ANGLE_2"], float(station_id),
                         float(index[t])])
        else:
            rows.append([t, values["RANGE"], values["DOPPLER_INSTANTANEOUS"], values["ANGLE_1"],
                         values["ANGLE_2"], float(station_id), float(index[t])])
    return IngestedMeasurements(measurement_type, np.asarray(rows, dtype=float), station_names)


def _tdm_segment_geometry(segment, station_names, where):
    """(one-based station id, spacecraft participant, two-way?) of a SEQUENTIAL segment."""
    if segment.get("MODE") != "SEQUENTIAL":
        raise ValueError(f"{where}: MODE = {segment.get('MODE')}; only SEQUENTIAL maps to observation rows.")
    path = segment.path
    names = segment.participants
    receiver = names[path[-1]]
    if receiver not in station_names:
        raise ValueError(f"{where}: PATH ends at {receiver!r}, which is not one of station_names.")
    if len(path) == 2:
        two_way = False
    elif len(path) == 3 and path[0] == path[2]:
        two_way = True
    else:
        raise ValueError(f"{where}: PATH {segment.get('PATH')} is neither one-way nor two-way; "
                         f"three-way and n-way data have no single-station row.")
    target = names[path[-2]]
    if target in station_names:
        raise ValueError(f"{where}: the participant before the receiver ({target!r}) is a station, "
                         f"not the tracked spacecraft.")
    return station_names.index(receiver) + 1, target, two_way


def _tdm_segment_conventions(segment, keywords, where):
    if segment.get("TIMETAG_REF", "RECEIVE") != "RECEIVE":
        raise ValueError(f"{where}: TIMETAG_REF = TRANSMIT; observation rows are receive-time tagged.")
    if "RANGE" in keywords:
        if segment.range_units != "km":
            raise ValueError(f"{where}: RANGE_UNITS = {segment.range_units}; only km converts without an ICD.")
        if float(segment.get("RANGE_MODULUS", "0")) != 0.0:
            raise ValueError(f"{where}: RANGE_MODULUS is non-zero (ambiguous range); resolve it before ingestion.")
    if keywords & {"ANGLE_1", "ANGLE_2"} and segment.get("ANGLE_TYPE") != "AZEL":
        raise ValueError(f"{where}: ANGLE_TYPE = {segment.get('ANGLE_TYPE')}; observation rows hold AZEL angles.")
    for key, value in segment.metadata:
        if key.startswith(("TRANSMIT_DELAY_", "RECEIVE_DELAY_")) and float(value) != 0.0:
            raise ValueError(f"{where}: {key} = {value} s is not modelled by the observation rows.")
    if segment.get("CORRECTIONS_APPLIED") == "NO" and any(
            k.startswith("CORRECTION_") and k != "CORRECTIONS_APPLIED" for k, _ in segment.metadata):
        raise ValueError(f"{where}: CORRECTION_* values are not applied (CORRECTIONS_APPLIED = NO).")
    if segment.get("DATA_QUALITY") == "DEGRADED":
        raise ValueError(f"{where}: DATA_QUALITY = DEGRADED.")


def ingested_to_tdm(
    ingested: IngestedMeasurements,
    *,
    spacecraft: str,
    et0_s: float,
    originator: str,
    creation_date,
    message_id: str | None = None,
):
    """Observation rows -> a CCSDS TDM (version 2.0) with one one-way segment per station.

    Time tags are TDB (``et0_s + t_s``), RANGE in km, DOPPLER_INSTANTANEOUS in km/s, angles AZEL
    in deg; ``PATH = 1,2`` from the spacecraft (PARTICIPANT_1) to the station (PARTICIPANT_2).
    The geometric observation rows have no light-time or transponder convention, so the one-way
    path is the faithful description. ``creation_date`` is a ``TDMEpoch`` (UTC) or its text.
    :func:`read_measurement_tdm` reads the result back to the same rows.
    """
    from .ccsds_tdm import (TDMEpoch, TDMRecord, TDMSegment, TrackingDataMessage,
                            tdb_seconds_to_tdm_epoch)

    obs = np.asarray(ingested.obs_data, dtype=float)
    position = ingested.measurement_type == "position"
    if ingested.measurement_type not in {"position", "range_rate"}:
        raise ValueError("measurement_type must be 'position' or 'range_rate'.")
    if not ingested.station_names:
        raise ValueError("ingested_to_tdm needs station_names to name the TDM participants.")
    station_col = 4 if position else 5
    if isinstance(creation_date, str):
        creation_date = TDMEpoch.parse(creation_date)
    segments = []
    for station_id in sorted({int(s) for s in obs[:, station_col]}):
        rows = obs[obs[:, station_col] == station_id]
        rows = rows[np.argsort(rows[:, 0], kind="stable")]
        records = []
        for row in rows:
            epoch = tdb_seconds_to_tdm_epoch(row[0], et0_s)
            values = [("RANGE", row[1] / 1000.0)]
            if not position:
                values.append(("DOPPLER_INSTANTANEOUS", row[2] / 1000.0))
            az, el = (row[2], row[3]) if position else (row[3], row[4])
            values += [("ANGLE_1", float(np.rad2deg(az))), ("ANGLE_2", float(np.rad2deg(el)))]
            records += [TDMRecord.from_value(k, epoch, v) for k, v in values]
        metadata = (
            ("TIME_SYSTEM", "TDB"),
            ("PARTICIPANT_1", spacecraft),
            ("PARTICIPANT_2", ingested.station_names[station_id - 1]),
            ("MODE", "SEQUENTIAL"),
            ("PATH", "1,2"),
            ("RANGE_UNITS", "km"),
            ("ANGLE_TYPE", "AZEL"),
        )
        segments.append(TDMSegment(metadata, tuple(records)))
    return TrackingDataMessage(version="2.0", creation_date=creation_date, originator=originator,
                               segments=tuple(segments), message_id=message_id)
