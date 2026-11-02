"""WP7.4 / CFG-04: kernels are loaded only when their content matches the manifest."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from lunar_od import kernel_manifest as KM
from lunar_od.spice_loader import PLANETARY_EPHEMERIS_KERNELS, required_kernels


def test_manifest_covers_every_required_kernel_for_every_ephemeris():
    for ephemeris in PLANETARY_EPHEMERIS_KERNELS:
        missing = set(required_kernels(ephemeris)) - set(KM.KERNEL_MANIFEST)
        assert not missing, (ephemeris, missing)
    for sha, source in KM.KERNEL_MANIFEST.values():
        assert len(sha) == 64 and int(sha, 16) >= 0
        assert source.startswith(KM.NAIF_GENERIC)


def test_text_kernel_hash_ignores_crlf_binary_does_not(tmp_path):
    lf, crlf = tmp_path / "a.tls.txt", tmp_path / "b.tls.txt"
    lf.write_bytes(b"\\begindata\nX = 1\n")
    crlf.write_bytes(b"\\begindata\r\nX = 1\r\n")
    assert KM.kernel_sha256(lf) == KM.kernel_sha256(crlf) == hashlib.sha256(lf.read_bytes()).hexdigest()
    b1, b2 = tmp_path / "a.bsp", tmp_path / "b.bsp"
    b1.write_bytes(b"\r\n")
    b2.write_bytes(b"\n")
    assert KM.kernel_sha256(b1) != KM.kernel_sha256(b2)


def test_unknown_and_changed_kernels_rejected(tmp_path, monkeypatch):
    good = tmp_path / "naif0012.tls.txt"
    good.write_bytes(b"leap seconds\n")
    monkeypatch.setitem(KM.KERNEL_MANIFEST, good.name,
                        (hashlib.sha256(b"leap seconds\n").hexdigest(), KM.NAIF_GENERIC + "lsk/naif0012.tls"))
    (record,) = KM.verify_kernels([good])
    assert record["name"] == good.name and record["source"].endswith("naif0012.tls")

    unknown = tmp_path / "mystery.bsp"
    unknown.write_bytes(b"x")
    with pytest.raises(KM.KernelIntegrityError, match="mystery.bsp: not in the kernel manifest"):
        KM.verify_kernels([good, unknown])

    good.write_bytes(b"leap seconds, edited\n")
    with pytest.raises(KM.KernelIntegrityError, match="naif0012.tls.txt: sha256"):
        KM.verify_kernels([good])
    assert not issubclass(KM.KernelIntegrityError, FileNotFoundError)


def test_load_spice_kernels_refuses_a_changed_kernel_before_furnishing(tmp_path, monkeypatch):
    """The real kernel set with one text kernel edited: nothing is furnished."""
    spice = pytest.importorskip("spiceypy")
    from lunar_od import spice_loader

    try:
        real = spice_loader.required_kernel_paths()
    except FileNotFoundError as exc:
        pytest.skip(f"SPICE kernels unavailable: {exc}")
    gm = tmp_path / "gm_de431.tpc.txt"
    shutil.copyfile(next(p for p in real if p.name == gm.name), gm)
    gm.write_bytes(gm.read_bytes() + b"\n")
    edited = [gm if p.name == gm.name else p for p in real]
    monkeypatch.setattr(spice_loader, "required_kernel_paths", lambda *args, **kwargs: edited)

    furnished = []
    monkeypatch.setattr(spice, "furnsh", furnished.append)
    monkeypatch.setattr(spice, "kclear", lambda: None)
    with pytest.raises(KM.KernelIntegrityError, match="gm_de431.tpc.txt"):
        spice_loader.load_spice_kernels()
    assert furnished == []
    assert spice_loader.load_spice_kernels(verify=False) == edited
    assert furnished == [str(p) for p in edited]


def test_validated_local_kernel_set_matches_manifest():
    from lunar_od import spice_loader

    try:
        paths = [spice_loader.required_kernel_paths(ephemeris=e) for e in PLANETARY_EPHEMERIS_KERNELS]
    except FileNotFoundError as exc:
        pytest.skip(f"SPICE kernels unavailable: {exc}")
    records = KM.verify_kernels({p for group in paths for p in group})
    assert {r["name"] for r in records} == {p.name for group in paths for p in group}
