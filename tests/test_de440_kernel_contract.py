"""DE440 lunar PA kernel contract: atomic pair + fail-closed preflight.

The DE440 capability is an ATOMIC TWO-FILE contract. Measured before this
contract existed:

    frame definition only -> SpiceFRAMEDATANOTFOUND
    binary PCK only       -> SpiceUNKNOWNFRAME
    both                  -> resolves

A half-configured DE440 setup is strictly worse than none, because the frame
definition alone also stops generic ``MOON_PA`` from resolving. These tests
pin that contract so the pair cannot drift apart.

SPICE's kernel pool is global process state, so every selective-kernel test
restores the standard pool in ``finally``.
"""

from __future__ import annotations

import unittest

import spiceypy as spice

from lunar_od.spice_loader import (
    MOON_PA_DE421_FRAME,
    MOON_PA_DE440_FRAME,
    MOON_PA_DE440_KERNELS,
    LunarFrameKernelError,
    load_spice_kernels,
    require_moon_pa_de440,
    resolve_kernel_dir,
)

#: Arbitrary epoch well inside the coverage of every kernel involved.
ET = 820497669.18392


def _restore_standard_pool() -> None:
    spice.kclear()
    load_spice_kernels(None, clear=False)


class DE440KernelContract(unittest.TestCase):
    """Selective-kernel states. Each restores the standard pool afterwards."""

    @classmethod
    def setUpClass(cls):
        cls.kernel_dir = resolve_kernel_dir()
        _restore_standard_pool()

    @classmethod
    def tearDownClass(cls):
        _restore_standard_pool()

    def _furnish_only(self, names):
        """Clear the pool and furnish exactly ``names``."""
        spice.kclear()
        for name in names:
            spice.furnsh(str(self.kernel_dir / name))

    # -- the atomic-pair matrix ------------------------------------------
    def test_case1_no_de440_files_fails_closed(self):
        try:
            self._furnish_only(())
            with self.assertRaises(LunarFrameKernelError) as ctx:
                require_moon_pa_de440(ET)
            self.assertIn(MOON_PA_DE440_FRAME, str(ctx.exception))
            self.assertIsNotNone(ctx.exception.__cause__)
        finally:
            _restore_standard_pool()

    def test_case2_frame_definition_only_fails_closed(self):
        """Half a pair must not be mistaken for DE440 readiness."""
        try:
            self._furnish_only((MOON_PA_DE440_KERNELS[0],))
            with self.assertRaises(LunarFrameKernelError):
                require_moon_pa_de440(ET)
        finally:
            _restore_standard_pool()

    def test_case3_binary_pck_only_fails_closed(self):
        try:
            self._furnish_only((MOON_PA_DE440_KERNELS[1],))
            with self.assertRaises(LunarFrameKernelError):
                require_moon_pa_de440(ET)
        finally:
            _restore_standard_pool()

    def test_case4_both_de440_files_pass(self):
        try:
            self._furnish_only(MOON_PA_DE440_KERNELS)
            require_moon_pa_de440(ET)  # must not raise
        finally:
            _restore_standard_pool()

    # -- the standard loader ---------------------------------------------
    def test_case5_standard_loader_resolves_de440(self):
        """The defect this patch fixes: DE440 must work after normal loading."""
        _restore_standard_pool()
        require_moon_pa_de440(ET)
        self.assertEqual(
            spice.pxform("J2000", MOON_PA_DE440_FRAME, ET).shape, (3, 3)
        )

    def test_case6_standard_loader_retains_de421(self):
        """DE440 support must not displace the legacy DE421 realization."""
        _restore_standard_pool()
        self.assertEqual(
            spice.pxform("J2000", MOON_PA_DE421_FRAME, ET).shape, (3, 3)
        )

    def test_de440_pair_is_declared_atomically(self):
        """Both filenames must be in the standard set, together."""
        from lunar_od.spice_loader import REQUIRED_KERNELS

        self.assertEqual(len(MOON_PA_DE440_KERNELS), 2)
        for name in MOON_PA_DE440_KERNELS:
            self.assertIn(name, REQUIRED_KERNELS)

    def test_generic_moon_pa_still_means_de421(self):
        """Adding DE440 must not silently rebind the generic alias.

        Both DE440 and DE421 text kernels define ``MOON_PA``, and the last one
        loaded wins. Listing the DE440 pair first keeps the historical DE421
        binding; appending it last rebinds every legacy generic-alias consumer
        to a ~1.65 m different realization. Load order is the only thing
        holding this, so it is pinned here.
        """
        import numpy as np

        _restore_standard_pool()
        generic = spice.pxform("J2000", "MOON_PA", ET)
        de421 = spice.pxform("J2000", MOON_PA_DE421_FRAME, ET)
        de440 = spice.pxform("J2000", MOON_PA_DE440_FRAME, ET)
        # The realizations must genuinely differ, or this test proves nothing.
        self.assertGreater(float(np.abs(de440 - de421).max()), 1e-8)
        np.testing.assert_array_equal(generic, de421)

    def test_preflight_has_no_generic_fallback(self):
        """Explicit realizations must differ from the generic alias."""
        self.assertNotEqual(MOON_PA_DE440_FRAME, "MOON_PA")
        self.assertNotEqual(MOON_PA_DE421_FRAME, "MOON_PA")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class ExplicitFrameAPI(unittest.TestCase):
    """The sampler must not carry a generic frame default (Stage A)."""

    def test_frame_parameter_has_no_default(self):
        import inspect

        from lunar_od.lunar_frames import sample_moon_pa_rotations

        sig = inspect.signature(sample_moon_pa_rotations)
        self.assertIs(sig.parameters["frame"].default, inspect.Parameter.empty)

    def test_omitting_frame_raises_typeerror(self):
        import numpy as np

        from lunar_od.lunar_frames import sample_moon_pa_rotations

        with self.assertRaises(TypeError):
            sample_moon_pa_rotations(ET, np.array([0.0]))
