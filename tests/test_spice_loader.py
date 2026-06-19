import tempfile
import unittest
from pathlib import Path

from lunar_od.spice_loader import REQUIRED_KERNELS, required_kernel_paths, resolve_kernel_dir


class SpiceLoaderTests(unittest.TestCase):
    def test_resolve_kernel_dir_uses_first_existing_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing"
            existing = Path(tmp) / "kernels"
            existing.mkdir()

            resolved = resolve_kernel_dir([missing, existing])
            self.assertEqual(resolved, existing)

    def test_required_kernel_paths_validates_file_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            kernel_dir = Path(tmp)
            for kernel_name in REQUIRED_KERNELS:
                (kernel_dir / kernel_name).write_text("", encoding="utf-8")

            paths = required_kernel_paths(kernel_dir)
            self.assertEqual([path.name for path in paths], list(REQUIRED_KERNELS))

    def test_required_kernel_paths_reports_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as caught:
                required_kernel_paths(tmp)

            self.assertIn(REQUIRED_KERNELS[0], str(caught.exception))


if __name__ == "__main__":
    unittest.main()

