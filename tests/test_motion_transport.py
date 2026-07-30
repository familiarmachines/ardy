from __future__ import annotations

import base64
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from motion_transport import encode_motion_file, materialize_motion_file


class MotionTransportTests(unittest.TestCase):
    def test_round_trip_materializes_a_local_motion_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ardy_motion_transport_test_") as tmpdir:
            root = Path(tmpdir)
            source = root / "generated motion.npz"
            source.write_bytes(b"ARDY motion bytes")

            payload = encode_motion_file(source)
            destination = materialize_motion_file(payload, root / "received")

            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertEqual(destination.parent, root / "received")
            self.assertTrue(destination.name.endswith("_generated_motion.npz"))

    def test_reuses_an_identical_cached_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ardy_motion_transport_test_") as tmpdir:
            root = Path(tmpdir)
            source = root / "motion.npz"
            source.write_bytes(b"same motion")
            payload = encode_motion_file(source)

            first = materialize_motion_file(payload, root / "received")
            first_mtime = first.stat().st_mtime_ns
            second = materialize_motion_file(payload, root / "received")

            self.assertEqual(first, second)
            self.assertEqual(second.stat().st_mtime_ns, first_mtime)

    def test_rejects_a_checksum_mismatch(self) -> None:
        payload = {
            "version": 1,
            "name": "motion.npz",
            "encoding": "base64",
            "size": 4,
            "sha256": "0" * 64,
            "data": base64.b64encode(b"data").decode("ascii"),
        }
        with (
            tempfile.TemporaryDirectory(prefix="ardy_motion_transport_test_") as tmpdir,
            self.assertRaisesRegex(ValueError, "checksum mismatch"),
        ):
            materialize_motion_file(payload, Path(tmpdir))

    def test_rejects_path_traversal_and_normalizes_the_name(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ardy_motion_transport_test_") as tmpdir:
            root = Path(tmpdir)
            source = root / "motion.npz"
            source.write_bytes(b"motion")
            payload = encode_motion_file(source)
            payload["name"] = "../../remote motion.npz"

            destination = materialize_motion_file(payload, root / "received")

            self.assertEqual(destination.parent, root / "received")
            self.assertTrue(destination.name.endswith("_remote_motion.npz"))

    def test_rejects_invalid_base64(self) -> None:
        payload = {
            "version": 1,
            "name": "motion.npz",
            "encoding": "base64",
            "size": 3,
            "sha256": "0" * 64,
            "data": "***",
        }
        with (
            tempfile.TemporaryDirectory(prefix="ardy_motion_transport_test_") as tmpdir,
            self.assertRaisesRegex(ValueError, "not valid base64"),
        ):
            materialize_motion_file(payload, Path(tmpdir))


if __name__ == "__main__":
    unittest.main()
