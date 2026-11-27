"""Offline tests for provider and budget guards, no RunPod allocation."""
import importlib.util
from pathlib import Path
from unittest import TestCase, mock
import tempfile
import hashlib

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_g1_pod_one_command.py"
spec = importlib.util.spec_from_file_location("g1_pod_supervisor", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PodSupervisorGuards(TestCase):
    def pod(self, **changes):
        p = {
            "id": "test-pod", "status": "RUNNING", "cloud": "COMMUNITY",
            "cudaVersion": "13.2", "gpu": {
                "id": "NVIDIA RTX 6000 Ada Generation", "count": 1},
            "cost": 0.74,
            "startedAt": "2026-11-26T20:00:00Z",
        }
        p.update(changes)
        return p

    def test_authorized_pod(self):
        with mock.patch.object(module, "runpod_request", return_value=self.pod()):
            got = module.validate_pod("test-pod", "unused")
        self.assertEqual(got["cloud"], "COMMUNITY")

    def test_secure_ada_accepted_at_authorized_rate(self):
        with mock.patch.object(module, "runpod_request",
                               return_value=self.pod(cloud="SECURE", cost=1.09)):
            got = module.validate_pod("test-pod", "unused")
        self.assertEqual(got["cloud"], "SECURE")

    def test_secure_a6000_accepted_at_authorized_rate(self):
        with mock.patch.object(module, "runpod_request",
                               return_value=self.pod(cloud="SECURE", cost=0.69,
                                                     gpu={"id": "NVIDIA RTX A6000", "count": 1})):
            got = module.validate_pod("test-pod", "unused")
        self.assertEqual(got["gpu"]["id"], "NVIDIA RTX A6000")

    def test_secure_a6000_rejected_above_ceiling(self):
        with mock.patch.object(module, "runpod_request",
                               return_value=self.pod(cloud="SECURE", cost=0.70,
                                                     gpu={"id": "NVIDIA RTX A6000", "count": 1})):
            with self.assertRaises(RuntimeError):
                module.validate_pod("test-pod", "unused")

    def test_higher_price_rejected(self):
        with mock.patch.object(module, "runpod_request",
                               return_value=self.pod(cost=1.09)):
            with self.assertRaises(RuntimeError):
                module.validate_pod("test-pod", "unused")

    def test_zero_price_is_not_accepted_as_verified_quote(self):
        with mock.patch.object(module, "runpod_request",
                               return_value=self.pod(cost=0)):
            with self.assertRaises(RuntimeError):
                module.validate_pod("test-pod", "unused")

    def test_wrong_gpu_rejected(self):
        with mock.patch.object(module, "runpod_request",
                               return_value=self.pod(gpu={
                                   "id": "NVIDIA RTX A6000", "count": 1})):
            with self.assertRaises(RuntimeError):
                module.validate_pod("test-pod", "unused")

    def test_unavailable_cuda_rejected(self):
        with mock.patch.object(module, "runpod_request",
                               return_value=self.pod(cudaVersion="12.8")):
            with self.assertRaises(RuntimeError):
                module.validate_pod("test-pod", "unused")

    def test_provider_start_time_in_utc(self):
        self.assertEqual(
            int(module.started_epoch(self.pod())),
            1795723200,
        )

    def test_checkpoint_digest_deterministic(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            (root / "a.txt").write_bytes(b"sample")
            digest = hashlib.sha256()
            digest.update(b"a.txt\0" + hashlib.sha256(b"sample").hexdigest().encode()
                          + b"\n")
            self.assertEqual(module.checkpoint_digest(root), digest.hexdigest())

    def test_missing_checkpoint_rejected(self):
        with self.assertRaises(RuntimeError):
            module.checkpoint_digest(Path("/nonexistent/g1/this-checkpoint-is-missing"))


if __name__ == "__main__":
    import unittest
    unittest.main()
