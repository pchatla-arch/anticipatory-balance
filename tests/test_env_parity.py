"""The parity guard must fail on any version difference against the pod's lock (AE-009)."""
from importlib import metadata
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import check_env_parity as parity  # noqa: E402

LOCK = '''version = 1

[[package]]
name = "mujoco"
version = "3.16.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "mujoco-mjx"
version = "3.16.0"

[[package]]
name = "warp-lang"
version = "1.18.0"
'''


class ParityTests(unittest.TestCase):
    def test_lock_parsing_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / 'uv.lock'
            lock.write_text(LOCK)
            locked = parity.locked_versions(lock)
        self.assertEqual(locked, {'mujoco': '3.16.0', 'mujoco-mjx': '3.16.0', 'warp-lang': '1.18.0'})
        installed_mujoco = metadata.version('mujoco')
        rows = parity.compare(locked, ('mujoco', 'warp-lang', 'not-installed-package'))
        by_name = {r['package']: r for r in rows}
        self.assertEqual(by_name['mujoco']['match'], installed_mujoco == '3.16.0')
        self.assertFalse(by_name['not-installed-package']['match'])
        self.assertIsNone(by_name['not-installed-package']['installed'])

    def test_main_exit_status_reflects_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / 'uv.lock'
            lock.write_text(LOCK.replace('3.16.0', '0.0.1'))
            out = Path(tmp) / 'parity.json'
            code = parity.main(['--lock', str(lock), '--json', str(out), '--watch', 'mujoco'])
            self.assertEqual(code, 1, 'a deliberately wrong pin must fail')
            self.assertIn('"parity": false', out.read_text())
            lock.write_text(LOCK.replace('3.16.0', metadata.version('mujoco')))
            self.assertEqual(parity.main(['--lock', str(lock), '--watch', 'mujoco']), 0)


if __name__ == '__main__':
    unittest.main()
