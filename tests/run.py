"""Run the tests without pytest: python tests/run.py (from the repository root). With pytest: pytest tests."""
import importlib
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))

failed = 0
for path in sorted((ROOT/'tests').glob('test_*.py')):
    module = importlib.import_module(path.stem)
    for name in sorted(n for n in dir(module) if n.startswith('test_')):
        t0 = time.time()
        try:
            getattr(module, name)()
            print(f'ok      {path.stem}.{name} ({time.time()-t0:.1f} s)', flush=True)
        except Exception:
            failed += 1
            print(f'FAILED  {path.stem}.{name}', flush=True)
            traceback.print_exc()
print('all tests passed' if not failed else f'{failed} test(s) failed')
sys.exit(1 if failed else 0)
