# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Run every phase test and exit non-zero if any failed.

The whole harness: discover tests/p*.py (this file excluded), run each with the
bundled uv, print a one-line PASS/FAIL per phase, and exit 0 only if all passed.
No test framework -- each phase test is a plain script that exits 0/non-zero.

    bin\\uv.exe run tests\\run_all.py
"""
import os
import re
import subprocess
import sys

HERE     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(HERE)
UV       = os.path.join(BASE_DIR, 'bin', 'uv.exe' if os.name == 'nt' else 'uv')


def phase_tests():
    """tests/p<NUM>_*.py, ordered by phase number."""
    found = []
    for name in os.listdir(HERE):
        m = re.match(r'p(\d+)_.*\.py$', name)
        if m:
            found.append((int(m.group(1)), name))
    return [name for _, name in sorted(found)]


def main():
    tests = phase_tests()
    if not tests:
        print('[run_all] no phase tests found')
        return 1
    print(f'[run_all] running {len(tests)} phase test(s) with {UV}\n')
    results = []
    for name in tests:
        path = os.path.join(HERE, name)
        print(f'=== {name} ' + '=' * (60 - len(name)))
        code = subprocess.run([UV, 'run', path], cwd=BASE_DIR).returncode
        results.append((name, code))
        print(f'--- {name}: {"PASS" if code == 0 else f"FAIL (exit {code})"}\n')
    failed = [n for n, c in results if c != 0]
    print('=' * 64)
    for name, code in results:
        print(f'  {"PASS" if code == 0 else "FAIL"}  {name}')
    print('=' * 64)
    if failed:
        print(f'[run_all] {len(failed)} failed: {", ".join(failed)}')
        return 1
    print(f'[run_all] all {len(results)} passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
