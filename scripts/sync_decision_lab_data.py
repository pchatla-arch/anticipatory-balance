"""Refresh or verify the evidence embedded in the offline Foresight decision figure."""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'analysis/operating_conditions/operating_conditions.json'
FIGURE = ROOT / 'docs/foresight/decision-lab.html'
MARKER = '<script type="application/json" id="study-data">'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='exit with an error if embedded data are stale')
    args = parser.parse_args()
    raw = json.loads(SOURCE.read_text())
    # The figure uses raw conditional counts and their within-scenario pairing.
    # The audit and illustration intervals remain in the source JSON.
    splits = {}
    for split, block in raw['splits'].items():
        cells = {}
        for delay, cell in block['cells'].items():
            cells[delay] = {field: cell[field] for field in ('arms', 'paired_differences')}
        splits[split] = {'cells': cells}
    embedded = json.dumps({'splits': splits}, separators=(',', ':'), sort_keys=True, allow_nan=False)
    embedded = embedded.replace('<', r'\u003c')
    html = FIGURE.read_text()
    if html.count(MARKER) != 1:
        raise SystemExit('Expected a single study-data element')
    start = html.index(MARKER) + len(MARKER)
    end = html.index('</script>', start)
    if args.check:
        if html[start:end] != embedded:
            raise SystemExit('The embedded evidence differs from the analysis JSON')
        print('Embedded decision-lab data match the archived analysis.')
    else:
        FIGURE.write_text(html[:start] + embedded + html[end:])


if __name__ == '__main__':
    main()
