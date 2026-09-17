"""Check literal manuscript state labels against the executable evaluator."""
import argparse
import re
from pathlib import Path
from scriptbreak_eval import STATES


def check_inventory(text):
    match = re.search(r'% BEGIN STATE INVENTORY(.*?)% END STATE INVENTORY', text, re.S)
    if match is None:
        raise ValueError('Marked manuscript state inventory is missing')
    labels = re.findall(r'\\texttt\{([^{}]+)\}', match.group(1))
    labels = [label.replace(r'\_', '_') for label in labels]
    if len(labels) != len(STATES) or set(labels) != STATES:
        raise ValueError('Manuscript labels differ from evaluator: '+repr(labels))
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manuscript', type=Path)
    args = parser.parse_args()
    check_inventory(args.manuscript.read_text(encoding='utf8'))
    print('Manuscript state inventory matches evaluator')
