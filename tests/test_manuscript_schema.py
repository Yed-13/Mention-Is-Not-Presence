import unittest
from pathlib import Path
from check_manuscript_schema import check_inventory


VALID = r'''% BEGIN STATE INVENTORY
\texttt{visible},\texttt{audible\_only},\texttt{depicted},
\texttt{referenced\_only},\texttt{uncertain}
% END STATE INVENTORY'''


class ManuscriptSchemaTests(unittest.TestCase):
    def test_exact_labels(self):
        self.assertEqual(len(check_inventory(VALID)), 5)

    def test_alias_is_rejected(self):
        with self.assertRaises(ValueError):
            check_inventory(VALID.replace(r'audible\_only', 'audible'))

    def test_duplicate_is_rejected(self):
        with self.assertRaises(ValueError):
            check_inventory(VALID.replace('depicted', 'visible'))

    def test_missing_inventory_is_rejected(self):
        with self.assertRaises(ValueError):
            check_inventory('No state inventory')

    def test_local_manuscript_when_available(self):
        path = Path(__file__).resolve().parents[1] / 'paper/main.tex'
        if not path.exists():
            self.skipTest('Manuscript is distributed separately from experiments')
        check_inventory(path.read_text(encoding='utf8'))
