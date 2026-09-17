import unittest
from primary_uncertainty import paired, scores


class PrimaryUncertaintyTests(unittest.TestCase):
    def test_scores(self):
        self.assertEqual(scores([2, 0, 0, 1, 1], 1), (1., 1.))

    def test_paired_identity(self):
        gold = [{'scene_id': 's', 'family_id': 'f', 'template_id': 't', 'text': 'A',
                 'candidates': [{'candidate_id': 'a', 'state': 'visible'}]}]
        pred = [{'scene_id': 's', 'candidates': [{'candidate_id': 'a', 'state': 'visible', 'evidence': ['A']}]}]
        r = paired(gold, pred, pred, iterations=100)
        for metric in ['macro_f1', 'family_exact_match']:
            self.assertEqual(r[metric]['difference_a_minus_b'], 0)
            self.assertEqual(r[metric]['cluster_bootstrap_95_ci'], [0, 0])

    def test_missing_predictions(self):
        gold = [{'scene_id': 's', 'family_id': 'f', 'template_id': 't', 'text': 'A',
                 'candidates': [{'candidate_id': 'a', 'state': 'visible'}]}]
        pred = [{'scene_id': 's', 'candidates': [{'candidate_id': 'a', 'state': 'visible', 'evidence': ['A']}]}]
        r = paired(gold, pred, [], iterations=100)
        self.assertEqual(r['family_exact_match']['cluster_bootstrap_95_ci'], [1, 1])
