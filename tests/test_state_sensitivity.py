import unittest
from state_sensitivity import candidate_state_scores


class StateSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.gold = [{'scene_id': 's', 'candidates': [
            {'candidate_id': 'a', 'state': 'visible'},
            {'candidate_id': 'b', 'state': 'depicted'}]}]

    def test_evidence_independent(self):
        pred = [{'scene_id': 's', 'candidates': [
            {'candidate_id': 'a', 'state': 'visible', 'evidence': []},
            {'candidate_id': 'b', 'state': 'depicted', 'evidence': ['absent']}]}]
        self.assertEqual(candidate_state_scores(self.gold, pred)['accuracy'], 1)

    def test_duplicate_candidate_remains_wrong(self):
        c = {'candidate_id': 'a', 'state': 'visible'}
        pred = [{'scene_id': 's', 'candidates': [c, c, {'candidate_id': 'b', 'state': 'depicted'}]}]
        self.assertEqual(candidate_state_scores(self.gold, pred)['accuracy'], .5)

    def test_duplicate_scene_remains_wrong(self):
        row = {'scene_id': 's', 'candidates': [{'candidate_id': 'a', 'state': 'visible'}]}
        self.assertEqual(candidate_state_scores(self.gold, [row, row])['accuracy'], 0)

    def test_invalid_and_missing_states_remain_wrong(self):
        pred = [{'scene_id': 's', 'candidates': [{'candidate_id': 'a', 'state': 'unknown'}]}]
        self.assertEqual(candidate_state_scores(self.gold, pred)['accuracy'], 0)
