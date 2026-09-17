import unittest

from vistory_visibility_eval import metrics


class ViStoryVisibilityEvalTests(unittest.TestCase):
    def setUp(self):
        self.gold = [{
            "scene_id": "s1", "family_id": "f1",
            "candidates": [
                {"candidate_id": "v", "visibility": "visible"},
                {"candidate_id": "n", "visibility": "not_visible"},
            ],
        }]

    def test_collapses_all_nonvisible_states(self):
        predictions = [{"scene_id": "s1", "candidates": [
            {"candidate_id": "v", "state": "visible"},
            {"candidate_id": "n", "state": "depicted"},
        ]}]
        result = metrics(self.gold, predictions)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["schema_valid_scene_rate"], 1.0)

    def test_missing_scene_is_wrong(self):
        result = metrics(self.gold, [])
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["schema_valid_scene_rate"], 0.0)

    def test_unknown_state_is_not_silently_collapsed(self):
        predictions = [{"scene_id": "s1", "candidates": [
            {"candidate_id": "v", "state": "visible"},
            {"candidate_id": "n", "state": "absent"},
        ]}]
        result = metrics(self.gold, predictions)
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["schema_valid_scene_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
