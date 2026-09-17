import unittest

from scriptbreak_eval import confusion_and_scores, paired_family_bootstrap, split_families, validate


def scene(scene_id, family_id, state="visible"):
    return {
        "scene_id": scene_id,
        "family_id": family_id,
        "text": "内景。林青站着。",
        "phenomenon": "explicit",
        "status": "approved_synthetic",
        "annotation_source": "programmatic_by_construction",
        "candidates": [{
            "candidate_id": "c1",
            "mention": "林青",
            "entity_type": "person",
            "state": state,
            "evidence": ["林青站着"],
        }],
    }


def prediction(scene_id):
    return {"scene_id": scene_id, "candidates": [{
        "candidate_id": "c1", "state": "visible", "evidence": ["林青站着"]
    }]}


class ScriptBreakEvaluationTests(unittest.TestCase):
    def test_validate_reference(self):
        self.assertTrue(validate([scene("s1", "f1")], require_reference=True))

    def test_bad_evidence(self):
        row = scene("s1", "f1")
        row["candidates"][0]["evidence"] = ["不存在"]
        with self.assertRaises(ValueError):
            validate([row], require_reference=True)

    def test_family_split_has_no_leakage(self):
        rows = [scene(f"s{i}", f"f{i // 2}") for i in range(12)]
        output = split_families(rows, 7, .5, .25, .25)
        memberships = {}
        for row in output:
            memberships.setdefault(row["family_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in memberships.values()))

    def test_missing_prediction_is_wrong(self):
        result = confusion_and_scores([scene("s1", "f1")], [])
        self.assertEqual(result["accuracy"], 0)
        self.assertEqual(result["confusion"]["visible"]["missing"], 1)

    def test_exact_prediction(self):
        result = confusion_and_scores([scene("s1", "f1")], [prediction("s1")])
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["evidence_exact_match"], 1)

    def test_unknown_scene_is_invalid(self):
        result = confusion_and_scores([scene("s1", "f1")], [{"scene_id": "unknown", "candidates": []}])
        self.assertEqual(result["schema_valid_scene_rate"], 0)
        self.assertTrue(result["validation_errors"])

    def test_duplicate_candidate_is_invalid(self):
        item = {"candidate_id": "c1", "state": "visible", "evidence": ["林青站着"]}
        result = confusion_and_scores([scene("s1", "f1")], [{"scene_id": "s1", "candidates": [item, item]}])
        self.assertEqual(result["schema_valid_scene_rate"], 0)
        self.assertEqual(result["accuracy"], 0)

    def test_bootstrap(self):
        gold = [scene("s1", "f1"), scene("s2", "f2")]
        result = paired_family_bootstrap(gold, [prediction("s1"), prediction("s2")], [], iterations=100, seed=1)
        self.assertEqual(result["difference_a_minus_b"], 1)


if __name__ == "__main__":
    unittest.main()
