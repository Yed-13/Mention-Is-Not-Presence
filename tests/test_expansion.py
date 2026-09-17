import unittest

from lexical_baselines import CharNgramNB, predict_rows
from prepare_expansion_experiments import balanced_family_subset
from prompts import CONDITIONS, build


def row(scene_id, family_id, template_id, state, text="甲站在门边"):
    return {
        "scene_id": scene_id,
        "family_id": family_id,
        "template_id": template_id,
        "phenomenon": template_id,
        "text": text,
        "status": "approved_synthetic",
        "annotation_source": "programmatic_by_construction",
        "candidates": [{"candidate_id": "c", "mention": "甲", "entity_type": "person",
                        "state": state, "evidence": ["甲"]}],
    }


class ExpansionTests(unittest.TestCase):
    def test_prompt_variants_keep_contract(self):
        sample = row("s", "f", "F01", "visible")
        for condition in CONDITIONS:
            built = build([sample], condition)[0]
            self.assertEqual(built["condition"], condition)
            self.assertIn("candidate_id", built["messages"][0]["content"])
            self.assertIn("s", built["messages"][1]["content"])

    def test_balanced_subset_preserves_whole_families(self):
        rows = []
        for template in ("F01", "F02"):
            for family in range(4):
                for member in range(2):
                    rows.append(row(f"{template}-{family}-{member}", f"{template}-{family}", template, "visible"))
        subset = balanced_family_subset(rows, .5, 7)
        counts = {}
        for item in subset:
            counts[item["family_id"]] = counts.get(item["family_id"], 0) + 1
        self.assertEqual(len(subset), 8)
        self.assertTrue(all(count == 2 for count in counts.values()))
        self.assertEqual({item["template_id"] for item in subset}, {"F01", "F02"})

    def test_lexical_baselines_emit_complete_schema(self):
        train = [row("s1", "f1", "F01", "visible"),
                 row("s2", "f2", "F02", "referenced_only", "甲没有出现")]
        model = CharNgramNB().fit(train)
        self.assertIn(model.predict("类型=person\n〈实体〉甲〈/实体〉没有出现"),
                      {"visible", "referenced_only"})
        output = predict_rows(train, train, "majority")
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0]["candidates"][0]["evidence"], ["甲"])


if __name__ == "__main__":
    unittest.main()
