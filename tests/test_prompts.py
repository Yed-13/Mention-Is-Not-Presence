import unittest
from prompts import build

ROW={"scene_id":"s1","family_id":"f1","text":"林青站着。","candidates":[{"candidate_id":"c1","mention":"林青","entity_type":"person"}]}
class PromptTests(unittest.TestCase):
    def test_same_contract(self):
        a=build([ROW],"direct")[0]; b=build([ROW],"guideline")[0]
        self.assertEqual(a["messages"][1],b["messages"][1])
        self.assertIn("depicted",a["messages"][0]["content"])
        self.assertIn("depicted",b["messages"][0]["content"])
    def test_preserves_candidate(self): self.assertIn('"candidate_id": "c1"',build([ROW],"direct")[0]["messages"][1]["content"])
if __name__ == "__main__": unittest.main()
