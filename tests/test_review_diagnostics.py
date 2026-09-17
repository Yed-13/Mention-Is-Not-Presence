import unittest
from review_diagnostics import positions, evidence_scores


class EvidenceDiagnosticsTests(unittest.TestCase):
    def test_repeated_spans(self):
        self.assertEqual(positions('ab ab',['ab']),{0,1,3,4})

    def test_partial_overlap(self):
        gold=[{'scene_id':'s','text':'abcd','candidates':[{'candidate_id':'a','state':'visible','evidence':['abcd']}]}]
        pred=[{'scene_id':'s','candidates':[{'candidate_id':'a','state':'visible','evidence':['ab']}]}]
        score=evidence_scores(gold,pred)
        self.assertEqual(score['character_precision'],1.)
        self.assertEqual(score['character_recall'],.5)
        self.assertAlmostEqual(score['character_f1'],2/3)

    def test_missing(self):
        gold=[{'scene_id':'s','text':'a','candidates':[{'candidate_id':'a','state':'visible','evidence':['a']}]}]
        self.assertEqual(evidence_scores(gold,[])['character_f1'],0.)
