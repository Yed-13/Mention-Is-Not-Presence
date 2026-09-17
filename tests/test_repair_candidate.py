import copy
from pathlib import Path
import unittest
from prepare_repaired_corpus import repair, animal_unit
from scriptbreak_eval import read_jsonl, validate
from lexical_baselines import ngrams

ROOT=Path(__file__).resolve().parents[1]


class RepairCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=read_jsonl(ROOT/'data/synthetic/corpus.jsonl')
        cls.repaired,cls.changes=repair(cls.original)

    def test_unique_and_disjoint(self):
        self.assertEqual(len({r['text'] for r in self.repaired}),400)
        groups={s:{r['text'] for r in self.repaired if r['split']==s} for s in ['train','dev','test']}
        for a,b in [('train','test'),('train','dev'),('dev','test')]: self.assertFalse(groups[a]&groups[b])

    def test_uncertainty_diversity(self):
        texts={r['text'] for r in self.repaired if any(c['state']=='uncertain' for c in r['candidates'])}
        self.assertEqual(len(texts),20)

    def test_reference_integrity_and_state_preservation(self):
        self.assertTrue(validate(self.repaired,require_reference=True))
        for a,b in zip(self.original,self.repaired):
            self.assertEqual([(c['candidate_id'],c['state']) for c in a['candidates']],
                             [(c['candidate_id'],c['state']) for c in b['candidates']])

    def test_original_not_mutated(self):
        before=copy.deepcopy(self.original)
        repair(self.original)
        self.assertEqual(before,self.original)

    def test_title_agreement(self):
        for r in self.repaired:
            if r['template_id']=='F05' and r['scene_id'].endswith('_1'):
                c=next(c for c in r['candidates'] if c['candidate_id']=='xuning')
                self.assertEqual(c['aliases'],[c['mention'][0]+'老师'])
                self.assertIn(c['aliases'][0],r['text'])

    def test_recorded_ngram_range(self):
        self.assertEqual(max(map(len,ngrams('abcde'))),4)
        self.assertNotIn('abcde',ngrams('abcde'))

    def test_animal_units(self):
        self.assertEqual(animal_unit('白马'),'匹')
        self.assertEqual(animal_unit('黑狗'),'只')
        self.assertEqual(animal_unit('水牛'),'头')


if __name__=='__main__': unittest.main()
