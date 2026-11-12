#!/usr/bin/env python3
"""Numerical and denominator regression checks for the new common comparison."""
import importlib.util,json,os,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cross_method',ROOT/'scripts/build_cross_method_comparison.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
REPORT=json.loads((ROOT/'docs/cross-method-comparison.json').read_text())

class CrossMethodTest(unittest.TestCase):
    def test_counts_and_fill_policy(self):
        a=REPORT['aggregate'];self.assertEqual((a['rgb_frames'],a['gt_hand_frames'],a['common_hand_frames'],a['common_acceleration_triplets']),(446,746,565,555))
        expected={'hawor':(715,31,746),'egofound3r':(565,35,600),'hggt':(715,0,715)}
        for k,(strict,fill,display) in expected.items():
            r=a['methods'][k];self.assertEqual((r['strict_hand_frames'],r['excluded_fill_hand_frames'],r['displayable_hand_frames']),(strict,fill,display));self.assertEqual(r['gt_joint_count'],11936)
    def test_independent_double_precision_results(self):
        # Independently computed using a separate PA-SVD implementation.
        expected={'egofound3r':[24.5048331423,16.5505436223,7.78542428384,19.3136375293,1.90622932321],
                  'hawor':[22.8469099125,13.9797068227,5.18603517915,26.2364377215,1.31551935529],
                  'hggt':[58.9130250241,16.6600011354,6.79571836555,58.2703277023,7.80512043553]}
        keys=['mpjpe16_mm','rr_mpjpe16_mm','pa_mpjpe16_mm','wrist_error_mm','rr_acceleration16_error_m_s2']
        for method,values in expected.items():
            for k,v in zip(keys,values):self.assertAlmostEqual(REPORT['aggregate']['methods'][method][k],v,places=8)
        self.assertAlmostEqual(REPORT['aggregate']['methods']['hggt']['adapted_mpjpe16_mm'],26.1266919647,places=8)
    def test_pck_full_gt_not_common_denominator(self):
        for key,n in [('hawor',10545),('egofound3r',8519),('hggt',4784)]:
            r=REPORT['aggregate']['methods'][key];self.assertEqual(r['correct_joints_50mm'],n);self.assertEqual(r['pck50_all_gt'],n/11936)
        self.assertEqual(REPORT['aggregate']['methods']['hggt']['adapted_pck50_all_gt'],9757/11936)
    def test_dex_failures_not_given_common_errors(self):
        for s in REPORT['samples'][:2]:
            self.assertEqual(s['common_hand_frames'],0)
            for r in s['methods'].values():
                for k in ['mpjpe16_mm','rr_mpjpe16_mm','pa_mpjpe16_mm','wrist_error_mm','rr_acceleration16_error_m_s2']:self.assertIsNone(r[k])
            self.assertGreater(s['methods']['hawor']['pck50_all_gt'],0)
            self.assertEqual(s['methods']['egofound3r']['pck50_all_gt'],0)
    def test_weighted_aggregation(self):
        for method in module.METHODS:
            for field in ['mpjpe16_mm','rr_mpjpe16_mm','pa_mpjpe16_mm','wrist_error_mm','rr_acceleration16_error_m_s2']:
                count='common_acceleration_triplets' if field=='rr_acceleration16_error_m_s2' else 'common_hand_frames'
                rows=[s['methods'][method] for s in REPORT['samples'] if s[count]>0]
                mean=sum(r[field]*r[count] for r in rows)/sum(r[count] for r in rows)
                self.assertAlmostEqual(mean,REPORT['aggregate']['methods'][method][field],places=10)
    def test_source_pair_masks(self):
        self.assertEqual([len(s['common_pairs']) for s in REPORT['samples']],[0,0,300,265])
        for s in REPORT['samples']:
            self.assertEqual(len({(x['source_frame_id'],x['side']) for x in s['common_pairs']}),s['common_hand_frames'])
    def test_pa_transform_and_no_reflection(self):
        x=np.random.default_rng(7).normal(size=(16,3));rot=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        y=1.8*x@rot+np.array([.1,.8,-.2]);np.testing.assert_allclose(module.procrustes(x,y),y,atol=1e-12)
        reflection=x*np.array([-1,1,1]);self.assertGreater(np.linalg.norm(module.procrustes(x,reflection)-reflection),.1)
    def test_base16_and_translation_invariance(self):
        self.assertEqual(module.BASE16,[0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19])
        truth=np.random.default_rng(8).normal(size=(5,2,21,3));pred=truth+np.arange(5)[:,None,None,None]*.7
        result=module.metric_arrays(pred,truth,np.array([0,.03,.071,.109,.15]))
        np.testing.assert_allclose(result['rr_mpjpe16_mm'],0,atol=1e-12)
        np.testing.assert_allclose(result['rr_acceleration16_error_m_s2'][1:-1],0,atol=1e-10)
    @unittest.skipUnless(os.environ.get('HGGT_SOURCE'),'Set HGGT_SOURCE to rerun from cached original predictions')
    def test_rebuild_and_source_hashes(self):
        legacy=Path(os.environ['HGGT_SOURCE']);self.assertEqual(module.build(ROOT,legacy),REPORT)
        for sample in REPORT['samples']:
            for name,digest in sample['source_sha256'].items():
                path=legacy/'docs'/name.removeprefix('hggt_hawor/') if name.startswith('hggt_hawor/') else ROOT/'docs'/name
                self.assertEqual(module.sha(path),digest)
if __name__=='__main__':unittest.main()
