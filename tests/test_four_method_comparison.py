#!/usr/bin/env python3
"""Regression tests for distinct comparison groups and real cached skeletons."""
import importlib.util,json,os,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import build_four_method_comparison as module
REPORT=json.loads((ROOT/'docs/four-method-comparison.json').read_text())
OLD=json.loads((ROOT/'docs/cross-method-comparison.json').read_text())
MATERIALS=json.loads((ROOT/'docs/egohsi3r-materials.json').read_text())

class FourMethodTest(unittest.TestCase):
    def test_explicit_groups_no_four_way_ranking(self):
        self.assertIsNone(REPORT['aggregate']);self.assertEqual(REPORT['all_four_common_hand_frames'],0)
        self.assertEqual([g['id'] for g in REPORT['groups']],['dexycb','aria'])
        self.assertEqual([g['scored_methods'] for g in REPORT['groups']],
                         [['hawor','hggt','egohsi3r'],['hawor','egofound3r','hggt']])
        self.assertEqual([(g['rgb_frames'],g['gt_hand_frames'],g['common_hand_frames'],g['common_acceleration_triplets']) for g in REPORT['groups']],
                         [(146,146,146,142),(300,600,565,555)])

    def test_failed_run_and_missing_run_are_different(self):
        for s in REPORT['samples']:
            if s['group_id']=='dexycb':
                r=s['methods']['egofound3r'];self.assertTrue(r['available']);self.assertEqual(r['coverage_fraction'],0)
                self.assertEqual(r['pck50_all_gt'],0);self.assertFalse(r['included_in_common_comparison'])
                for key in module.FIELDS:self.assertIsNone(r[key])
            else:
                r=s['methods']['egohsi3r'];self.assertFalse(r['available'])
                for key in module.FIELDS+['coverage_fraction','strict_hand_frames','pck50_all_gt','correct_joints_50mm']:
                    self.assertIsNone(r[key])

    def test_aria_original_common_errors_preserved(self):
        for s in REPORT['samples'][2:]:
            old=next(x for x in OLD['samples'] if x['id']==s['id'])
            self.assertEqual(s['common_pairs'],old['common_pairs'])
            for method in ['hawor','egofound3r','hggt']:
                for field in module.FIELDS+['coverage_fraction','pck50_all_gt','strict_hand_frames']:
                    self.assertAlmostEqual(s['methods'][method][field],old['methods'][method][field],places=10)

    def test_pck_group_denominators(self):
        d,a=REPORT['groups']
        self.assertEqual(d['methods']['egohsi3r']['gt_joint_count'],2336)
        for key,n in [('hawor',1941),('egofound3r',0),('hggt',0),('egohsi3r',1776)]:
            self.assertEqual(d['methods'][key]['correct_joints_50mm'],n)
            self.assertEqual(d['methods'][key]['pck50_all_gt'],n/2336)
        for key,n in [('hawor',8604),('egofound3r',8519),('hggt',4784)]:
            self.assertEqual(a['methods'][key]['pck50_all_gt'],n/9600)
        self.assertEqual(d['methods']['hggt']['adapted_pck50_all_gt'],1544/2336)
        self.assertEqual(a['methods']['hggt']['adapted_pck50_all_gt'],8213/9600)

    def test_independent_egohsi3r_scores(self):
        # Independently recomputed from original extracted NPZ with separate SVD.
        values={'dexycb_box':[45.42094017574711,5.018893903589286,2.612454075128402,46.19076224772533,3.842621506395516],
                'dexycb_banana':[18.61233278293348,4.56375607198332,2.184716494891805,19.05897654108904,2.952551239870198]}
        for s in REPORT['samples'][:2]:
            for field,value in zip(module.FIELDS,values[s['id']]):self.assertAlmostEqual(s['methods']['egohsi3r'][field],value,places=10)
        r=REPORT['groups'][0]['methods']['egohsi3r']
        for field,value in zip(module.FIELDS,[32.20025707792121,4.794442370194563,2.4015149944637795,32.810703543082774,3.403854473601345]):
            self.assertAlmostEqual(r[field],value,places=10)

    def test_weighted_group_means(self):
        for group in REPORT['groups']:
            for method in group['scored_methods']:
                for field in module.FIELDS:
                    count='common_acceleration_triplets' if field=='rr_acceleration16_error_m_s2' else 'common_hand_frames'
                    rows=[s['methods'][method] for s in REPORT['samples'] if s['id'] in group['sample_ids']]
                    mean=sum(r[field]*r[count] for r in rows)/sum(r[count] for r in rows)
                    self.assertAlmostEqual(group['methods'][method][field],mean,places=10)

    def test_source_masks_and_sides(self):
        self.assertEqual([len(s['common_pairs']) for s in REPORT['samples']],[74,72,300,265])
        for s in REPORT['samples'][:2]:self.assertEqual({p['side'] for p in s['common_pairs']},{'right'})
        for s in REPORT['samples']:
            self.assertEqual(len({(p['source_frame_id'],p['side']) for p in s['common_pairs']}),s['common_hand_frames'])

    def test_skeleton_contract_and_projection(self):
        for sample in MATERIALS['samples']:
            path=ROOT/'docs'/sample['geometry'];g=json.loads(path.read_text())
            self.assertEqual(g['vertex_count'],0);self.assertNotIn('vertices',g);self.assertNotIn('faces',g)
            self.assertEqual(g['hand_order'],['left','right']);self.assertEqual(g['coordinate_space'],'camera');self.assertEqual(g['units'],'m')
            arrays={}
            for key in ['joints','valid','display_valid','joint_valid','joints2d','projection_K']:
                spec=g[key];p=path.parent/spec['url'];self.assertEqual(p.stat().st_size,spec['byte_length'])
                self.assertEqual(module.sha(p),spec['sha256'])
                arrays[key]=np.fromfile(p,dtype={'float32':'<f4','uint8':'u1'}[spec['dtype']]).reshape(spec['shape'])
            valid=arrays['joint_valid'].astype(bool)
            self.assertTrue((arrays['valid'][:,0]==0).all());self.assertTrue((arrays['valid'][:,1]==1).all())
            projected=np.einsum('tij,thnj->thni',arrays['projection_K'].astype(float),arrays['joints'].astype(float))
            projected=projected[...,:2]/projected[...,2:]
            np.testing.assert_allclose(projected[valid],arrays['joints2d'][valid],atol=8e-5,rtol=1e-7)

    def test_public_artifacts_exclude_machine_paths(self):
        paths=[ROOT/'docs/egohsi3r-materials.json',ROOT/'docs/four-method-comparison.json']
        paths += [ROOT/'docs'/s['geometry'] for s in MATERIALS['samples']]
        for p in paths:
            text=p.read_text()
            for pattern in ['/nvme-','/Users/','/tmp/','/home/']:self.assertNotIn(pattern,text)

    def test_gt_identity_evidence_and_frozen_run(self):
        self.assertEqual(MATERIALS['original_sequence_count'],9);self.assertEqual(MATERIALS['original_frame_count'],2126)
        for s in MATERIALS['samples']:
            self.assertTrue(s['gt_joint_arrays_exactly_equal']);self.assertTrue(s['gt_masks_exactly_equal'])
            self.assertTrue(s['timestamps_equal']);self.assertEqual(s['gt_max_difference_m'],0)
            self.assertFalse(s['input_video_bytes_identical'])
            self.assertEqual(s['base16_native_vs_official_max_difference_m'],0)
            self.assertEqual(s['camera_id'],'836212060125')
            for video in s['videos']:self.assertTrue(video['url'].startswith(MATERIALS['source_site']))

    def test_old_report_unchanged(self):
        self.assertEqual(module.sha(ROOT/'docs/cross-method-comparison.json'),REPORT['sources']['previous_report_sha256'])

    @unittest.skipUnless(os.environ.get('HGGT_SOURCE'),'Set HGGT_SOURCE to recompute from original cached inputs')
    def test_rebuild_from_cached_outputs(self):
        self.assertEqual(module.build(ROOT,Path(os.environ['HGGT_SOURCE'])),REPORT)

if __name__=='__main__':unittest.main()
