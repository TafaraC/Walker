import tempfile
import unittest
from pathlib import Path

from contact_slip.scalar_scene import convert
from contact_slip.sensitivity import plan
from variable_training.common import digest


class ScalarSceneTests(unittest.TestCase):
    def fixture(self, directory):
        scene=directory/'soccer.rsg'
        scene.write_text('(setContactSlipMode true)\n   (setContactSlip 0.1 0.1)\n(keepOtherPhysics unchanged)\n')
        sha=digest(scene)
        marker=f'WALKER_SCENE_VERIFIED slip=0.1 sha256={sha}'
        (directory/'naosoccersim.rb').write_text(f'check "{sha}"\nputs "{marker}"\n')
        (directory/'rcssserver3d.rb').write_text('original launch\n')
        return dict(sha256=sha,startup_marker=marker)

    def test_only_setter_and_integrity_references_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp); manifest=self.fixture(directory)
            result=convert(directory,manifest,.1)
            self.assertEqual((directory/'soccer.rsg').read_text(),
                             '(setContactSlipMode true)\n   (setContactSlip 0.1)\n(keepOtherPhysics unchanged)\n')
            self.assertEqual(result['sha256'],digest(directory/'soccer.rsg'))
            self.assertEqual(result['scene_setter_method'],'single_argument_v1')
            self.assertIn(result['startup_marker'],(directory/'naosoccersim.rb').read_text())
            self.assertNotIn(manifest['sha256'],(directory/'naosoccersim.rb').read_text())
            for name,sha in result['config_sha256'].items():
                self.assertEqual(digest(directory/name),sha)
            self.assertEqual(manifest['sha256'],result['legacy_generated_scene_sha256'])

    def test_reject_wrong_value_or_modified_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp); manifest=self.fixture(directory)
            with self.assertRaises(ValueError): convert(directory,manifest,.2)
            (directory/'soccer.rsg').write_text('(setContactSlip 0.1 0.1)\n')
            with self.assertRaises(ValueError): convert(directory,manifest,.1)

    def test_scalar_suite_is_isolated_oracle_only(self):
        p=plan('scalar')
        self.assertEqual(p['scene_method'],'scalar')
        self.assertEqual(p['policies'],{'oracle':None})
        self.assertEqual(p['slips'],[.025,.1,.2,1.,10.])
        self.assertNotIn('scene_method',plan('stress'))
        self.assertNotIn('scene_method',plan())


if __name__=='__main__': unittest.main()
