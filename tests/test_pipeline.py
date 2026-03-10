"""
Pipeline Tests
==============
Test all pipeline stages with synthetic data.
Run with: python tests/test_pipeline.py
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PipelineConfig, RGBDConfig, KeyframeConfig, ActionConfig
from core.rgbd_preprocessing import (
    depth_to_hha, depth_to_text, fuse_rgbd_frame, pil_to_base64
)
from core.keyframe_extraction import KeyframeExtractor, detect_action_boundaries
from core.vlm_client import parse_vlm_response, AKGAction
from core.action_serializer import actions_to_dict
from data.reflect_loader import SyntheticREFLECTDataset


class TestRGBDPreprocessing(unittest.TestCase):

    def setUp(self):
        self.cfg = RGBDConfig()
        self.rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.depth = np.random.randint(500, 2000, (480, 640), dtype=np.uint16)

    def test_hha_output_shape(self):
        hha = depth_to_hha(self.depth, self.cfg)
        self.assertEqual(hha.shape, (480, 640, 3))
        self.assertEqual(hha.dtype, np.uint8)

    def test_depth_text_format(self):
        text = depth_to_text(self.depth, self.rgb, self.cfg)
        self.assertIn("depth_grid", text)
        self.assertIn("center", text)

    def test_fuse_hha(self):
        result = fuse_rgbd_frame(self.rgb, self.depth, self.cfg)
        self.assertIn("pil_image", result)
        self.assertIn("depth_text", result)
        self.assertIn("depth_m", result)
        # Side-by-side image should be double width
        w, h = result["pil_image"].size
        self.assertEqual(w, 640 * 2)

    def test_fuse_depth_channel(self):
        cfg = RGBDConfig(depth_strategy="depth_channel")
        result = fuse_rgbd_frame(self.rgb, self.depth, cfg)
        w, h = result["pil_image"].size
        self.assertEqual(w, 640)

    def test_pil_to_base64(self):
        result = fuse_rgbd_frame(self.rgb, self.depth, self.cfg)
        b64 = pil_to_base64(result["pil_image"])
        self.assertIsInstance(b64, str)
        self.assertGreater(len(b64), 100)

    def test_zero_depth(self):
        depth_zero = np.zeros((480, 640), dtype=np.uint16)
        result = fuse_rgbd_frame(self.rgb, depth_zero, self.cfg)
        self.assertIsNotNone(result["pil_image"])


class TestKeyframeExtraction(unittest.TestCase):

    def setUp(self):
        self.cfg = KeyframeConfig(min_action_gap_frames=5, max_keyframes_per_action=3)

    def _make_velocity_signal(self):
        """Synthetic velocity with clear minima at action boundaries."""
        v = np.ones(100) * 0.1
        # High motion in action regions
        v[10:25] = 0.3
        v[40:60] = 0.25
        v[75:90] = 0.28
        # Low motion at boundaries
        v[[0, 28, 63, 95]] = 0.01
        return v

    def test_boundary_detection(self):
        v = self._make_velocity_signal()
        boundaries = detect_action_boundaries(v, self.cfg)
        self.assertGreaterEqual(len(boundaries), 2)
        self.assertEqual(boundaries[0], 0)
        self.assertEqual(boundaries[-1], len(v) - 1)

    def test_full_extractor(self):
        dataset = SyntheticREFLECTDataset(n_episodes=1, frames_per_episode=60, fps=10.0)
        episodes = dataset.list_episodes()
        ep = episodes[0]

        cfg_rgbd = RGBDConfig()
        extractor = KeyframeExtractor(self.cfg, depth_scale=cfg_rgbd.depth_scale)

        from core.rgbd_preprocessing import fuse_rgbd_frame
        for frame_idx, fps, rgb, depth in dataset.iter_frames(ep):
            fused = fuse_rgbd_frame(rgb, depth, cfg_rgbd)
            extractor.ingest(frame_idx, fps, rgb, depth, fused)

        segments = extractor.finalize()
        self.assertGreater(len(segments), 0)

        bundles = extractor.get_all_bundles(segments)
        for seg_id, blist in bundles.items():
            self.assertGreater(len(blist), 0)
            for b in blist:
                self.assertIsNotNone(b.fused_pil)
                self.assertIsInstance(b.depth_text, str)


class TestVLMResponseParser(unittest.TestCase):

    def _dummy_bundles(self):
        """Create minimal bundle stubs for testing."""
        from core.keyframe_extraction import KeyframeBundle
        from PIL import Image
        return [KeyframeBundle(
            frame_idx=10, timestamp_s=1.0,
            rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            depth_raw=np.zeros((480, 640), dtype=np.uint16),
            fused_pil=Image.new("RGB", (640, 480)),
            depth_text="depth_grid[center:0.80m]",
            velocity=0.05, segment_id=0,
        )]

    def _dummy_seg(self):
        from core.keyframe_extraction import ActionSegment
        return ActionSegment(
            start_frame=0, end_frame=20,
            keyframe_indices=[10], mean_velocity=0.1, depth_change=0.05
        )

    def test_clean_json(self):
        raw = json.dumps({
            "action_core": "CUTTING",
            "sub_action": "slice",
            "objects_involved": ["knife", "carrot"],
            "contact_type": "blade_contact",
            "spatial_relation": "above cutting board",
            "depth_context": "knife at 0.35m",
            "confidence": 0.92,
            "reasoning": "Blade is moving downward"
        })
        action = parse_vlm_response(raw, self._dummy_seg(), 0,
                                    self._dummy_bundles(), ActionConfig())
        self.assertIsNotNone(action)
        self.assertEqual(action.action_core, "CUTTING")
        self.assertAlmostEqual(action.confidence, 0.92)

    def test_markdown_fenced_json(self):
        raw = "```json\n" + json.dumps({
            "action_core": "POURING",
            "sub_action": "pour",
            "objects_involved": ["kettle", "pot"],
            "contact_type": "pour",
            "spatial_relation": "above pot",
            "depth_context": "kettle tip at 0.5m",
            "confidence": 0.85,
            "reasoning": ""
        }) + "\n```"
        action = parse_vlm_response(raw, self._dummy_seg(), 0,
                                    self._dummy_bundles(), ActionConfig())
        self.assertIsNotNone(action)
        self.assertEqual(action.action_core, "POURING")

    def test_unknown_action_core_normalised(self):
        raw = json.dumps({
            "action_core": "TRANSPORT",  # not in taxonomy
            "sub_action": "move",
            "objects_involved": [],
            "contact_type": "grasp",
            "spatial_relation": "",
            "depth_context": "",
            "confidence": 0.7,
            "reasoning": ""
        })
        action = parse_vlm_response(raw, self._dummy_seg(), 0,
                                    self._dummy_bundles(), ActionConfig())
        self.assertEqual(action.action_core, "PICK_AND_PLACE")

    def test_low_confidence_filtered(self):
        raw = json.dumps({
            "action_core": "MIXING",
            "sub_action": "stir",
            "objects_involved": [],
            "contact_type": "none",
            "spatial_relation": "",
            "depth_context": "",
            "confidence": 0.1,  # below threshold
            "reasoning": ""
        })
        cfg = ActionConfig(min_confidence=0.3)
        action = parse_vlm_response(raw, self._dummy_seg(), 0,
                                    self._dummy_bundles(), cfg)
        self.assertIsNone(action)

    def test_no_json_in_response(self):
        raw = "I cannot determine the action from these images."
        action = parse_vlm_response(raw, self._dummy_seg(), 0,
                                    self._dummy_bundles(), ActionConfig())
        self.assertIsNone(action)


class TestActionSerializer(unittest.TestCase):

    def _dummy_actions(self):
        return [
            AKGAction(
                segment_id=0, action_core="PICK_AND_PLACE", sub_action="transport",
                start_time_s=0.0, end_time_s=3.2,
                objects_involved=["pot", "stove"],
                contact_type="grasp", spatial_relation="above stove",
                depth_context="pot at 0.6m", confidence=0.91,
                reasoning="robot transports pot"
            ),
            AKGAction(
                segment_id=1, action_core="COOK_COOL", sub_action="heat",
                start_time_s=3.5, end_time_s=10.0,
                objects_involved=["pot", "burner"],
                contact_type="none", spatial_relation="on burner",
                depth_context="flame detected", confidence=0.88,
                reasoning="burner on"
            ),
        ]

    def test_schema_structure(self):
        data = actions_to_dict(
            self._dummy_actions(),
            task_description="boil water",
            total_frames=300, video_fps=30.0
        )
        self.assertIn("schema_version", data)
        self.assertIn("action_sequence", data)
        self.assertIn("summary", data)
        self.assertEqual(len(data["action_sequence"]), 2)
        self.assertEqual(data["summary"]["total_actions"], 2)

    def test_action_counts(self):
        data = actions_to_dict(self._dummy_actions(), task_description="test")
        counts = data["summary"]["action_counts"]
        self.assertEqual(counts.get("PICK_AND_PLACE"), 1)
        self.assertEqual(counts.get("COOK_COOL"), 1)

    def test_duration_computed(self):
        data = actions_to_dict(self._dummy_actions(), task_description="test",
                               total_frames=300, video_fps=30.0)
        self.assertAlmostEqual(data["duration_s"], 10.0)


class TestSyntheticDataset(unittest.TestCase):

    def test_generates_frames(self):
        ds = SyntheticREFLECTDataset(n_episodes=2, frames_per_episode=30, fps=5.0)
        episodes = ds.list_episodes()
        self.assertEqual(len(episodes), 2)
        frames = list(ds.iter_frames(episodes[0]))
        self.assertGreater(len(frames), 0)
        fi, fps, rgb, depth = frames[0]
        self.assertEqual(rgb.shape[2], 3)
        self.assertEqual(depth.dtype, np.uint16)

    def test_step_sampling(self):
        ds = SyntheticREFLECTDataset(n_episodes=1, frames_per_episode=60)
        ep = ds.list_episodes()[0]
        all_frames = list(ds.iter_frames(ep, step=1))
        every_3rd = list(ds.iter_frames(ep, step=3))
        self.assertAlmostEqual(len(every_3rd), len(all_frames) // 3, delta=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
