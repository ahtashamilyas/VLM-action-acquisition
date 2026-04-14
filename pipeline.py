import argparse
import re
import logging
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ActionConfig, KeyframeConfig, OllamaConfig, PipelineConfig, RGBDConfig
)
from core.rgbd_preprocessing import fuse_rgbd_frame, iter_video_frames, load_rgbd_frame
from core.keyframe_extraction import KeyframeExtractor
from core.vlm_client import OllamaVLMClient, parse_vlm_response
from core.action_serializer import save_json, print_summary
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ── Core Pipeline Function ─────────────────────────────────────────────────────

def run_pipeline(
    cfg: PipelineConfig,
    frame_iterator,          # yields (frame_idx, fps, rgb, depth)
    source_name: str = "unknown",
    dry_run: bool = False,
) -> dict:
    log.info(f"Pipeline start — source: {source_name}")
    extractor = KeyframeExtractor(cfg.keyframe, depth_scale=cfg.rgbd.depth_scale)
    total_frames = 0
    fps_seen = 30.0
    # ── Stage 1 & 2: RGBD Preprocessing + Frame Ingestion ─────────────────
    log.info("Stage 1/3: RGBD preprocessing & keyframe ingestion…")
    for frame_idx, fps, rgb, depth in frame_iterator:
        fps_seen = fps
        fused = fuse_rgbd_frame(rgb, depth, cfg.rgbd)
        extractor.ingest(frame_idx, fps, rgb, depth, fused)
        total_frames += 1
        if total_frames % 50 == 0:
            log.info(f"  Ingested {total_frames} frames…")

    log.info(f"  Total frames: {total_frames}")

    # ── Stage 2: Boundary Detection + Keyframe Selection ──────────────────
    log.info("Stage 2/3: Action boundary detection & keyframe selection…")
    segments = extractor.finalize()
    bundles_by_seg = extractor.get_all_bundles(segments)

    log.info(f"  Detected {len(segments)} action segments, "
             f"{sum(len(v) for v in bundles_by_seg.values())} keyframes total")

    if not segments:
        log.warning("No segments detected — check velocity threshold or input data")
        return save_json([], cfg.output_dir + "/empty_result.json",
                         cfg.task_description, source_name, total_frames, fps_seen)

    # ── Stage 3: VLM Inference ──────────────────────────────────────────────
    log.info("Stage 3/3: VLM action detection…")
    actions = []

    if dry_run:
        log.warning("DRY RUN: skipping VLM calls")
        from core.vlm_client import AKGAction
        for seg_id, seg in enumerate(segments):
            actions.append(AKGAction(
                segment_id=seg_id,
                action="[demo]",
                start_time_s=bundles_by_seg[seg_id][0].timestamp_s if bundles_by_seg[seg_id] else 0.0,
                end_time_s=bundles_by_seg[seg_id][-1].timestamp_s if bundles_by_seg[seg_id] else 1.0,
                confidence=0.5,
            ))
    else:
        client = OllamaVLMClient(cfg.ollama)
        for seg_id, (seg, bundles) in enumerate(
            zip(segments, bundles_by_seg.values())
        ):
            log.info(f"  Querying VLM for segment {seg_id+1}/{len(segments)} "
                     f"(frames {seg.start_frame}–{seg.end_frame})…")
            try:
                raw = client.query_segment(bundles, seg, cfg.task_description, cfg.action)
                action = parse_vlm_response(raw, seg, seg_id, bundles, cfg.action)
                if action:
                    actions.append(action)
                    log.info(f"    → {action.action} (conf={action.confidence:.2f})")
                else:
                    log.warning(f"    → No valid action extracted for segment {seg_id}")
            except Exception as e:
                log.error(f"    VLM error on segment {seg_id}: {e}")

    # ── Output ──────────────────────────────────────────────────────────────
    out_path = os.path.join(
        cfg.output_dir,
        Path(source_name).stem + "_actions.json"
    )
    data = save_json(
        actions,
        output_path=out_path,
        task_description=cfg.task_description,
        source_video=source_name,
        total_frames=total_frames,
        video_fps=fps_seen,
    )
    print_summary(data)
    return data


# ── CLI Entry Point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="REFLECT RGBD→VLM→AKG Action Extraction Pipeline"
    )

    # Input sources (mutually exclusive)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Path to RGB video file (.mp4, .avi, etc.)")
    src.add_argument("--rgb_dir", help="Directory of RGB PNG frames")
    src.add_argument("--reflect_data", help="Path to REFLECT dataset root")
    src.add_argument("--synthetic", action="store_true",
                     help="Use synthetic data (no real input needed)")

    # Optional depth
    parser.add_argument("--depth_dir", default=None,
                        help="Directory of 16-bit depth PNG frames (for --video or --rgb_dir)")

    # Dataset filtering
    parser.add_argument("--task", default=None,
                        help="Filter REFLECT episodes by task name")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of synthetic episodes to generate")

    # VLM / pipeline config
    parser.add_argument("--ollama_url", default="url webserver",
                        help="Ollama server URL (can be OpenWebUI)")
    parser.add_argument("--user_id", default="email",
                        help="Your university email (OpenWebUI login identity)")
    parser.add_argument("--api_key", default="api_key",
                        help="OpenWebUI API key (sk-xxxx) — get from Settings → Account → API Keys")
    parser.add_argument("--model", default="openchat:7b",
                        help="Ollama vision model name")
    parser.add_argument("--task_desc", default="a kitchen robot performing a meal preparation task",
                        help="Natural language task description for VLM prompt")
    parser.add_argument("--depth_strategy", choices=["hha", "depth_as_text", "depth_channel"],
                        default="hha", help="RGBD depth encoding strategy")
    parser.add_argument("--output_dir", default="output",
                        help="Directory for JSON output files")
    parser.add_argument("--frame_step", type=int, default=1,
                        help="Process every N-th frame (1=all, 3=every 3rd)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip VLM calls (test preprocessing only)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build config
    cfg = PipelineConfig(
        ollama=OllamaConfig(
            base_url=args.ollama_url,
            user_id=args.user_id,
            api_key=args.api_key,
            model=args.model
        ),
        rgbd=RGBDConfig(depth_strategy=args.depth_strategy),
        keyframe=KeyframeConfig(),
        action=ActionConfig(),
        task_description=args.task_desc,
        output_dir=args.output_dir,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dispatch by input source ─────────────────────────────────────────────
    if args.synthetic:
        from data.reflect_loader import SyntheticREFLECTDataset
        dataset = SyntheticREFLECTDataset(n_episodes=args.episodes)
        for ep in dataset.list_episodes():
            cfg.task_description = ep.task_description
            run_pipeline(
                cfg,
                frame_iterator=dataset.iter_frames(ep, step=args.frame_step),
                source_name=f"{ep.task_name}_{ep.episode_id}",
                dry_run=args.dry_run,
            )

    elif args.reflect_data:
        from data.reflect_loader import REFLECTDataset
        dataset = REFLECTDataset(args.reflect_data)
        for ep in dataset.list_episodes(task_filter=args.task):
            cfg.task_description = ep.task_description
            run_pipeline(
                cfg,
                frame_iterator=dataset.iter_frames(ep, step=args.frame_step),
                source_name=f"{ep.task_name}_{ep.episode_id}",
                dry_run=args.dry_run,
            )

    elif args.video:
        run_pipeline(
            cfg,
            frame_iterator=iter_video_frames(
                args.video, args.depth_dir, step=args.frame_step
            ),
            source_name=args.video,
            dry_run=args.dry_run,
        )

    elif args.rgb_dir:
        def rgb_dir_iter(rgb_dir, depth_dir, step):
            from pathlib import Path
            import cv2
            files = sorted(Path(rgb_dir).glob("*.png")) + \
                    sorted(Path(rgb_dir).glob("*.jpg"))
            depth_p = Path(depth_dir) if depth_dir else None
            for i, f in enumerate(files):
                if i % step != 0:
                    continue
                rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
                depth_path = (depth_p / f.name) if depth_p and (depth_p / f.name).exists() else None
                depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH).astype("uint16") \
                    if depth_path else __import__("numpy").zeros(rgb.shape[:2], dtype="uint16")
                yield i, 30.0, rgb, depth

        run_pipeline(
            cfg,
            frame_iterator=rgb_dir_iter(args.rgb_dir, args.depth_dir, args.frame_step),
            source_name=args.rgb_dir,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
