"""CLI entry point: python -m frozen_substrate [command] [args]"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="frozen_substrate",
        description="Frozen Substrate: bio-inspired persistence filter",
    )
    sub = p.add_subparsers(dest="command")

    # --- demo ---
    demo_p = sub.add_parser("demo", help="Run synthetic moving-Gaussian demo")
    demo_p.add_argument("-o", "--output", default="output", help="Output directory")
    demo_p.add_argument("--frames", type=int, default=240, help="Number of frames")
    demo_p.add_argument("--preset", choices=["default", "fast", "high_res"], default="default")
    demo_p.add_argument("--render", action="store_true", help="Render PNG sequences")

    # --- process ---
    proc_p = sub.add_parser("process", help="Process a video file or webcam")
    proc_p.add_argument("input", nargs="?", default=None, help="Video file path (omit for webcam)")
    proc_p.add_argument("-o", "--output", default="output", help="Output directory")
    proc_p.add_argument("--preset", choices=["default", "fast", "high_res"], default="default")
    proc_p.add_argument("--max-frames", type=int, default=0, help="Max frames (0=all)")
    proc_p.add_argument("--render", action="store_true", help="Render PNG sequences")
    proc_p.add_argument("--webcam", action="store_true", help="Use webcam input")

    # --- info ---
    sub.add_parser("info", help="Print package info and available presets")

    return p


def _get_configs(preset: str):
    from frozen_substrate.redesign.config import SubstrateConfig, ReadoutConfig, VideoIOConfig

    if preset == "fast":
        scfg = SubstrateConfig.fast()
    elif preset == "high_res":
        scfg = SubstrateConfig.high_res()
    else:
        scfg = SubstrateConfig.default()
    rcfg = ReadoutConfig.for_substrate(scfg)
    vcfg = VideoIOConfig()
    return scfg, rcfg, vcfg


def cmd_demo(args):
    from frozen_substrate.redesign import Pipeline
    from frozen_substrate.viz import save_cubes_npz, print_cube_summary, render_composite

    scfg, rcfg, vcfg = _get_configs(args.preset)
    pipe = Pipeline(scfg, rcfg, vcfg, seed=0)

    os.makedirs(args.output, exist_ok=True)

    cubes = []
    metas = []

    def moving_gaussian(H, W, t, period=120, sigma=2.0):
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        angle = 2 * np.pi * (t / period)
        x0 = cx + min(12.0, W * 0.24) * np.cos(angle)
        y0 = cy + min(12.0, H * 0.24) * np.sin(angle)
        y = np.arange(H)[:, None]
        x = np.arange(W)[None, :]
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)).astype(np.float32)

    print(f"Running demo: {args.frames} frames, preset={args.preset} "
          f"({scfg.height}x{scfg.width}, {scfg.n_layers} layers)")

    for t in range(args.frames):
        frame = moving_gaussian(scfg.height, scfg.width, t)
        out = pipe.process_frame(frame)
        if out is not None:
            cube, meta = out
            cubes.append(cube)
            metas.append(meta)

    if not cubes:
        print("No cubes produced (too few frames).")
        return

    cubes_arr = np.stack(cubes, axis=0)
    print_cube_summary(cubes_arr, metas)

    npz_path = os.path.join(args.output, "cubes.npz")
    save_cubes_npz(npz_path, cubes_arr, metas)
    print(f"Saved: {npz_path}")

    if args.render:
        n_a = len(metas[-1].get("a_layers", ()))
        render_composite(cubes_arr, args.output, a_channels=n_a)


def cmd_process(args):
    from frozen_substrate.redesign import Pipeline
    from frozen_substrate.viz import save_cubes_npz, print_cube_summary, render_composite

    try:
        import cv2
    except ImportError:
        print("Error: opencv-python is required for video processing.")
        print("Install it with: pip install opencv-python")
        sys.exit(1)

    scfg, rcfg, vcfg = _get_configs(args.preset)
    pipe = Pipeline(scfg, rcfg, vcfg, seed=0)
    os.makedirs(args.output, exist_ok=True)

    if args.webcam or args.input is None:
        print("Opening webcam...")
        cap = cv2.VideoCapture(0)
    else:
        if not os.path.isfile(args.input):
            print(f"Error: file not found: {args.input}")
            sys.exit(1)
        cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print("Error: could not open video source.")
        sys.exit(1)

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_video_frames <= 0:
        total_video_frames = None

    print(f"Processing: preset={args.preset} ({scfg.height}x{scfg.width}, {scfg.n_layers} layers)")
    if total_video_frames:
        limit = args.max_frames if args.max_frames > 0 else total_video_frames
        print(f"  Video frames: {total_video_frames}, will process: {min(limit, total_video_frames)}")

    cubes = []
    metas = []
    frame_idx = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = pipe.process_frame(rgb)
        if out is not None:
            cube, meta = out
            cubes.append(cube)
            metas.append(meta)
        frame_idx += 1
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break
        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}...", end="\r")

    cap.release()
    print(f"\nProcessed {frame_idx} frames -> {len(cubes)} cubes")

    if not cubes:
        print("No cubes produced.")
        return

    cubes_arr = np.stack(cubes, axis=0)
    print_cube_summary(cubes_arr, metas)

    npz_path = os.path.join(args.output, "cubes.npz")
    save_cubes_npz(npz_path, cubes_arr, metas)
    print(f"Saved: {npz_path}")

    if args.render:
        n_a = len(metas[-1].get("a_layers", ()))
        render_composite(cubes_arr, args.output, a_channels=n_a)


def cmd_info(_args):
    from frozen_substrate.redesign.config import SubstrateConfig, ReadoutConfig

    print("Frozen Substrate")
    print("================")
    print()
    print("Presets:")
    for name, factory in [("default", SubstrateConfig.default),
                          ("fast", SubstrateConfig.fast),
                          ("high_res", SubstrateConfig.high_res)]:
        cfg = factory()
        rcfg = ReadoutConfig.for_substrate(cfg)
        print(f"  {name:10s}  {cfg.height}x{cfg.width}, {cfg.n_layers} layers  "
              f"A={rcfg.a_layers} B={rcfg.b_layers}")
    print()
    print("Usage:")
    print("  python -m frozen_substrate demo              # synthetic demo")
    print("  python -m frozen_substrate demo --render     # + PNG output")
    print("  python -m frozen_substrate process video.mp4 # process video")
    print("  python -m frozen_substrate process --webcam  # webcam input")
    print("  python -m frozen_substrate info              # this message")


def main():
    parser = _make_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "demo": cmd_demo,
        "process": cmd_process,
        "info": cmd_info,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
