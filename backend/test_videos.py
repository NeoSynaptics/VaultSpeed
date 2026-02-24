"""Batch-test all saved raw videos through the analyzer pipeline."""
import sys
import time
from pathlib import Path

from analyzer import analyze_video

VIDEOS_DIR = Path(__file__).parent / "saved_videos"
OUTPUT_DIR = Path(__file__).parent / "test_output"


def _detect_source(filename: str) -> str:
    """Detect source from filename tag (_lib_ or _cam_)."""
    if "_lib_" in filename:
        return "library"
    if "_cam_" in filename:
        return "camera"
    # Old filenames without source tag — default to camera
    return "camera"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw_files = sorted(VIDEOS_DIR.glob("*_raw.mp4"))

    if not raw_files:
        print("No raw videos found in saved_videos/")
        sys.exit(1)

    print(f"Found {len(raw_files)} raw videos\n")
    print(f"{'File':<50} {'Src':<4} {'Avg':>6} {'Peak':>6} {'Trim':>5} {'FPS':>4} {'Time':>6}")
    print("-" * 90)

    results = []
    for raw in raw_files:
        source = _detect_source(raw.name)
        out = OUTPUT_DIR / raw.name.replace("_raw", "_test_annotated")
        t0 = time.time()
        try:
            stats = analyze_video(str(raw), str(out), source=source)
            elapsed = time.time() - t0
            print(
                f"{raw.name:<50} {source[:3]:<4} {stats['avg_kmh']:>5.1f} {stats['peak_kmh']:>5.1f} "
                f"{stats['trimmed_seconds']:>4.1f}s {stats['fps']:>4.0f} {elapsed:>5.1f}s"
            )
            results.append((raw.name, stats, None))
        except Exception as e:
            elapsed = time.time() - t0
            print(f"{raw.name:<50} {source[:3]:<4} ERROR: {e}  ({elapsed:.1f}s)")
            results.append((raw.name, None, str(e)))

    print("\n" + "=" * 90)
    ok = sum(1 for _, s, e in results if s and s["peak_kmh"] > 5)
    print(f"Results: {ok}/{len(results)} videos with peak > 5 km/h")
    print(f"Test outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
