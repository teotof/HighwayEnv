import argparse
from pathlib import Path

import imageio.v2 as imageio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a single PNG frame from an MP4 render."
    )
    parser.add_argument("--video", required=True, help="Path to input MP4.")
    parser.add_argument("--output", required=True, help="Path to output PNG.")
    parser.add_argument(
        "--frame-index",
        type=int,
        default=-1,
        help="Exact frame index to extract. If negative, uses --time-sec.",
    )
    parser.add_argument(
        "--time-sec",
        type=float,
        default=2.0,
        help="Time in seconds to extract when --frame-index is not given.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = imageio.get_reader(str(video_path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30.0))

    if args.frame_index >= 0:
        frame_index = args.frame_index
    else:
        frame_index = max(0, int(round(args.time_sec * fps)))

    frame = reader.get_data(frame_index)
    imageio.imwrite(output_path, frame)
    reader.close()
    print(output_path)


if __name__ == "__main__":
    main()
