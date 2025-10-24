import os
import sys
import argparse


def main():
    # Ensure we can import training.model.* as a package
    v2_root = os.path.abspath(os.path.dirname(__file__))
    if v2_root not in sys.path:
        sys.path.insert(0, v2_root)

    from training.model.pipeline_inference import S2ST_Pipeline

    parser = argparse.ArgumentParser(description="Run S2ST inference on a single WAV file.")
    parser.add_argument("--input", type=str, default=None, help="Path to input .wav (default: first in v2/data/en)")
    parser.add_argument("--output", type=str, default=None, help="Path to output .wav (default: v2/results/<name>_vi.wav)")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Force device (default: auto)")
    args = parser.parse_args()

    # Resolve device
    if args.device:
        device = args.device
    else:
        device = (
            "cuda"
            if (os.environ.get("CUDA_VISIBLE_DEVICES") or __import__("torch").cuda.is_available())
            else "cpu"
        )

    # Resolve input
    if args.input:
        inp = os.path.abspath(args.input)
        if not os.path.isfile(inp):
            raise FileNotFoundError(f"Input file not found: {inp}")
    else:
        in_dir = os.path.join(v2_root, "data", "en")
        if not os.path.isdir(in_dir):
            raise FileNotFoundError(f"Missing input dir: {in_dir}")
        wavs = [f for f in os.listdir(in_dir) if f.lower().endswith('.wav')]
        if not wavs:
            raise FileNotFoundError(f"No .wav files in: {in_dir}")
        wavs.sort()
        inp = os.path.join(in_dir, wavs[0])

    # Resolve output
    if args.output:
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    else:
        out_dir = os.path.join(v2_root, 'results')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(inp))[0] + '_vi.wav')

    print(f"Device: {device}")
    print(f"Input:  {inp}")
    print(f"Output: {out_path}")

    pipe = S2ST_Pipeline(device=device)
    pipe.translate(inp, out_path)
    print("Done.")


if __name__ == '__main__':
    main()
