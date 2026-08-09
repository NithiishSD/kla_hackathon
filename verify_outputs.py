"""
Run this AFTER infer.py, before submitting -- verifies the output
directory is actually correct, not just that infer.py exited without
crashing.

Usage:
    python verify_outputs.py <test_images_dir> <output_dir>
"""

import os
import sys

import numpy as np


def main():
    if len(sys.argv) != 3:
        print("Usage: python verify_outputs.py <test_images_dir> <output_dir>")
        return
    test_dir, out_dir = sys.argv[1], sys.argv[2]

    input_files = sorted(f for f in os.listdir(test_dir) if f.endswith(".npy"))
    output_files = sorted(f for f in os.listdir(out_dir) if f.endswith(".npy"))

    print(f"Input files:  {len(input_files)}")
    print(f"Output files: {len(output_files)}")

    missing = set(input_files) - set(output_files)
    extra = set(output_files) - set(input_files)
    if missing:
        print(f"\nFAIL: {len(missing)} input file(s) have NO matching output "
              f"(first 10): {sorted(missing)[:10]}")
    if extra:
        print(f"\nWARNING: {len(extra)} output file(s) don't correspond to "
              f"any input (first 10) -- stale files from a previous run?: "
              f"{sorted(extra)[:10]}")
    if not missing and not extra:
        print("Filename match: OK -- every input has exactly one output, no extras.")

    print("\nChecking each output file for correctness...")
    problems = []
    for fname in output_files:
        in_path = os.path.join(test_dir, fname)
        out_path = os.path.join(out_dir, fname)
        try:
            in_arr = np.load(in_path)
            out_arr = np.load(out_path)
        except Exception as e:
            problems.append((fname, f"failed to load: {e}"))
            continue

        if np.isnan(out_arr).any():
            problems.append((fname, "contains NaN"))
            continue
        if np.isinf(out_arr).any():
            problems.append((fname, "contains Inf"))
            continue
        if out_arr.std() < 1e-6:
            problems.append((fname, f"degenerate output (near-constant, "
                                     f"std={out_arr.std():.2e}) -- model may "
                                     f"have failed silently on this input"))
            continue

        expected_h, expected_w = in_arr.shape[0] * 2, in_arr.shape[1] * 2
        if out_arr.shape != (expected_h, expected_w):
            problems.append((fname, f"shape mismatch: got {out_arr.shape}, "
                                     f"expected {(expected_h, expected_w)}"))
            continue

    if problems:
        print(f"\nFAIL: {len(problems)} file(s) have problems:")
        for fname, issue in problems[:20]:
            print(f"  {fname}: {issue}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
    else:
        print(f"All {len(output_files)} output files passed: correct shape, "
              f"no NaN/Inf, no degenerate outputs.")

    # Report overall value range across a sample, for a sanity eyeball --
    # should look like physical intensity units, not [0,1] or garbage.
    sample_files = output_files[:20]
    all_vals = np.concatenate([np.load(os.path.join(out_dir, f)).ravel()
                                for f in sample_files])
    print(f"\nValue range across {len(sample_files)} sampled outputs: "
          f"[{all_vals.min():.4f}, {all_vals.max():.4f}] "
          f"(should look like raw physical intensity units -- if this is "
          f"suspiciously close to exactly [0.0, 1.0], denormalization may "
          f"not have run correctly)")

    if not missing and not extra and not problems:
        print("\n=== READY TO SUBMIT ===")
    else:
        print("\n=== DO NOT SUBMIT YET -- fix the issues above first ===")


if __name__ == "__main__":
    main()