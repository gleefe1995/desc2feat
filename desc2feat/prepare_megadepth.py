"""Convert trusted LoFTR/eLoFTR MegaDepth scene_info NPZ files to JSONL pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def convert(npz_root, scene_list, data_root, output, min_overlap=.4, pairs_per_scene=0,
            seed=42, bidirectional=False, check_paths=False, pose_only=False):
    npz_root, data_root, output = Path(npz_root), Path(data_root).resolve(), Path(output)
    names = [line.split()[0] for line in Path(scene_list).read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not names:
        raise ValueError("Scene list is empty.")
    if len(names) != len(set(names)):
        raise ValueError("Scene list contains duplicates.")
    generator = np.random.default_rng(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count, scenes = 0, 0
    with temporary.open("w", encoding="utf-8") as destination:
        for name in names:
            filename = name if name.endswith(".npz") else name + ".npz"
            path = npz_root / filename
            # Upstream scene_info stores pair tuples as pickled NumPy objects.
            # Use only trusted scene metadata, just as with upstream training.
            with np.load(path, allow_pickle=True) as info:
                required = {"image_paths", "intrinsics", "poses", "pair_infos"}
                if not pose_only:
                    required.add("depth_paths")
                missing = required - set(info.files)
                if missing:
                    raise ValueError(f"{path} is missing scene arrays: {sorted(missing)}")
                pair_infos = info["pair_infos"]
                eligible = [i for i, pair in enumerate(pair_infos) if float(pair[1]) > min_overlap]
                if pairs_per_scene and len(eligible) > pairs_per_scene:
                    eligible = sorted(generator.choice(eligible, pairs_per_scene, replace=False).tolist())
                image_paths = info["image_paths"]
                depth_paths = info["depth_paths"] if not pose_only else None
                intrinsics, poses = info["intrinsics"], info["poses"]
                if eligible:
                    scenes += 1
                for index in eligible:
                    pair = pair_infos[index]
                    idx0, idx1 = map(int, pair[0])
                    directions = [(idx0, idx1), (idx1, idx0)] if bidirectional else [(idx0, idx1)]
                    for source, target in directions:
                        paths = {"image0": data_root / str(image_paths[source]),
                                 "image1": data_root / str(image_paths[target])}
                        if not pose_only:
                            paths.update({"depth0": data_root / str(depth_paths[source]),
                                          "depth1": data_root / str(depth_paths[target])})
                        if check_paths:
                            for file in paths.values():
                                if not file.is_file():
                                    raise FileNotFoundError(file)
                        row = {"pair_id": f"{Path(name).stem}:{source}:{target}", "scene_id": Path(name).stem,
                               **{key: str(file) for key, file in paths.items()},
                               "K0": np.asarray(intrinsics[source]).reshape(3, 3).tolist(),
                               "K1": np.asarray(intrinsics[target]).reshape(3, 3).tolist(),
                               "T_0to1": (np.asarray(poses[target]).reshape(4, 4) @ np.linalg.inv(np.asarray(poses[source]).reshape(4, 4))).tolist(),
                               "overlap": float(pair[1])}
                        destination.write(json.dumps(row, allow_nan=False) + "\n")
                        count += 1
    if not count:
        raise ValueError("No pairs passed the overlap threshold; output was not replaced.")
    temporary.replace(output)
    return {"scenes": scenes, "pairs": count, "output": str(output), "min_overlap_exclusive": min_overlap,
            "pairs_per_scene": pairs_per_scene, "seed": seed, "bidirectional": bidirectional, "pose_only": pose_only}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-root", required=True)
    parser.add_argument("--scene-list", required=True, help="One scene name or NPZ filename per line, using an official split")
    parser.add_argument("--data-root", required=True, help="Prefix prepended to image_paths and depth_paths in scene_info")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-overlap", type=float, default=.4, help="Retain scores strictly above this threshold; use 0 for standard evaluation")
    parser.add_argument("--pairs-per-scene", type=int, default=0, help="Deterministically subsample each scene; 0 includes all eligible pairs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bidirectional", action="store_true", help="Also write reverse pairs to train both source roles")
    parser.add_argument("--check-paths", action="store_true")
    parser.add_argument("--pose-only", action="store_true", help="Export evaluation cameras without requiring depth_paths or depth files")
    args = parser.parse_args(argv)
    if not 0 <= args.min_overlap <= 1 or args.pairs_per_scene < 0:
        parser.error("min-overlap must be in [0,1] and pairs-per-scene must be nonnegative")
    print(json.dumps(convert(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
