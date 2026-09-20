# Data and command reference

Run commands from the new `desc2feat/` project directory. Python 3.9+, PyTorch 2.4+, NumPy and Pillow are required for installation. JSON configs need no YAML dependency. `pip install -e '.[train,eval]'` installs the optional YAML, HDF5 and OpenCV dependencies declared by this project. Only load checkpoints from a trusted source: training checkpoints contain Python and NumPy RNG state and use PyTorch's full checkpoint loader.

## Manifest schema

Use one JSON object per line. Paths resolve relative to the manifest directory. Training and validation must be disjoint at the scene level. A batch must contain a single geometry type; use separate homography and depth manifests.

Homography supervision, mapping original image 0 pixel centers into original image 1 pixel centers:

```json
{"pair_id":"scene-a-0-1","image0":"images/a.jpg","image1":"images/b.jpg","H_0to1":[[1,0,5],[0,1,2],[0,0,1]]}
```

Metric depth and camera supervision:

```json
{"image0":"images/0.jpg","image1":"images/1.jpg","depth0":"depths/0.h5","depth1":"depths/1.h5","K0":[[800,0,415.5],[0,800,311.5],[0,0,1]],"K1":[[800,0,415.5],[0,800,311.5],[0,0,1]],"T_0to1":"transforms/0_1.npy"}
```

`K0`, `K1` are original-resolution intrinsics. `T_0to1` is a 4×4 transform **from camera 0 coordinates to camera 1 coordinates**. For camera-from-world extrinsics `T0`, `T1`, compute `T_0to1 = T1 @ inverse(T0)`. Depth is camera-z depth, in the same length units as translation. It must have the same original height and width as its image. Zero, negative, and nonfinite depth values are invalid. HDF5 defaults to the dataset `depth` and needs `h5py`.

Any geometry field accepts an inline numeric array, `.npy` path, `.npz` path with `#key`, or `{"path":"arrays.npz","key":"K0"}`. Multi-array NPZ archives require an explicit key unless they contain the field's default key.

Images become grayscale floats in [0, 1]. The long side is resized to `data.resize` (including upsampling), preserving aspect ratio; use 0 for original resolution. The transform for each axis is `x_resized = (x_original + 0.5) * scale - 0.5`, matching `align_corners=False`. Intrinsics and homographies transform as `K' = A K` and `H' = A1 H inverse(A0)`. Depth uses nearest-exact interpolation. Collation pads the bottom and right to multiples of 32 and returns boolean masks; padding does not alter intrinsics. Inference NPZ coordinates are mapped back with `inverse(A)`.

The manifest interface covers MegaDepth and ScanNet pair exports; it does not download datasets or assume one upstream repository's metadata layout. Prepare train/validation/test pair lists from the official splits, respecting each dataset's access terms. Homography smoke data is only a plumbing check; it cannot establish outdoor 3D matching or localization quality.

## Convert existing MegaDepth metadata

The converter accepts the supplied eLoFTR repository's scene NPZ format (`image_paths`, `depth_paths`, `intrinsics`, `poses`, `pair_infos`). Use trusted metadata: upstream NumPy object arrays require `allow_pickle=True`.

```bash
python -m desc2feat.prepare_megadepth --npz-root /data/megadepth/scene_info --scene-list /data/splits/train.txt --data-root /data/megadepth --output /data/manifests/train.jsonl --min-overlap 0.4 --pairs-per-scene 1000 --bidirectional --check-paths
python -m desc2feat.prepare_megadepth --npz-root /data/megadepth/scene_info --scene-list /data/splits/test.txt --data-root /data/megadepth --output /data/manifests/test.jsonl --min-overlap 0 --pose-only --check-paths
```

Overlap filtering is strict (`score > threshold`), matching the supplied upstream dataset. `--pairs-per-scene` chooses a fixed deterministic subset without replacement; zero includes all eligible pairs. `--bidirectional` writes both source/target orders after subsampling and therefore doubles the pair count. This supports training both roles in an asymmetric model. It is optional and should be fixed across compared experiments. The converter uses the explicitly supplied scene list and never randomly mixes scenes between splits. Paths written into the manifest are absolute. `--pose-only` supports evaluation metadata without depth paths; pose evaluation reads only images and camera geometry. Training still requires both depth maps for camera-based supervision.

## Training

```bash
OMP_NUM_THREADS=2 python -m desc2feat.train --smoke-test --output runs/smoke
python -m desc2feat.train --config configs/megadepth.json --train-manifest /data/train.jsonl --val-manifest /data/val.jsonl --eloftr-checkpoint /weights/eloftr-training.ckpt --output runs/megadepth
python -m desc2feat.train --resume runs/megadepth/last.pt --output runs/megadepth
torchrun --standalone --nproc_per_node=2 -m desc2feat.train --config configs/megadepth.json --train-manifest /data/train.jsonl --val-manifest /data/val.jsonl --output runs/ddp
```

`--smoke-test` automatically uses the small `configs/smoke.json` when no configuration is supplied; default is two optimizer steps on deterministic 64×64 translations plus validation. CLI flags include `--epochs`, `--batch-size`, `--workers`, `--max-steps`, and `--device`. `--max-steps` is the total optimizer-update limit, including updates restored from a checkpoint. AMP runs only on CUDA. Gradient accumulation divides the last partial accumulation window by its actual size. Checkpoints include model, configuration, AdamW state, cosine schedule, AMP scaler, epoch/batch offset, global step, and RNG state for every distributed rank. Resume requires the same number of ranks, sampler, dataset and configuration for reproducibility; floating-point GPU kernels can still be nondeterministic. The original scheduler is restored on resume. Checkpoints are written after each epoch or the explicit step limit, with `last.pt` and periodic epoch snapshots. JSONL logs contain loss terms and validation loss; validation uses supervised outputs under `eval()` and never updates batch normalization.

The eLoFTR initializer loads only compatible RepVGG **training-form** backbone tensors and reports missing/skipped tensors. Newly introduced detector, transformer and refinement modules still require training. Fused deployment checkpoints are not interchangeable with training-form weights.

## Inference, pair evaluation and timing

```bash
python -m desc2feat.infer image0.jpg image1.jpg --checkpoint runs/megadepth/last.pt --output matches.npz --visualization matches.png --deploy
python -m desc2feat.evaluate /data/test.jsonl --checkpoint runs/megadepth/last.pt --output evaluation.json --deploy
python -m desc2feat.benchmark --checkpoint runs/megadepth/last.pt --image0 image0.jpg --image1 image1.jpg --resize 832 --warmup 20 --iterations 100 --deploy --output latency.json
```

Inference outputs `keypoints0`, `keypoints1`, `confidence`, `original_hw0`, and `original_hw1` arrays in a compressed NPZ. Each row in the two keypoint arrays is a match; coordinates are in original image pixels. `--allow-random` is required to run inference, evaluation or timing without a checkpoint, and marks results as untrained.

Evaluation retains all pairs, including zero-match and RANSAC failures. Homographies report forward transfer precision at 1/3/5 original pixels, estimated-homography corner-error AUC at 3/5/10 pixels, and failures. Camera pairs report symmetric epipolar precision using the maximum of the two point-to-epipolar-line distances at 1/3/5 pixels, and relative-pose AUC at 5/10/20 degrees. Pose error is the maximum of rotation and sign-ambiguous translation-direction error. Zero-baseline poses count as estimation failures. Both pair-mean precision (zero for no matches) and pooled precision are reported; pooled precision alone can hide failures. OpenCV is required for pose and RANSAC homography estimation; without it homography transfer precision is available and AUC is null. JSON `null` pair errors denote failed estimates. This is pair-level evaluation; visual localization requires a separate fixed 3D reconstruction, retrieval protocol, PnP pipeline and official benchmark evaluation.

Timing excludes image I/O, resizing, checkpoint loading and host-to-device transfer. CUDA runs synchronize each measurement and report peak allocated memory, warmup count, median and p90. Query count, image shapes and score-matrix element count are reported. Add `--profile` for a separate instrumented forward with backbone/detector, transformer, coarse and fine CPU/device activity times; these include profiler overhead and are not interchangeable with synchronized wall-clock latency. Compare methods on identical devices, image sizes, precision, synchronization, query budget and deployment settings; matrix-element savings alone are not measured speedups. Random weights change match counts and fine-stage workload, so untrained timing is only diagnostic.
