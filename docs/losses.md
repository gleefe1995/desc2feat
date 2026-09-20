# Training objectives and geometry

This implementation combines all four enabled loss families in the local
`ALIKE_training/training/train.py` recipe with the three eLoFTR matching loss
families. It **does not reproduce either upstream training procedure exactly**.
The architecture, sparse query assignment, descriptor-map resolution, local
matching distribution and geometry conventions differ. An experiment should
identify this as the Desc2Feat adaptation, not an unchanged ALIKE/eLoFTR baseline.

`Desc2FeatLoss(config["loss"])(output, batch)` returns the differentiable total
`loss` and the unweighted `peaky`, `reprojection`, `repeatability`, `descriptor`,
`coarse`, `fine`, `local` components. Disabled components are zero. Every default
is exposed in `desc2feat/losses.py::DEFAULT_LOSS_CONFIG`; unknown keys are errors.

## Coordinate and visibility contract

- Images are `B×1×H×W`, masks are `B×H×W`, and keypoints are `B×K×2` **pixel
  (x,y)** coordinates. Integer coordinates are pixel centers. All sampling uses
  `align_corners=False`: `grid = 2*(pixel + .5)/(W,H) - 1`.
- A feature cell `(j,i)` represents `((j+.5)*W/Wf-.5,
  (i+.5)*H/Hf-.5)`. Thus stride-eight cell `(0,0)` represents `(3.5,3.5)`.
  Full image dimensions include padding. Original image masks remove padding.
- Supervision accepts either `H_0to1` (`B×3×3`), or `depth0`, `depth1`
  (`B×H×W`), `K0`, `K1` (`B×3×3`), `T_0to1` (`B×4×4`). Intrinsics must refer
  to the resized images. Depth is camera z, and its length unit must agree with
  translation. The rigid transform maps camera0 coordinates to camera1.
- Depth projection uses bilinear depth sampling, positive finite depth, and
  target relative depth consistency below `depth_relative_threshold=0.2`.
  Sampling requires valid support; holes, padding and inconsistent depths are
  ignored. A calculable projection outside the target pixel-center rectangle
  supplies an out-of-view descriptor target. Occluded/depth-unknown points are
  **not** silently relabeled as out-of-view.
- `warp_keypoints` returns `(warped, valid)`. Its projected coordinates retain
  gradients; only validity/assignment decisions are detached.
  `warp_keypoints_with_visibility` additionally returns `known_outside`.
- Sampling, coordinate transforms, descriptor distributions and loss arithmetic
  use float32 even inside mixed precision (float64 sampling/geometry is preserved
  for numerical checks). Casting normalized grids to fp16/bfloat16 can produce
  significant image-coordinate errors, so `sample_map` explicitly avoids it.

## ALIKE-family terms

| Component | Definition in this implementation | Default weight |
| --- | --- | ---: |
| Peaky | Mean normalized local detector dispersion for valid points with detached score above 0.1, in both images | 0.5 |
| Reprojection localization | Symmetric pixel L1 error for mutual nearest geometric detector pairs within 5 pixels, with both detached scores above 0.1 | 1.0 |
| Descriptor-guided score repeatability | The score product at a source keypoint and its projected target location weights `1 - descriptor_reliability`; score products are normalized per image/direction | 1.0 |
| Neural reprojection descriptor distribution (NRE) | Negative log probability of the projected target under the descriptor similarity distribution, or the explicit out-of-view bin | 5.0 |

The four weights, score threshold, match radius and temperatures follow the local
ALIKE **training script** (`w_pk=.5`, `w_rp=1`, `w_sp=1`, `w_ds=5`, `sc_th=.1`,
`train_gt_th=5`, `temp_sp=.1`, `temp_ds=.1`). In particular, the descriptor class
constructor's default `.02` is not the training script's `.1`.

The reprojection assignment uses detached symmetric Euclidean distances.
After choosing a mutual pair, its symmetric L1 error is recomputed from the
live keypoint coordinates and live geometric projections. Detaching the warp
itself would prevent the intended localization learning and is not done.

Descriptor reliability at the projected point is bilinear interpolation of
`exp((cosine_similarity - 1)/repeatability_temperature)`. This reliability is
detached, as in ALIKE, to guide score learning. The sampled target score and
source score retain gradients; the geometric sampling target is detached.

NRE uses normalized query descriptors against the other image's unconditioned
descriptor map. For visible keypoints it applies softmax over valid target cells
and minimizes **negative log of the bilinearly interpolated probability**. This
is not interpolation of log probabilities. The four neighboring cells are
border-clamped at the map edges; masked interpolation support is excluded and
remaining weights renormalized. Probability interpolation is evaluated stably
with log-sum-exp. For known-outside keypoints it appends the upstream fixed
cosine-score `1` dustbin and minimizes its negative log probability. Following
the ALIKE reference, this bin is included for out-of-view examples only.

NRE is evaluated in query chunks of `descriptor_chunk_size=128` to limit
individual working tensors. This does not eliminate the training autograd memory
needed for all descriptor distributions. Both directions are supervised. The
map is coarse here, rather than ALIKE's full-resolution descriptor map, and this
baseline uses the detected query set without upstream auxiliary random queries.
Those are material adaptations to report in ablations.

## eLoFTR-family terms

| Component | Sparse-query adaptation | Default weight |
| --- | --- | ---: |
| Coarse focal | Positive focal loss at each visible source query's projected target cell in the `K×(HcWc)` dual-softmax confidence | 1.0 |
| Fine focal | Positive focal loss for the nearest valid target sample in the query's local fine window | 1.0 |
| Local subpixel L2 | Squared error of the expected local 3×3 target position, normalized by local sample spacing | 0.5 |

Focal defaults are `focal_alpha=.25`, `focal_gamma=2`, `pos_weight=1`,
`neg_weight=1`, `coarse_sparse_supervision=True`, and
`fine_sparse_supervision=True`, matching upstream eLoFTR's defaults. Here
“sparse supervision” means **positive-only focal terms**, not the architectural
sparse-to-dense query design. Optional dense-negative focal supervision is
available by setting the corresponding option to `False`; it allocates a full
negative mask. As in the upstream implementation, alpha multiplies both
positive and negative terms.

Coarse ground-truth cells use `floor((warped_pixel+.5)*(Wc/W,Hc/H))`. Assignment
and visibility are detached; gradients flow through query descriptors,
transformer context and their differentiable keypoint sampling. Multiple source
queries can legitimately share one coarse target cell; all their valid labels
are retained. The resulting competition in column softmax is an architectural
tradeoff to test, especially with many nearby detector points.

Fine logits are already temperature-scaled by the model. The fine loss accepts
ground truth inside the half-cell-extended fine window, and supervises its
nearest valid sample. The local regression requires valid geometry, a valid
center sample, ground truth inside available local sample support, and
`max(abs(normalized_target_offset)) < local_correct_threshold` (default `1`).
It trains the model's expected local coordinate. This is the one-source-query
counterpart of eLoFTR's patch-to-patch focal and local expectation regression;
it does not instantiate eLoFTR's full fine patch-pair confidence matrix.

All seven families are enabled by default. Empty or wholly masked supervision
returns graph-connected zeros; no fabricated positive match is inserted into the
loss. Ground-truth candidate injection in the model's training fine sampler is
a separate mechanism. The provided tests check non-square half-pixel sampling,
projection gradients, depth visibility versus out-of-view targets, NRE probability
interpolation and dustbins, all seven components, and empty/masked backward passes.

The combined weights inherit two different reference recipes and are starting
settings, not experimentally validated joint-loss calibration. Log each component,
its gradient scale, detector repeatability, and downstream matching/localization
metrics; include objective-by-objective ablations before claiming the combined
training improves accuracy or difficult scenes.
