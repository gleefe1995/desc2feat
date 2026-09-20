`desc2feat/vendor/repvgg.py` is copied from the local
`EfficientLoFTR/src/loftr/backbone/repvgg.py`; the unused `loguru` import was
removed. Its original attribution to RepVGG is retained. The license distributed
with the supplied EfficientLoFTR checkout is reproduced verbatim in
`desc2feat/vendor/EfficientLoFTR.LICENSE`.

The remaining model, geometry, losses and pipeline are newly implemented.
The architecture and losses draw on ALIKE and EfficientLoFTR; these adaptations
are documented in `docs/architecture.md` and `docs/losses.md`. Neither ALIKE's
original backbone nor the full pretrained eLoFTR transformer/fine matcher is
silently substituted into this architecture.
