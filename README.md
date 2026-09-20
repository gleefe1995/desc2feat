# Desc2Feat

eLoFTR의 RepVGG backbone에 ALIKE 방식의 subpixel detector를 결합한
**실험용 sparse-to-dense matcher**입니다. `image0`의 최대 1,000개 keypoint와
descriptor로 `image1` 전체 coarse map을 검색하고, 두 단계 target refinement로
sparse correspondences를 반환합니다.

훈련·추론·pair 평가·속도 측정 코드를 구현했습니다. 실제 MegaDepth/ScanNet 학습
가중치나 성능 비교 결과는 아직 없습니다. eLoFTR보다 정확하거나 빠르다는 주장은
이 코드만으로 성립하지 않습니다.

현재 검증 결과는 [38개 테스트와 실행 기록](docs/validation.md)에 있습니다.

## 구조와 loss

1. 공유 RepVGG + 경량 FPN으로 stride 2/4/8 feature를 추출합니다.
2. source에서 NMS/top-K와 local soft-argmax로 subpixel keypoint를 검출합니다.
3. sparse source token과 pooled target context 사이 self/cross-attention을 합니다.
4. K×HW dual-softmax, target 9×9 pixel search, 3×3 subpixel refinement를 수행합니다.
5. source 좌표를 유지한 최종 sparse match를 원본 영상 좌표로 반환합니다.

학습에는 ALIKE의 **peaky, reprojection localization, score repeatability,
descriptor NRE** 네 항과 eLoFTR 계열의 **coarse focal, fine focal, local regression**
세 항을 함께 사용합니다. target detector는 보조 loss를 위해 학습 때만 켭니다.
모든 sampling, 좌표 변환과 resize geometry는 `align_corners=False` 규약을 따릅니다.
NMS/top-K index는 미분하지 않으며 선택된 peak의 좌표와 descriptor를 통해
matching loss가 detector에 연결됩니다.

원본 모델들의 loss 계열을 새 비대칭 구조에 맞게 적용했습니다. 특히 stride-8
descriptor NRE와 source-point-to-target-window fine loss는 원본 구현과 다릅니다.
상세 수식과 차이는 [loss 문서](docs/losses.md), backbone 선택과 좌표 정의는
[구조 문서](docs/architecture.md)에 기록했습니다.

## 실행

현재 작업공간에서 프로젝트 경로는 `/home/gleefe/desc2feat/desc2feat`입니다.
PyTorch 2.4 이상 환경을 권장합니다. 기본 config는 JSON이라 PyYAML 없이 실행됩니다.

```bash
cd /home/gleefe/desc2feat/desc2feat
python -m pip install -e '.[train,eval]'

# 합성 데이터로 2-step 훈련, validation, checkpoint 저장
OMP_NUM_THREADS=2 python -m desc2feat.train --smoke-test --output runs/smoke

# 실제 depth/camera geometry 기반 학습
python -m desc2feat.train --config configs/megadepth.json \
  --train-manifest /data/train.jsonl --val-manifest /data/val.jsonl \
  --eloftr-checkpoint /weights/eloftr-training.ckpt --output runs/megadepth

# 학습 재개
python -m desc2feat.train --resume runs/megadepth/last.pt --output runs/megadepth

# 이미지 pair 추론
python -m desc2feat.infer image0.jpg image1.jpg \
  --checkpoint runs/megadepth/last.pt --output matches.npz \
  --visualization matches.png --deploy

# pair geometry 평가 / GPU latency 측정
python -m desc2feat.evaluate /data/test.jsonl \
  --checkpoint runs/megadepth/last.pt --output evaluation.json
python -m desc2feat.benchmark --checkpoint runs/megadepth/last.pt \
  --image0 image0.jpg --image1 image1.jpg --resize 832 \
  --warmup 20 --iterations 100 --deploy --output latency.json

# 수치·gradient·padding·빈 입력 검증
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

현재 시스템 기본 `python`에는 torch가 없지만
`/home/gleefe/anaconda3/envs/lightglue/bin/python`으로 CPU 검증을 수행했습니다.
해당 interpreter는 Python 3.8/PyTorch 2.4.1이며 소스에서 직접 실행할 수 있습니다.
새 설치에는 `pyproject.toml`의 Python 3.9+ 요구사항을 따르세요.
CUDA driver가 동작하지 않아 이 환경에서 GPU latency와 AMP/DDP CUDA 실행은 검증하지 못했습니다.

데이터 manifest에는 image 경로와 `H_0to1`, 또는 depth·intrinsics·camera transform을
기록합니다. 경로, resize, pose convention, DDP, 평가 지표는
[데이터와 CLI 문서](docs/data.md)에 설명했습니다.
eLoFTR checkpoint에서는 호환되는 **training-form backbone만** 초기화합니다.
새 detector/transformer/fine head의 학습을 생략할 수 없습니다.

## 논문 포지션

Sparse-to-dense 검색과 query transformer는 S2DNet, COTR, ECO-TR 등과 겹칩니다.
따라서 ALIKE+eLoFTR 결합이나 `align_corners=False`만으로 ICCV/CVPR contribution을
주장하기는 어렵습니다. 제한 K에서의 공동 최적화가 실제 이득을 내는지,
target keypoint 검출이 실패하는 조건에서 localization이 개선되는지,
eLoFTR full/optimized 대비 정확도–지연시간 곡선을 개선하는지가 핵심입니다.

832²와 stride 8, K=1,000이면 correlation 원소는 약 1.17억에서 1,082만으로
10.8배 줄어듭니다. 이것이 전체 추론 속도 10.8배 향상을 뜻하지는 않습니다.
source에도 CNN과 keypoint 검출은 필요하므로 양쪽 모두 검출하기 어려운 장면을
일반적으로 해결한다고 주장할 수 없습니다.

선행 연구의 공식 출처, contribution 후보, 필수 ablation, localization 평가
설계는 [연구 포지셔닝](docs/research_positioning.md)에 정리했습니다.
이 구현은 가설을 검증하기 위한 첫 baseline입니다.

## 파일 안내

| 파일 | 역할 |
| --- | --- |
| `desc2feat/model.py` | backbone, detector, transformer, coarse/fine matcher |
| `desc2feat/geometry.py` | half-pixel sampling, homography/depth warp |
| `desc2feat/losses.py` | ALIKE 4항 + eLoFTR 계열 3항 |
| `desc2feat/data.py` | image/geometry manifest와 padding |
| `desc2feat/prepare_megadepth.py` | 기존 eLoFTR scene NPZ → pair manifest |
| `desc2feat/train.py` | AMP, accumulation, DDP, checkpoint/resume |
| `desc2feat/infer.py` | 이미지 pair → 원본 좌표 NPZ/시각화 |
| `desc2feat/evaluate.py` | homography/relative-pose 평가 |
| `desc2feat/benchmark.py` | synchronized latency와 peak memory |
| `configs/smoke.json` | CPU 실행 검증용 축소 모델 |
| `configs/megadepth.json` | 기본 1,000-query 모델 학습 시작 config |

원본 `EfficientLoFTR/`, `ALIKE_training/`은 수정하지 않았습니다.
복사한 RepVGG 코드의 출처와 제공된 license는 [THIRD_PARTY.md](THIRD_PARTY.md)에 있습니다.
