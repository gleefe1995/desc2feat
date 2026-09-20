# 구현 검증 기록

검증일: 2026-09-20. 실제 데이터셋 학습 결과나 방법 간 성능 비교가 아닌
코드 실행·수치·gradient 검증이다.

환경: `/home/gleefe/anaconda3/envs/lightglue/bin/python`, Python 3.8.20,
PyTorch 2.4.1+cu121, NumPy 1.24.4, Pillow 10.4.0. CPU 2 threads로 실행했다.
CUDA는 사용할 수 없었다. 새 설치의 Python 요구사항은 `pyproject.toml`을 따른다.

## 확인한 결과

- `python -m unittest discover -s tests -v`: **38개 통과**.
  half-pixel 좌표와 depth visibility, OOV dustbin, NRE probability 보간,
  7개 loss, 빈 match/padding, source detector로의 matcher gradient,
  streamed/full coarse 일치, RepVGG fusion, GT 없는 추론을 포함한다.
- 기본 14,951,876-parameter 모델에서 128² image pair의 전체 loss forward/backward가
  성공했으며 gradient가 모두 finite였다. 7개 항이 모두 0보다 큰 예제를 확인했다.
- 기본 모델의 832² inference에서 coarse shape는 `[1,1000,10816]`,
  source valid query 1,000개, target valid cell 10,816개였다.
  confidence가 finite이고 target detector 출력은 `None`이었다.
- `--smoke-test` 축소 모델에서 두 번 optimizer update, validation, checkpoint
  저장이 성공했다. 최종 확인 로그의 training loss는 26.6634와 29.0284,
  validation loss는 27.2496이었다. 서로 다른 두 pair 값이며 학습 수렴을 뜻하지 않는다.
  첫 batch의 local loss는 crop support 밖이라 0, 다음 batch는 1.1514였다.
- CPU 4-step 연속 학습과 2-step 저장 후 2-step 재개에서 model tensor와
  Torch RNG state가 bitwise 동일했다. 같은 config, sampler와 dataset을 사용했다.
- 저장한 smoke checkpoint로 `infer --deploy`의 NPZ/PNG 생성,
  `evaluate`의 zero-match 실패 집계, `benchmark` 및 `--profile` 실행을 확인했다.
- MegaDepth converter는 mock scene NPZ에서 overlap filtering, 양방향 pair,
  `T1 @ inverse(T0)`와 역방향 transform을 수치적으로 확인했다.

## 재현 명령

프로젝트 디렉터리에서 실행한다. `PY`는 위의 검증용 interpreter 경로로 바꿀 수 있다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m desc2feat.train \
  --smoke-test --output runs/smoke
```

## 아직 검증하지 않은 부분

MegaDepth/ScanNet 실제 training, HPatches/Aachen/InLoc benchmark,
eLoFTR/LightGlue 대비 정확도·속도, CUDA AMP, CUDA DDP,
실제 pretrained eLoFTR checkpoint의 warm start는 검증하지 않았다.
backbone initializer는 호환되는 이름을 만든 mock checkpoint로 검사했다.
합성 smoke checkpoint는 실제 매칭에 사용할 학습된 모델이 아니다.

배포 성능은 training 완료된 checkpoint와 충분한 match가 발생하는 실제 pair로
측정해야 한다. Random/smoke 모델의 match 수가 적으면 fine 비용이 과소 측정된다.
