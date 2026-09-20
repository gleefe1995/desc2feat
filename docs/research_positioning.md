# desc2feat: 연구 포지셔닝과 검증 계획

검토일: 2026-09-20. 아래의 성능 향상과 contribution은 **검증할 연구 가설**이다. 이 문서는 모델 학습 결과, 벤치마크 우위, 학회 채택 가능성을 입증한 결과물이 아니다. 방법 설명은 연구 방향을 제안하며, 현재 구현 여부는 README와 실제 코드를 기준으로 확인한다. 관련 연구는 공식 논문, 저자 프로젝트 페이지와 공식 저장소를 확인했다. 이 조사는 관련 문헌의 완전한 목록이나 신규성 보증은 아니다.

## 먼저 판단할 점

제안의 타당한 출발점은 **두 영상에서 동일한 3D 지점을 모두 검출해야 하는 제약을 한쪽에서 없앤다**는 것이다. image0에서 선택한 K개 지점의 descriptor를 image1의 모든 coarse 위치와 비교하고, image1의 연속 좌표를 세밀하게 복원한다. image0의 keypoint identity를 유지할 수 있어, 이미 만들어 둔 SfM map의 3D track을 image1로 옮기는 localization에서도 자연스럽다.

다만 sparse-to-dense 매칭 자체는 이미 확립된 분야다. ALIKE와 eLoFTR를 연결했다는 설명만으로 ICCV/CVPR 수준의 독립적인 신규성을 주장하기는 어렵다. 가장 설득력 있는 연구 질문은 다음과 같다.

> **고정된 source query 예산에서 검출 위치, descriptor, coarse-to-fine matcher를 함께 최적화하면, target 검출 실패에 대한 내성을 유지하면서 eLoFTR보다 나은 정확도–지연시간 곡선을 얻을 수 있는가?**

여기에서 새로움은 개별 부품의 이름보다, 제한된 query를 어떻게 선택하고 학습시키며 실제 downstream 이득으로 연결했는지에 달려 있다. 공동 학습 그 자체도 최초라고 할 수 없다. DISK는 올바른 매칭 수를 목표로 local feature를 end-to-end 학습했고, SiLK도 검출과 기술자의 학습 설계를 다룬다. [DISK 논문](https://arxiv.org/abs/2006.13566), [SiLK 논문](https://openaccess.thecvf.com/content/ICCV2023/html/Gleize_SiLK_Simple_Learned_Keypoints_ICCV_2023_paper.html).

## 가장 가까운 선행 연구

| 연구 | 이미 제시한 핵심 | desc2feat에서 실제로 구별해야 할 점 |
| --- | --- | --- |
| S2DHM, 3DV 2019 | reference의 sparse keypoint와 query의 dense hypercolumn을 매칭하는 visual localization | 단방향 검출과 map descriptor 재사용 자체는 신규성이 아니다. [공식 코드](https://github.com/germain-hug/S2DHM). |
| S2DNet, ECCV 2020 | sparse-to-dense correspondence map을 분류 문제로 학습하여 정밀한 대응점을 추정 | descriptor-to-dense 위치 분류 자체는 신규성이 아니다. 공동 학습하는 query selector, attention 구조, 속도 및 조건별 정확도 차이를 입증해야 한다. [저자 페이지](https://www.hugogermain.com/s2dnet), [논문](https://arxiv.org/abs/2004.01673). |
| COTR, ICCV 2021 | 한 영상의 임의 query point를 입력해 Transformer로 다른 영상의 대응점 추정; 반복 확대 기반 정밀화 | sparse query와 Transformer의 조합 자체는 신규성이 아니다. 좌표 query 회귀와 descriptor-conditioned correlation 사이의 차이, 비용을 설명해야 한다. [논문](https://openaccess.thecvf.com/content/ICCV2021/html/Jiang_COTR_Correspondence_Transformer_for_Matching_Across_Images_ICCV_2021_paper.html). |
| ECO-TR, ECCV 2022 | 공유 multi-scale feature와 단계별 Transformer로 arbitrary query의 coarse-to-fine 대응점을 한 번의 forward에서 추정 | 효율적인 query 기반 coarse-to-fine도 기존 주제다. 같은 query/정확도 예산에서 비교하고 query-clustering과의 관계를 설명해야 한다. [저자 페이지](https://dltan7.github.io/ecotr/), [논문](https://arxiv.org/abs/2209.12213). |
| ALIKE / ALIKED | 미분 가능한 subpixel 검출과 descriptor 학습; ALIKED는 deformable descriptor extraction을 확장 | detector와 descriptor의 출처를 명확히 인용하고 수정한 sampling 및 loss를 구분한다. ALIKED용 LightGlue 가중치를 ALIKE에 그대로 쓸 수 있다고 가정해서는 안 된다. [ALIKE 논문](https://arxiv.org/abs/2112.02906), [ALIKE 코드](https://github.com/Shiaoming/ALIKE), [ALIKED 코드](https://github.com/Shiaoming/ALIKED). |
| Efficient LoFTR, CVPR 2024 | aggregated attention 및 정밀한 두 단계 correlation을 이용하는 효율적인 semi-dense matcher | source 축을 K개 query로 바꾸는 것이 주된 구조 변화다. upstream의 full/opt 및 skip-softmax 구성을 모두 비교해야 한다. [논문](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Efficient_LoFTR_Semi-Dense_Local_Feature_Matching_with_Sparse-Like_Speed_CVPR_2024_paper.html), [공식 코드](https://github.com/zju3dv/EfficientLoFTR). |
| LightGlue, ICCV 2023 | 양쪽 sparse feature를 학습 매칭하고 쉬운 쌍에서 계산을 줄임 | detector 포함 전체 지연시간, adaptive stopping/pruning, 입력 K를 통제한다. frozen ALIKE의 nearest-neighbor만 비교하면 충분하지 않다. [논문](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html), [공식 코드](https://github.com/cvg/LightGlue). |
| OnePose / OnePose++ | sparse SfM 기반 2D–3D 매칭; OnePose++는 query keypoint 검출 없이 point cloud와 영상 feature를 직접 매칭 | OnePose++의 sparse-to-dense 2D–3D attention 및 저텍스처 동기는 매우 가깝다. desc2feat의 범용 2D–2D 학습 및 고정된 source identity와 구별한다. [OnePose](https://arxiv.org/abs/2205.12257), [OnePose++ 공식 페이지](https://zju3dv.github.io/onepose_plus_plus/). |
| Sparse-NCNet, ECCV 2020 | correlation tensor를 희소화한 뒤 sparse 4D convolution으로 처리하고 정밀화 | 이름의 sparse가 source detector query를 뜻하지 않는다. 탐색 공간 희소화의 선행으로 다룬다. [논문](https://arxiv.org/abs/2004.10566), [공식 코드](https://github.com/ignacio-rocco/sparse-ncnet). |
| ASpanFormer / TopicFM | adaptive attention span / topic에 따른 context와 matching | context 계산을 줄이면서 견고성을 얻는 다른 축이다. attention pooling을 새 기법으로 주장한다면 비교가 필요하다. [ASpanFormer 논문](https://arxiv.org/abs/2208.14201), [TopicFM 논문](https://arxiv.org/abs/2207.00328). |
| XFeat, CVPR 2024 | 가벼운 backbone과 sparse/semi-dense matching 및 refinement | 작은 backbone과 빠른 refinement만으로 차별화하기 어렵다. 실제 배포 속도 비교에 포함한다. [논문](https://openaccess.thecvf.com/content/CVPR2024/html/Potje_XFeat_Accelerated_Features_for_Lightweight_Image_Matching_CVPR_2024_paper.html). |

2026년 투고를 준비한다면 2024년 모델만을 SOTA로 취급하면 안 된다. 예를 들어 LoMa는 데이터/모델 규모를 확장한 local feature matching과 HardMatch 평가를 제안하며, AsymLoc은 offline teacher와 online student의 비대칭 localization 및 detector/descriptor distillation을 제안한다. 두 문헌은 확인한 arXiv 버전을 기준으로 하며, 여기서 학회 게재 여부를 추정하지 않는다. 연구 시작 시와 투고 직전에 관련 baseline을 다시 고정해야 한다. [LoMa, 2026-07 v2](https://arxiv.org/abs/2604.04931v2), [AsymLoc, 2026-07 v2](https://arxiv.org/abs/2604.09445v2).

## 주장 가능한 contribution 후보

다음은 결과가 뒷받침할 때 사용할 수 있는 논문 구성안이다. 아직 입증된 contribution 목록은 아니다.

1. **제한된 source query의 공동 최적화.** 미분 가능한 subpixel query와 descriptor, 비대칭 coarse-to-fine matching을 함께 학습한다. frozen detector 또는 detached query 좌표와 비교해, 같은 K에서 정확한 매칭 수와 공간 분포가 좋아지는 원인을 보인다. query 점수는 detector 반복성뿐 아니라 실제 matching 성공과 연결되는지 확인한다.
2. **source 지점을 보존하는 효율적 correspondence 검색.** source descriptor를 중심으로 target 전체 coarse map을 검색하고 연속 좌표를 찾는다. source의 기존 3D track identity를 유지하면서 dense–dense correlation을 피한다. 이 계산 구조가 어느 해상도와 K에서 효과적인지 분석하고, eLoFTR full/opt와 동등 정확도에서 실측 이득을 제시한다.
3. **target 검출 제약의 비용을 측정하는 평가.** target에서 대응 keypoint가 검출되지 않은 경우를 GT geometry로 구분한다. 이 부분집합의 recall 증가가 전체 pose/localization 성공률로 이어지는지 보여 준다. 양방향 저텍스처 전체를 해결한다는 포괄적 주장보다 검증 가능하다.

세 항목이 단순 조합의 실험 보고를 넘어서려면, 예를 들어 **query budget에 맞춘 학습 목적 또는 선택 방법**처럼 다른 matcher에도 적용 가능한 구체적인 방법적 기여가 필요할 수 있다. 이것도 그 자체로 최초라고 가정하지 말고 가까운 learned detector/query-selection 문헌과 별도로 비교해야 한다.

## 계산량 주장: softmax와 전체 추론을 분리하기

832 × 832 영상에서 coarse stride가 8이면 N = 104 × 104 = 10,816이다. batch 1에서 다음과 같다.

| 명시적으로 만드는 similarity matrix | 원소 수 | FP32 한 개 행렬 | FP16 한 개 행렬 |
| --- | ---: | ---: | ---: |
| N × N | 116,985,856 | 약 446.3 MiB | 약 223.1 MiB |
| K × N, K = 1,000 | 10,816,000 | 약 41.3 MiB | 약 20.6 MiB |

이는 직접 계산한 값이다. 원소 수는 10.816배 줄지만 **네트워크 전체가 10.816배 빨라진다는 뜻은 아니다**. backbone, dense target context, fine refinement, detector, gather/scatter, GPU kernel launch, normalization 등의 비용은 별도다. dual-softmax 중간값, gradient와 optimizer 상태를 더하면 실제 peak memory도 위 한 행렬 크기와 다르다.

공통 backbone을 써도 보통 두 영상의 dense feature 추출은 여전히 필요하다. source descriptor가 K개라는 사실만으로 source CNN 비용이 K에 비례하지 않는다. 또한 target에서 N × N self-attention을 그대로 실행하면 coarse matching만 줄여도 큰 병목이 남는다. 가능한 기본 설계는 pooled 또는 local target context, K × K source self-attention, K와 pooled target 간 cross-attention, 마지막 K × N correlation이다. target context를 더 강하게 줄였을 때의 정확도 변화는 ablation으로 검증해야 한다.

upstream eLoFTR의 `src/loftr/utils/coarse_matching.py`에는 `skip_softmax`가 이미 있다. 따라서 핵심을 “softmax를 없앴다”로 설명하기보다 **correlation의 source 축을 줄였다**고 설명해야 한다. softmax를 생략한 eLoFTR와도 비교해야 한다. [공식 coarse matching 코드](https://github.com/zju3dv/EfficientLoFTR/blob/main/src/loftr/utils/coarse_matching.py).

## backbone과 학습에 대한 제안

첫 모델은 eLoFTR의 RepVGG 계열 backbone을 유지하는 편이 비교를 해석하기 쉽다. 공유 trunk에서 stride 2/4/8 feature를 만들고, 경량 detector/descriptor head와 fine feature fusion을 붙인다. 첫 실험부터 ALIKE 전체 backbone와 eLoFTR 전체 backbone을 별도로 두면 추론 이득을 잃고 어느 요소가 성능을 냈는지 구분하기 어렵다. backbone 공유는 설계 제안이며 성능 보장은 아니다.

정밀한 source 좌표를 원하면 저해상도 coarse feature만 upsample한 점수보다 early feature를 포함한 detector head를 평가한다. full-resolution 고차원 descriptor map의 메모리가 크면, descriptor를 낮은 해상도에 유지하고 source subpixel에서 sampling하는 구성을 비교한다. 이때 target refinement에도 대응하는 해상도의 feature를 사용해야 한다.

학습 loss는 “두 논문의 이름을 붙인 합”보다 실제 수식과 gradient 경로를 명시해야 한다. 로컬 ALIKE training 코드의 기본 항목은 peaky/dispersity, reprojection localization, score-map repeatability/reliability, descriptor reprojection이며 triplet은 별도 선택 항목이다. 로컬 eLoFTR는 coarse focal, fine correlation focal, local subpixel regression을 둔다. 새로운 source point-to-target-window refinement가 upstream의 patch-to-patch fine matching과 다르면 fine loss도 **adapted loss**라고 써야 하며 원본과 동등하다고 부르면 안 된다.

공동 학습에서 확인할 사항:

- hard NMS/top-K의 **선택 index**는 일반적으로 미분되지 않는다. 선택된 peak 주변 soft-argmax, descriptor sampling, 해당 좌표에서의 fine feature sampling을 통한 경로와 구분한다. 따라서 “완전히 미분 가능한 top-K 검출”이라는 표현은 피한다.
- matcher loss만 backward했을 때 detector score-head에 유한하고 0이 아닌 gradient가 도달하는지 확인한다. 총 loss에서 detector 보조 loss가 존재한다는 사실만으로 joint optimization이 증명되지는 않는다.
- GT correspondence 생성에 모델 좌표를 사용할 때 target assignment를 detach하는지 명시한다. GT 좌표 자체를 움직여 loss를 줄이는 퇴행적인 경로가 생기지 않게 한다.
- source local patch와 target local patch의 정렬이 잘못되면 subpixel supervision이 의미를 잃는다. train의 GT crop과 inference의 predicted crop을 분리하고, GT crop만 사용했을 때의 노출 편향을 측정한다.
- 완전히 overlap이 없거나 깊이가 무효인 경우를 unmatched supervision과 구분한다. 관측 불가능한 점을 정답 음성으로 오염시키지 않는다.
- ALIKE의 반복성 loss는 양쪽 검출을 유도하지만 추론의 target 검출은 불필요하다. 이 보조 loss가 편향을 주는지 제거/감쇠 실험을 한다. 학습에서만 image1 detector를 켜는 비용과 추론 비용도 분리한다.

`align_corners=False`는 contribution보다는 좌표 일관성 조건이다. 너비 W의 feature grid에서 중심 좌표 x를 정규화할 때 `u = 2 * (x + 0.5) / W - 1`을 쓴다. 역변환은 `x = (u + 1) * W / 2 - 0.5`이다. resize의 half-pixel 좌표 규약과 strided convolution의 sampling origin은 별개이므로, stride feature를 어느 영상 좌표에 대응시킬지 명시하고 모든 loss와 crop에 동일하게 적용해야 한다. identity, 평행이동, 비정방형 영상, padding/crop, resize한 intrinsics에 대한 좌표 검증이 필요하다.

## 결과가 뒤집힐 수 있는 가설과 필수 ablation

| 가설 | 비교 | 반증 조건 또는 주의점 |
| --- | --- | --- |
| target detector 제약을 없애면 어려운 target에서 recall이 오른다 | 같은 source keypoint에서 sparse target matching 대 dense target matching | target detector 성공/실패별 recall 차이가 없거나 pose 성능으로 연결되지 않으면 핵심 동기가 약해진다. |
| query 공동 학습이 K 사용 효율을 높인다 | frozen detector / joint descriptor만 / detached coordinates / 전체 joint | 동일 K·backbone·학습량에서 이득이 없으면 공동 검출 학습을 contribution으로 삼기 어렵다. |
| global target context가 필요하다 | context 제거 / pooled context / dense context | 약한 context로도 동일하면 더 복잡한 Transformer를 정당화하기 어렵다. |
| subpixel source 및 target 정밀화가 중요하다 | integer source / subpixel source, fine 제거 / 1단계 / 2단계 | coarse pose만 좋아지고 정확한 pixel error가 나아지지 않으면 정밀도 주장을 제한한다. |
| 제한된 K로 eLoFTR 정확도를 유지한다 | K = 256, 512, 1,000, 2,000, 4,000 | 큰 K에서만 유지되면 속도상의 장점이 사라지는 지점을 보고해야 한다. |
| learned selector가 필요하다 | 동일 K의 regular grid / random / ALIKE / task-trained selection | regular grid가 동등하면 ALIKE detector가 속도·정확도에 어떤 역할인지 재검토한다. |
| loss를 함께 쓰는 것이 유효하다 | 각 ALIKE 항 제거, coarse/fine/local 항 제거, 단계별→joint | loss 총합만 보고하면 효과와 gradient 충돌을 설명할 수 없다. |
| source 보존이 localization에 유리하다 | source 고정 대 양쪽 refinement, 같은 3D map/검색 후보 | source를 이동시키면 기존 3D point identity가 깨질 수 있다. map도 바뀌면 별도의 비교가 필요하다. |

추가로 row-softmax, dual-softmax, explicit unmatched probability/quality를 비교할 가치가 있다. K × N의 column 경쟁은 dense–dense에서의 경쟁과 다르며, 여러 유효 source point가 같은 target coarse cell에 들어가면 mutual-nearest 제약이 좋은 match를 없앨 수 있다. 정밀화 전 coarse cell 단위 중복 제거와 정밀화 후 pixel 단위 중복 제거를 구분한다.

## 평가 프로토콜

초기 구현 검증과 논문 성능 검증을 구분한다. synthetic homography에서 loss가 줄고 backward가 된다는 사실은 geometry 코드의 일부를 검증할 뿐, real multi-view 성능이나 localization 정확도를 검증하지 않는다.

논문 실험은 최소한 다음을 포함하는 것이 좋다.

- **Relative pose:** MegaDepth와 ScanNet의 공개 pair/split을 고정하고 rotation/translation angular error의 max에 대한 AUC@5/10/20을 보고한다. RANSAC solver, threshold, 반복 수, resize, intrinsics 변환과 seed를 통일한다. 먼저 official checkpoint의 baseline 재현으로 평가기를 검증한다.
- **Pixel precision / homography:** HPatches illumination/viewpoint split별 precision, recall, matching accuracy 및 homography 정확도. 검출 수와 최종 match 수를 통제한 결과를 함께 낸다.
- **Visual localization:** Aachen Day-Night 및 가능하면 InLoc. retrieval shortlist, database images, 3D map, PnP와 verification을 통제한다. 고정 map에서 map observation 좌표를 source query로 사용하는 실험과, desc2feat detector로 map을 다시 만드는 실험을 분리한다. 전자는 matcher를, 후자는 전체 파이프라인을 평가한다.
- **비대칭 조건:** image0만 열화, image1만 열화, 둘 다 열화를 분리한다. blur, 노출 변화, viewpoint와 texture 통계를 사용하고 GT geometry에서 target detector coverage를 계산한다. 실제 day/night 데이터와 합성 열화를 함께 보고 합성 결과만으로 일반화하지 않는다.
- **최신/강한 baseline:** eLoFTR full/opt, SuperPoint+LightGlue, ALIKED+LightGlue, XFeat 및 자원이 허용하는 강한 dense matcher. S2DNet/ECO-TR는 적어도 방법 차이와 조건을 명확히 비교한다. ALIKE+LightGlue를 넣으려면 ALIKE descriptor용으로 훈련한 matcher가 필요하다. 다른 extractor용 가중치를 그대로 쓴 결과를 공정한 baseline으로 쓰지 않는다. LightGlue 공식 저장소는 자체 feature용 학습에 Glue Factory를 안내한다. [LightGlue training 안내](https://github.com/cvg/LightGlue#training-and-evaluation).

속도는 GPU 모델, PyTorch/CUDA, precision, image shape, batch size, K, 최대 match 수, warmup 횟수, sample 수, compile/FlashAttention 여부를 기록한다. CUDA synchronization/event를 사용하고 median과 p90/p95 latency, throughput, peak allocated memory를 함께 낸다. 모델 forward 전체와 backbone/detection/context/coarse/fine의 시간을 분리하되, 개별 구간 timer가 추가한 synchronization 비용을 전체 결과에 혼동하지 않는다. feature extraction을 한 모델에만 제외하지 않는다.

Localization에서는 두 운영 모드를 나란히 보고해야 한다: **새로운 image pair를 모두 처리하는 end-to-end latency**, **database source features가 사전 계산되어 있는 query latency**. target feature가 pair-dependent cross-attention 이전에만 cache 가능한지 확인한다. 여러 database 이미지에 대해 같은 query를 반복할 때 backbone과 query-independent context를 재사용할 수 있지만, pair-conditioned feature까지 재사용 가능하다고 가정하면 정확성이 깨진다.

“최소 정확도 유지”는 실험 전에 허용 오차를 정해야 한다. 예를 들어 AUC@5 손실 0.5 percentage point 이하 및 localization 주요 threshold의 하락 없음 같은 기준을 **연구팀이 선택**할 수 있다. 이는 보편적 학계 기준이 아니라 예시다. seed 반복과 scene/pair 단위 bootstrap confidence interval을 함께 보고, 평균의 작은 차이를 과장하지 않는다.

## ICCV/CVPR 가능성에 대한 현실적인 판단

**현재 아이디어 설명만으로는 채택 가능성을 높게 평가하기 어렵지만, 검증할 가치가 있는 연구 방향이다.** S2DNet/COTR/ECO-TR/OnePose++와의 관계를 인정한 뒤, 제한된 source query를 선택·학습하는 구체적 방법과 반복 가능한 결과가 필요하다. ALIKE+eLoFTR라는 조합명, `align_corners=False`, K × N 행렬 크기 계산은 그 자체로 충분한 contribution이 아니다.

LightGlue와 eLoFTR의 중간에 있는 모델도 지연시간–정확도 곡선의 새로운 Pareto 지점을 만들면 유용할 수 있다. 그러나 eLoFTR의 해상도/레이어/정밀도를 줄인 baseline, LightGlue의 K를 늘린 baseline과 비교했을 때도 이점이 있는지 봐야 한다. 더 작은 eLoFTR가 같은 속도에 더 정확하면 현재 포지션은 약하다. 반대로 전체 pose 정확도는 비슷해도, target 검출 실패가 잦은 조건에서 명확한 localization 개선과 map feature 재사용 비용 절감이 반복되면 더 명료한 논문이 될 수 있다.

첫 의사결정은 큰 훈련을 오래 돌리기 전에 내릴 수 있다: baseline 평가 재현 → 같은 backbone의 sparse-to-dense 모델 → K sweep → frozen/joint/regular-grid ablation → target-only 열화 및 실제 night query 평가. 이 결과에서 개선이 나오지 않으면 모듈을 계속 추가하기보다 query 선택 또는 활용할 배포 조건에 관한 가설을 수정하는 것이 좋다.
