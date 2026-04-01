# Raw Dataset Layouts

이 문서는 `yolopv26`가 실제로 참조하는 원본 데이터 구조를 정리한다. 중심은 AIHUB이며, BDD100K는 보조 섹션으로만 다룬다.

## 경로 전제

- repo에서는 항상 `seg_dataset/...`를 논리 경로로 사용한다.
- 실제 저장 위치는 심볼릭 링크일 수 있다. 현재 워크스페이스도 repo 밖 저장소를 가리키는 `seg_dataset` 심볼릭 링크 구성을 사용할 수 있다.
- 따라서 문서에서 말하는 `seg_dataset/AIHUB/...`는 “repo 안의 물리 디렉터리”가 아니라 “repo에서 접근하는 논리 dataset root”로 이해하면 된다.

## AIHUB 개요

현재 PV26 계열에서 직접 다루는 AIHUB 원본 source는 세 가지다.

- `차선-횡단보도 인지 영상(수도권)`
- `신호등-도로표지판 인지 영상(수도권)`
- `도로장애물·표면 인지 영상(수도권)`

### 읽은 문서

실제 구조 판단에 사용한 문서는 아래다.

- `seg_dataset/AIHUB/docs/차선_횡단보도_인지_영상(수도권)_데이터_구축_가이드라인.pdf`
- `seg_dataset/AIHUB/docs/수도권신호등표지판_인공지능 데이터 구축활용 가이드라인_통합수정_210607.pdf`
- `seg_dataset/AIHUB/도로장애물·표면 인지 영상(수도권)/061.도로장애물_표면_인지_영상(수도권)_데이터_구축_가이드라인.pdf`
- `seg_dataset/AIHUB/도로장애물·표면 인지 영상(수도권)/09. [도로상태 및 자율버스 과제]도로장애물,표면 인지 영상(수도권).pdf`

보조 문서:

- `seg_dataset/AIHUB/docs/01. [주행 환경 정적 객체 인지 과제] 차선,횡단보도 인지 영상(수도권).pdf`
- `seg_dataset/AIHUB/docs/055.신호등표지판영상(수도권지역).pdf`

주의:

- `055.신호등표지판영상(수도권지역).pdf`는 현재 로컬 파일 기준으로 실질 내용이 거의 없어서 구조 판단의 1차 근거로 쓰지 않는다.
- obstacle 문서는 `docs/`가 아니라 raw root 내부에 있다.

## AIHUB 공통 구조

lane/traffic 가이드라인에서 공통으로 확인되는 공식 구조는 아래다.

- 최상위 split은 `Training / Validation / Test`
- NIA 배포 관점 split 비율은 `8:1:1`
- 업로드 단위는 압축 묶음이며, 해상도/주야간/지역/묶음 번호를 이름에 포함한다
- 문서 예시:
  - `m_1920_1200_daylight_train_1.zip`
  - `c_1920_1200_night_validation_1.zip`
- 원천과 라벨이 분리된다
  - 원천데이터(Image)
  - 라벨링데이터(Json)

다만 실제 로컬 보유분은 전체 dataset이 아니라 일부 압축 묶음만 풀린 상태일 수 있다. 문서상의 “전체 구조”와 현재 디스크에 있는 “subset 구조”는 구분해서 봐야 한다.

## AIHUB Lane

### 문서 기준

- 과제: 차선/횡단보도 인지 영상 (수도권)
- 라벨 형태:
  - 차선: `polyline`
  - 정지선: `polyline`
  - 횡단보도: `polygon`
- 예시 이미지 메타:
  - `filename`
  - `imsize: [1280, 720]`
- 문서 설명상 영상 단독 데이터와 다중 센서 데이터가 모두 존재한다.

### 현재 로컬 subset

현재 로컬에서 확인된 상태:

- Training: 이미지 `30,000`, JSON `30,000`
- Validation: 이미지 `27,700`, JSON `27,700`
- Test: 없음
- archive:
  - Training `.tar:1`, `.zip:1`
  - Validation `.tar:2`

현재 로컬 디렉터리 예시:

```text
seg_dataset/AIHUB/차선-횡단보도 인지 영상(수도권)/
  Training/
    [원천]c_1280_720_daylight_train_1.tar
    [라벨]c_1280_720_daylight_train_1.zip
    [원천]c_1280_720_daylight_train_1/
      c_1280_720_daylight_train_1/
        16608323.jpg
        16608423.jpg
        ...
    [라벨]c_1280_720_daylight_train_1/
      16608323.json
      16608423.json
      ...
  Validation/
    [원천]c_1280_720_daylight_validation_1.tar
    [라벨]c_1280_720_daylight_validation_1.tar
    [원천]c_1280_720_daylight_validation_1/
      c_1280_720_daylight_val_1/
        10002688.jpg
        10002689.jpg
        ...
    [라벨]c_1280_720_daylight_validation_1/
      10002688.json
      10002689.json
      ...
```

핵심 포인트:

- 로컬에는 `c_1280_720_daylight_*_1` 묶음만 내려와 있다.
- 전체 dataset의 모든 지역/주야간/해상도 묶음이 있는 상태가 아니다.
- validation 원천 내부 하위 폴더명은 `...validation_1`인데, 그 안쪽 실제 폴더명은 `...val_1`로 줄어든 경우가 있다. 따라서 path는 문자열 규칙보다 실제 파일 트리를 기준으로 찾는 편이 안전하다.

## AIHUB Traffic

### 문서 기준

- 과제: 신호등/도로표지판 인지 영상 (수도권)
- 라벨 형태: `bounding box`
- 예시 이미지 메타:
  - `filename`
  - `imsize: [1280, 720]`
- 표지판 crop 분류셋이 탐지 데이터와 별도로 존재한다.
- 문서상 압축 묶음 naming은 지역/해상도/주야간/split/묶음 번호를 포함한다.

### 현재 로컬 subset

현재 로컬에서 확인된 상태:

- Training raw images: `30,000`
- Validation raw images: `30,000`
- Training crop images: `578,245`
- JSON: train `30,000`, val `30,000`
- Test: 없음

현재 로컬 디렉터리 예시:

```text
seg_dataset/AIHUB/신호등-도로표지판 인지 영상(수도권)/
  Training/
    [원천]c_train_1280_720_daylight_1.tar
    [라벨]c_train_1280_720_daylight_1.tar
    표지판코드분류crop데이터1.tar
    표지판코드분류crop데이터2.tar
    [원천]c_train_1280_720_daylight_1/
      s01000200.jpg
      s01000201.jpg
      ...
    [라벨]c_train_1280_720_daylight_1/
      c_train_1280_720_daylight_1/
        s01000200.json
        s01000201.json
        ...
    표지판코드분류crop데이터1/
      result_1/
        101/
          201018_0288_002.jpg
        104/
          200046_0027_001.jpg
          ...
  Validation/
    [원천]c_validation_1280_720_daylight_1.tar
    [라벨]c_validation_1280_720_daylight_1.tar
    [원천]c_validation_1280_720_daylight_1/
      s01000100.jpg
      s01000101.jpg
      ...
    [라벨]c_validation_1280_720_daylight_1/
      c_validation_1280_720_daylight_1/
        s01000100.json
        s01000101.json
        ...
```

핵심 포인트:

- detector 표준화는 원천 주행 이미지와 JSON만 사용한다.
- `표지판코드분류crop데이터*`는 sign classifier용 auxiliary asset이다.
- 로컬에는 `c_train_1280_720_daylight_1`, `c_validation_1280_720_daylight_1`만 있고 Test 및 다른 조건 묶음은 없다.

## AIHUB Obstacle

### 문서 기준

- 과제: 도로장애물/표면 인지 영상 (수도권)
- 문서상 원천 데이터 형식:
  - `*.mp4`
- 가공/이미지 단위 데이터:
  - `*.png`
  - `*.json`
- 이미지 해상도는 `1920x1080` 또는 `1280x720` 언급이 있다.

### 현재 로컬 subset

현재 로컬에서 확인된 상태:

- Training: 이미지 `1,972`, JSON `1,972`
- Validation: 이미지 `5,460`, JSON `5,460`
- Test: 없음

현재 로컬 디렉터리 예시:

```text
seg_dataset/AIHUB/도로장애물·표면 인지 영상(수도권)/
  Training/
    Images/
      TOA/
        1.Frontback_A01/
          V0F_HY_0002_20210120_141338_E_CH0_Seoul_Sun_Frontback_Day_94775.png
          V0F_HY_0003_20210121_131649_N_CH2_Seoul_Sun_Frontback_Day_05420.png
          ...
    Annotations/
      TOA/
        1.Frontback_A01/
          V0F_HY_0002_20210120_141338_E_CH0_Seoul_Sun_Frontback_Day_94775_BBOX.json
          V0F_HY_0003_20210121_131649_N_CH2_Seoul_Sun_Frontback_Day_05420_BBOX.json
          ...
  Validation/
    Images/
      TOA/
        1.Frontback_F01/
          V0F_HY_0033_20201228_151155_E_CH1_Seoul_Sun_Frontback_Day_79029.png
          ...
    Annotations/
      TOA/
        1.Frontback_F01/
          ...
```

핵심 포인트:

- obstacle은 lane/traffic처럼 `[원천]...tar`, `[라벨]...tar` 형태가 아니라 이미 `Images/Annotations` 트리로 정리된 subset이 로컬에 존재한다.
- 파일명 자체가 route/time/camera/channel/weather/view/frame 정보를 길게 encode한다.
- annotation 파일은 image stem 뒤에 `_BBOX`가 붙는다.

## BDD100K 요약

BDD100K는 이 프로젝트에서 중요도가 AIHUB보다 낮다. 그래도 경로 기준은 아래 두 개만 보면 된다.

```text
seg_dataset/BDD100K/
  bdd100k_images_100k/100k/<split>/*.jpg
  bdd100k_labels/100k/<split>/*.json
```

canonical sample naming은 `bdd100k_det_100k_<split>_<relative_id>` 형태의 safe slug를 사용한다.

## Standardized Output 구조

AIHUB와 BDD는 raw를 그대로 학습에 쓰지 않고 canonical output으로 바꾼다.

AIHUB:

```text
seg_dataset/pv26_od_bootstrap/canonical/aihub_standardized/
  images/<split>/<sample_id>.<ext>
  labels_scene/<split>/<sample_id>.json
  labels_det/<split>/<sample_id>.txt
  meta/...
```

BDD:

```text
seg_dataset/pv26_od_bootstrap/canonical/bdd100k_det_100k/
  images/<split>/<sample_id>.<ext>
  labels_scene/<split>/<sample_id>.json
  labels_det/<split>/<sample_id>.txt
  meta/...
```

여기서 `sample_id`는 raw 원본의 basename이 아니라, standardizer가 `dataset_key + split + relative_id`를 safe slug로 만든 값이다. 이 점 때문에 나중에 AIHUB 추가 압축 묶음을 더 받아도 stem 충돌 가능성이 낮다.

## OD Bootstrap 이후 ID 규약

`od_bootstrap`에서는 ID가 한 단계 더 늘어난다.

- `sample_id`
  - 각 canonical dataset 내부 stem
- `sample_uid`
  - build-exhaustive-od에서 쓰는 전역 키
  - 규약: `<dataset_key>__<split>__<sample_id>`
- `bootstrap_sample_uid`
  - exhaustive OD materialization scene JSON에 기록되는 UID
- `final_sample_id`
  - build-final-dataset 이후 최종 merged dataset에서 쓰는 stem
  - 현재 계약은 `scene_path.stem`

현재 중요한 점:

- exhaustive OD는 이미 `sample_uid` 기반 파일명을 사용하므로 source dataset 간 충돌을 피한다.
- lane canonical도 standardizer가 만든 `sample_id` 자체가 충분히 긴 safe slug라서, 추가 AIHUB 묶음이 더 들어와도 그대로 확장 가능하다.
- final merged dataset은 raw `image.file_name`이 아니라 `final_sample_id`를 기준으로 파일명을 다시 정한다.
