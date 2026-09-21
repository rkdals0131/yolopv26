#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/home/kai/yolopv26"
AIHUB_ROOT="$REPO_ROOT/seg_dataset/AIHUB"
DATASET_ROOT="$AIHUB_ROOT/차선-횡단보도 인지 영상(수도권)"
STAGING_ROOT="$AIHUB_ROOT/_download_lane_197"
PACKAGE_ROOT="$STAGING_ROOT/053.차선-횡단보도_인지_영상(수도권)/01.데이터/1.Training"
AIHUBSHELL="$HOME/.local/bin/aihubshell"
APIKEY_FILE="$HOME/.config/aihub/apikey"
FILEKEYS="49501,49502,49503,49504,49505,49506,49507,49508,49807,49751,49695,49655,49648,49604,49532,49533"

exec 9>"$AIHUB_ROOT/.lane_download.lock"
if ! flock -n 9; then
    echo "다른 차선 데이터 다운로드가 실행 중입니다." >&2
    exit 1
fi

test -x "$AIHUBSHELL" || { echo "aihubshell이 없습니다: $AIHUBSHELL" >&2; exit 1; }
test -s "$APIKEY_FILE" || { echo "AIHub API key 파일이 없습니다: $APIKEY_FILE" >&2; exit 1; }

AIHUB_APIKEY="$(tr -d '\r\n' < "$APIKEY_FILE")"
test -n "$AIHUB_APIKEY" || { echo "AIHub API key가 비어 있습니다." >&2; exit 1; }
export AIHUB_APIKEY

mkdir -p "$STAGING_ROOT" "$DATASET_ROOT/Training" "$REPO_ROOT/runs"

echo "AIHub dataset 197의 1280x720 주간 train 2~8과 야간 train 1을 준비합니다."
echo "user1에 있는 주간 train 1과 주간 validation 1은 다운로드하지 않습니다."
df -h "$AIHUB_ROOT"

archives=(
    "$PACKAGE_ROOT/원천데이터_0610/1280_720/daylight/c_1280_720_daylight_train_2.tar"
    "$PACKAGE_ROOT/원천데이터_0610/1280_720/daylight/c_1280_720_daylight_train_3.tar"
    "$PACKAGE_ROOT/원천데이터_0610/1280_720/daylight/c_1280_720_daylight_train_4.tar"
    "$PACKAGE_ROOT/원천데이터_0610/1280_720/daylight/c_1280_720_daylight_train_5.tar"
    "$PACKAGE_ROOT/원천데이터_0610/1280_720/daylight/c_1280_720_daylight_train_6.tar"
    "$PACKAGE_ROOT/원천데이터_0610/1280_720/daylight/c_1280_720_daylight_train_7.tar"
    "$PACKAGE_ROOT/원천데이터_0610/1280_720/daylight/c_1280_720_daylight_train_8.tar"
    "$PACKAGE_ROOT/원천데이터_0610/1280_720/night/c_1280_720_night_train_1.tar"
    "$PACKAGE_ROOT/라벨링데이터/1280_720/daylight/c_1280_720_daylight_train_2.zip"
    "$PACKAGE_ROOT/라벨링데이터/1280_720/daylight/c_1280_720_daylight_train_3.zip"
    "$PACKAGE_ROOT/라벨링데이터/1280_720/daylight/c_1280_720_daylight_train_4.zip"
    "$PACKAGE_ROOT/라벨링데이터/1280_720/daylight/c_1280_720_daylight_train_5.zip"
    "$PACKAGE_ROOT/라벨링데이터/1280_720/daylight/c_1280_720_daylight_train_6.zip"
    "$PACKAGE_ROOT/라벨링데이터/1280_720/daylight/c_1280_720_daylight_train_7.zip"
    "$PACKAGE_ROOT/라벨링데이터/1280_720/daylight/c_1280_720_daylight_train_8.zip"
    "$PACKAGE_ROOT/라벨링데이터/1280_720/night/c_1280_720_night_train_1.tar"
)

need_download=false
for archive_file in "${archives[@]}"; do
    if [[ ! -s "$archive_file" ]]; then
        need_download=true
        break
    fi
done

if "$need_download"; then
    cd "$STAGING_ROOT"
    "$AIHUBSHELL" -mode d -datasetkey 197 -filekey "$FILEKEYS"
else
    echo "선택한 AIHub 압축 파일이 모두 있어 다운로드를 건너뜁니다."
fi

extract_archive() {
    local archive_file="$1"
    local output_dir="$2"
    local marker="$output_dir/.pv26_extracted"

    test -s "$archive_file" || { echo "압축 파일이 없습니다: $archive_file" >&2; exit 1; }
    if [[ -f "$marker" ]]; then
        echo "이미 압축 해제됨: $output_dir"
        return
    fi

    mkdir -p "$output_dir"
    case "$archive_file" in
        *.tar) tar -xf "$archive_file" -C "$output_dir" ;;
        *.zip) unzip -qo "$archive_file" -d "$output_dir" ;;
        *) echo "지원하지 않는 압축 파일: $archive_file" >&2; exit 1 ;;
    esac
    touch "$marker"
}

for group_number in 2 3 4 5 6 7 8; do
    group_name="c_1280_720_daylight_train_${group_number}"
    extract_archive \
        "$PACKAGE_ROOT/원천데이터_0610/1280_720/daylight/${group_name}.tar" \
        "$DATASET_ROOT/Training/[원천]${group_name}"
    extract_archive \
        "$PACKAGE_ROOT/라벨링데이터/1280_720/daylight/${group_name}.zip" \
        "$DATASET_ROOT/Training/[라벨]${group_name}"
done

night_group="c_1280_720_night_train_1"
extract_archive \
    "$PACKAGE_ROOT/원천데이터_0610/1280_720/night/${night_group}.tar" \
    "$DATASET_ROOT/Training/[원천]${night_group}"
extract_archive \
    "$PACKAGE_ROOT/라벨링데이터/1280_720/night/${night_group}.tar" \
    "$DATASET_ROOT/Training/[라벨]${night_group}"

label_count=$(find "$DATASET_ROOT/Training" -type f -name '*.json' | wc -l)
image_count=$(find "$DATASET_ROOT/Training" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)

echo "차선 Training labels: $label_count"
echo "차선 Training images: $image_count"
du -sh "$DATASET_ROOT" "$STAGING_ROOT"
df -h "$AIHUB_ROOT"
echo "다운로드 및 압축 해제가 완료되었습니다."
