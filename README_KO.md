# DLC 2026 최종 추론

한국어 | [English](README.md)

이 저장소에는 수정하지 않은 BF16 `Qwen/Qwen2.5-3B-Instruct` 베이스 모델로
DLC 2026 최종 답안을 생성한 추론 코드가 들어 있습니다. 각 문제를 16번 풀고,
출력 길이 상한에 도달했으면서 끝부분에 답 태그가 없는 풀이만 4,096토큰으로
다시 생성합니다. 같은 조건이 이어지면 8,192토큰으로 한 번 더 생성합니다.
최종 답은 다수결로 정하며, 동률이면 먼저 생성된 답을 선택합니다.

## 실행 조건

- Apple Silicon Mac이 필요합니다.
- [MLX 설치 조건](https://ml-explore.github.io/mlx/build/html/install.html)에 따라 macOS 14 이상이 필요합니다.
- 실행 환경, 모델, 결과를 저장할 디스크 여유 공간을 10GB 이상 권장합니다.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)가 필요합니다.

기준 답안은 macOS 26.5.1, M5 Pro, 통합 메모리 64GB, Python 3.12.13,
`uv 0.11.28`, `mlx-lm==0.31.3` 환경에서 생성했습니다. 같은 답안을 재현하려면
배치 크기 128을 유지해야 합니다. 더 적은 메모리를 탑재한 Mac에서는 전체
실행을 확인하지 않았으며, 배치 크기를 낮추면 생성 결과가 달라집니다.

`uv`가 없다면 다음 명령으로 설치합니다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 준비

```bash
git clone https://github.com/pysunn14/deep-learning-challenge-2026-final.git
cd deep-learning-challenge-2026-final
uv sync --frozen

mkdir -p data
cp /path/to/test_submission.csv data/test_submission.csv

MODEL_DIR="$(uv run hf download Qwen/Qwen2.5-3B-Instruct \
  --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --quiet)"
```

모델 다운로드 크기는 약 6.2GB입니다. 이 명령은 공식
[`hf download` 사용법](https://huggingface.co/docs/huggingface_hub/en/guides/cli)에
따라 고정 개정판을 내려받고, `--model`에 전달할 정확한 스냅샷 폴더를
`MODEL_DIR`에 저장합니다.
새 터미널을 열었다면 `MODEL_DIR=...` 명령을 다시 실행합니다. 이미 내려받은
모델은 캐시에서 찾으므로 다시 다운로드하지 않습니다.

## 입력 확인

```bash
uv run dlc-final validate
```

`validate`는 입력 파일을 읽고 행 수를 표시합니다. CSV에는 `id`, `question` 열이
필요합니다. `run` 명령은 모델을 불러올 때 개정판, 모델 파일 SHA-256, BF16
정밀도와 양자화 여부를 확인합니다.

## 추론 실행

```bash
uv run dlc-final run \
  --model "$MODEL_DIR" \
  --execute
```

다른 터미널에서는 다음 명령으로 진행 상태를 JSON 형식으로 확인할 수 있습니다.

```bash
uv run dlc-final status
```

일시 중단하려면 `Ctrl-C`를 누릅니다. 입력 파일, 모델 스냅샷, 설정, 결과 폴더를
바꾸지 않고 같은 `run` 명령을 다시 실행하면 마지막으로 완전히 저장된 배치
다음부터 이어서 처리합니다.

재시도 순서는 `1,024 -> 4,096 -> 8,192`로 고정되어 있습니다. 앞 단계 출력이
실제로 토큰 상한에 도달했고 열린 생각 블록 밖의 끝 답 태그가 없을 때만 다음
단계를 실행합니다. 완료된 짧은 단계는 저장된 실행 계약과 SHA-256을 검사한 뒤
모델 생성 없이 그대로 재사용합니다.

## 결과 검사와 제출

```bash
uv run dlc-final verify
```

대회에 제출할 파일은 다음 파일 하나입니다.

```text
runs/final-base-n16/submission.csv
```

공식 제출 파일은 입력 순서를 유지한 `id,question,answer` 형식이며 answer 열에는
정수만 포함합니다. `answers.csv`는 `id,answer` 열만 담은 대조용 사본이며 공식
제출 파일이 아닙니다. `verify`는 두 파일의 기록된 SHA-256과 모든 `id,answer`
쌍이 서로 같은지까지 검사합니다.

## 재현성 안내

최종 설정은 온도 `0.6`, `top_p=0.95`, 배치 크기 `128`, 16개 고정 시드를
사용합니다. 이 구현은 기본 시드와 각 배치에 포함된 문제 ID를 함께 해시하여
배치별 난수 시드를 정합니다.

배치 크기나 문제 순서를 바꾸면 배치별 난수 시드가 바뀌며, 생성된 풀이와 최종
다수결 답도 달라질 수 있습니다. 기준 답안을 재현할 때는 커밋된
`config/final.json`을 그대로 사용합니다.

완료된 기준 실행의 전체 소요 시간은 9시간 2분 22초이며, 실제 Metal 활성
메모리 최고치는 17.87GiB입니다. 이어서 실행한 8,192토큰 단계는 210개 생성을
1시간 5분 3초에 처리했습니다. 이 중 상한에 계속 닿은 42개 생성은 기록한 뒤
투표에서 제외했습니다.
