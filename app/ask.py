"""第1回: 素のLLMに質問し、応答時間とトークン数を比較する。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask --model gemini-3.5-flash-lite --model gemini-3.5-flash \\
        --question "LoRAとは何ですか？" --question "RAGとは何ですか？" --pause 4
    uv run python -m app.ask  # 質問を省略すると、講座に固有の質問をする

モデルと質問の全組み合わせを順番に実行する。429（レート制限）を受けたら、
残りの呼び出しは行わない。
"""

import argparse
import math
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def _non_negative_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("秒数は数値で指定してください") from e
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("秒数は有限の0以上の値を指定してください")
    return seconds


@dataclass
class Measurement:
    model: str
    question: str
    elapsed_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    response_text: str | None = None
    error: str | None = None
    status_code: int | None = None


def ask(question: str, model: str = MODEL, client: genai.Client | None = None) -> Measurement:
    """1回のAPI呼び出しを計測する。応答時間はクライアント側の経過時間。"""
    if client is None:
        client = genai.Client(api_key=get_api_key())

    started_at = time.perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=question)
    except errors.APIError as e:
        return Measurement(
            model=model,
            question=question,
            elapsed_seconds=time.perf_counter() - started_at,
            input_tokens=None,
            output_tokens=None,
            error=api_error_message(e.code, e.message),
            status_code=e.code,
        )

    elapsed_seconds = time.perf_counter() - started_at
    usage = response.usage_metadata
    return Measurement(
        model=model,
        question=question,
        elapsed_seconds=elapsed_seconds,
        input_tokens=usage.prompt_token_count if usage is not None else None,
        output_tokens=usage.candidates_token_count if usage is not None else None,
        response_text=response.text,
    )


def _token_total(values: list[int | None]) -> str:
    known_values = [value for value in values if value is not None]
    if not known_values:
        return "不明"
    total = sum(known_values)
    if len(known_values) != len(values):
        return f"{total}（一部不明）"
    return str(total)


def print_summary(results: list[Measurement], models: list[str]) -> None:
    print("\n--- 比較サマリー ---")
    print("model | 成功/試行 | 平均応答時間 | 入力token合計 | 出力token合計 | 429")
    for model in models:
        model_results = [result for result in results if result.model == model]
        successful = [result for result in model_results if result.error is None]
        average_seconds = (
            f"{sum(result.elapsed_seconds for result in successful) / len(successful):.2f}s"
            if successful
            else "—"
        )
        rate_limited = sum(result.status_code == 429 for result in model_results)
        print(
            f"{model} | {len(successful)}/{len(model_results)} | {average_seconds} | "
            f"{_token_total([result.input_tokens for result in successful])} | "
            f"{_token_total([result.output_tokens for result in successful])} | "
            f"{rate_limited}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLMの応答時間とトークン数を、複数の質問・モデルで比較する"
    )
    parser.add_argument("question", nargs="?", help="比較する質問（1つ）")
    parser.add_argument(
        "--question",
        dest="additional_questions",
        action="append",
        metavar="TEXT",
        help="比較する質問（複数指定可。省略時は既定の質問）",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        metavar="NAME",
        help=f"比較するモデル（複数指定可。既定: {MODEL}）",
    )
    parser.add_argument(
        "--pause",
        type=_non_negative_seconds,
        default=0,
        metavar="SECONDS",
        help="呼び出し間に入れる待ち時間（秒。無料枠の制限回避を保証するものではない）",
    )
    args = parser.parse_args()

    questions = ([args.question] if args.question else []) + (args.additional_questions or [])
    if not questions:
        questions = [DEFAULT_QUESTION]
    models = args.models or [MODEL]

    print("[計測方法] API呼び出し直前から応答受信までのクライアント側経過時間")
    print(f"[呼び出し予定] {len(questions)} 質問 × {len(models)} モデル = {len(questions) * len(models)} 回")

    client = genai.Client(api_key=get_api_key())
    results: list[Measurement] = []
    stop_for_rate_limit = False
    for question in questions:
        for model in models:
            if results and args.pause:
                time.sleep(args.pause)
            result = ask(question, model, client)
            results.append(result)
            print(f"\n[model] {result.model}")
            print(f"[質問] {result.question}")
            print(f"[応答時間] {result.elapsed_seconds:.2f} 秒")
            if result.error is not None:
                print(f"[APIエラー] {result.error}")
                if result.status_code == 429:
                    stop_for_rate_limit = True
                    break
                continue

            input_count = result.input_tokens if result.input_tokens is not None else "不明"
            output_count = result.output_tokens if result.output_tokens is not None else "不明"
            print(f"[トークン数] 入力: {input_count}, 出力: {output_count}")
            print(result.response_text or "[応答テキストなし]")
        if stop_for_rate_limit:
            break

    if stop_for_rate_limit:
        print("\n429を受けたため、レート制限への追加リクエストを避けて比較を中断しました。")
    print_summary(results, models)
    print(
        "\n[無料枠について] トークン数はこの実行の使用量であり、残りの利用枠ではありません。"
        "RPM・TPM・日次上限はモデルやアカウントの利用階層で異なります。"
        "公式の制限値と照らし合わせてください。"
    )


if __name__ == "__main__":
    main()
