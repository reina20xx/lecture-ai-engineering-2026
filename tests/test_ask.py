from types import SimpleNamespace

from app.ask import Measurement, ask, print_summary


class FakeClient:
    def __init__(self, response):
        def generate_content(**kwargs):
            assert kwargs == {"model": "test-model", "contents": "question"}
            return response

        self.models = SimpleNamespace(generate_content=generate_content)


def install_fake_client(monkeypatch, response):
    monkeypatch.setattr("app.ask.get_api_key", lambda: "test-key")

    def create_client(api_key):
        assert api_key == "test-key"
        return FakeClient(response)

    monkeypatch.setattr(
        "app.ask.genai.Client",
        create_client,
    )


def test_ask_measures_latency_and_token_counts(monkeypatch):
    ticks = iter([1.0, 2.25])
    monkeypatch.setattr("app.ask.time.perf_counter", lambda: next(ticks))
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=8),
        text="answer",
    )
    install_fake_client(monkeypatch, response)

    result = ask("question", "test-model")

    assert result.elapsed_seconds == 1.25
    assert result.input_tokens == 12
    assert result.output_tokens == 8
    assert result.response_text == "answer"
    assert result.error is None


def test_ask_handles_missing_usage_metadata(monkeypatch):
    ticks = iter([1.0, 1.5])
    monkeypatch.setattr("app.ask.time.perf_counter", lambda: next(ticks))
    response = SimpleNamespace(usage_metadata=None, text="answer")
    install_fake_client(monkeypatch, response)

    result = ask("question", "test-model")

    assert result.input_tokens is None
    assert result.output_tokens is None


def test_summary_compares_model_totals(capsys):
    results = [
        Measurement("model-a", "q1", 1.0, 10, 5),
        Measurement("model-a", "q2", 3.0, None, 7),
        Measurement("model-b", "q1", 2.0, 8, 4, error="limited", status_code=429),
    ]

    print_summary(results, ["model-a", "model-b"])

    output = capsys.readouterr().out
    assert "model-a | 2/2 | 2.00s | 10（一部不明） | 12 | 0" in output
    assert "model-b | 0/1 | — | 不明 | 不明 | 1" in output
