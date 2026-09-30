"""Tests for the two-stage evaluation pipeline.

Tests cover:
  - programmatic_recall_check: substring matching (full, partial, case-insensitive)
  - compute_f1: edge cases (perfect, both empty, one empty, partial, case-insensitive)
  - evaluate_response: stage 1 default, stage 2 always-on when opted in
  - llm_judge_coverage: mocked adapter call; route resolution from a project config
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from osprey.models import provider_registry
from osprey.models.provider_registry import ProviderRegistry
from osprey.services.channel_finder.benchmarks.evaluation import (
    ChannelExtractionResult,
    JudgeRoute,
    compute_f1,
    evaluate_response,
    llm_judge_coverage,
    programmatic_recall_check,
    resolve_judge,
)
from osprey.services.channel_finder.core.exceptions import (
    ConfigurationError,
    CoverageJudgeError,
)

_ROUTE = JudgeRoute("als-apg", "claude-sonnet-5", "https://gateway.example.org/v1", api_key="k")

# ---------------------------------------------------------------------------
# programmatic_recall_check
# ---------------------------------------------------------------------------


class TestProgrammaticRecallCheck:
    """Tests for stage-1 programmatic recall."""

    def test_all_found(self):
        """All expected channels present in text."""
        text = (
            "The recommended channels are SR:MAG:DIPOLE:B05:CURRENT:SP "
            "and SR:MAG:QUAD:Q1:CURRENT:RB."
        )
        expected = [
            "SR:MAG:DIPOLE:B05:CURRENT:SP",
            "SR:MAG:QUAD:Q1:CURRENT:RB",
        ]
        found, missing = programmatic_recall_check(text, expected)
        assert found == expected
        assert missing == []

    def test_partial(self):
        """Some expected channels missing from text."""
        text = "Found channel SR:MAG:DIPOLE:B05:CURRENT:SP in the system."
        expected = [
            "SR:MAG:DIPOLE:B05:CURRENT:SP",
            "SR:MAG:QUAD:Q1:CURRENT:RB",
        ]
        found, missing = programmatic_recall_check(text, expected)
        assert found == ["SR:MAG:DIPOLE:B05:CURRENT:SP"]
        assert missing == ["SR:MAG:QUAD:Q1:CURRENT:RB"]

    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        text = "The channel sr:mag:dipole:b05:current:sp is available."
        expected = ["SR:MAG:DIPOLE:B05:CURRENT:SP"]
        found, missing = programmatic_recall_check(text, expected)
        assert found == expected
        assert missing == []

    def test_none_found(self):
        """No expected channels in text."""
        text = "No relevant channels were found."
        expected = ["SR:MAG:DIPOLE:B05:CURRENT:SP"]
        found, missing = programmatic_recall_check(text, expected)
        assert found == []
        assert missing == expected

    def test_empty_expected(self):
        """Empty expected list returns empty found and missing."""
        found, missing = programmatic_recall_check("some text", [])
        assert found == []
        assert missing == []


# ---------------------------------------------------------------------------
# compute_f1
# ---------------------------------------------------------------------------


class TestComputeF1:
    """Tests for precision/recall/F1 computation."""

    def test_perfect(self):
        """Predicted equals expected -> (1.0, 1.0, 1.0)."""
        channels = ["A", "B", "C"]
        precision, recall, f1 = compute_f1(channels, channels)
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_empty_both(self):
        """Both empty -> (1.0, 1.0, 1.0)."""
        precision, recall, f1 = compute_f1([], [])
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_empty_predicted(self):
        """Predicted empty, expected non-empty -> (0.0, 0.0, 0.0)."""
        precision, recall, f1 = compute_f1([], ["A", "B"])
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_empty_expected(self):
        """Expected empty, predicted non-empty -> (0.0, 0.0, 0.0)."""
        precision, recall, f1 = compute_f1(["A", "B"], [])
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_partial(self):
        """Partial overlap gives correct precision/recall/F1."""
        predicted = ["A", "B", "C"]  # 3 predicted
        expected = ["A", "B", "D"]  # 3 expected, 2 overlap (A, B)

        precision, recall, f1 = compute_f1(predicted, expected)
        # tp=2, precision=2/3, recall=2/3, f1=2/3
        assert precision == pytest.approx(2 / 3)
        assert recall == pytest.approx(2 / 3)
        assert f1 == pytest.approx(2 / 3)

    def test_precision_and_recall_differ(self):
        """Different precision and recall values."""
        predicted = ["A", "B"]  # 2 predicted
        expected = ["A", "B", "C", "D"]  # 4 expected, 2 overlap

        precision, recall, f1 = compute_f1(predicted, expected)
        # tp=2, precision=2/2=1.0, recall=2/4=0.5
        assert precision == 1.0
        assert recall == 0.5
        expected_f1 = 2 * 1.0 * 0.5 / (1.0 + 0.5)
        assert f1 == pytest.approx(expected_f1)

    def test_case_insensitive(self):
        """F1 scoring is case-insensitive (EPICS PV convention)."""
        predicted = ["sr:mag:dipole:b05:current:sp"]
        expected = ["SR:MAG:DIPOLE:B05:CURRENT:SP"]
        precision, recall, f1 = compute_f1(predicted, expected)
        assert f1 == 1.0


# ---------------------------------------------------------------------------
# llm_judge_coverage (mocked)
# ---------------------------------------------------------------------------


class TestLlmJudgeCoverage:
    """Tests for LLM-based coverage judging with a mocked adapter call."""

    _PATCH = "osprey.models.providers.als_apg.execute_litellm_completion"

    @patch(_PATCH)
    def test_returns_covered_and_extras(self, mock_completion):
        """Indices resolve back to expected strings; extras pass through."""
        mock_completion.return_value = ChannelExtractionResult(
            covered_expected_indices=[0, 1],
            extra_recommended=["CH:Z"],
            reasoning="Agent enumerated A and B; also recommended Z.",
        )
        covered, extras = llm_judge_coverage("some response text", ["CH:A", "CH:B"], judge=_ROUTE)
        assert covered == ["CH:A", "CH:B"]
        assert extras == ["CH:Z"]
        mock_completion.assert_called_once()

    @patch(_PATCH)
    def test_out_of_range_indices_dropped(self, mock_completion):
        """Hallucinated indices (negative or beyond length) are filtered out."""
        mock_completion.return_value = ChannelExtractionResult(
            covered_expected_indices=[0, 5, -1, 99],
            extra_recommended=[],
            reasoning="Mix of valid and invalid indices.",
        )
        covered, extras = llm_judge_coverage("response", ["CH:A", "CH:B"], judge=_ROUTE)
        assert covered == ["CH:A"]
        assert extras == []

    @patch(_PATCH)
    def test_a_reply_without_a_verdict_fails_the_query(self, mock_completion):
        """A reply that is not a structured verdict is an error, never 'covered nothing'."""
        mock_completion.return_value = "raw string response"
        with pytest.raises(CoverageJudgeError, match="als-apg/claude-sonnet-5"):
            llm_judge_coverage("some response", ["CH:A"], judge=_ROUTE)

    @patch(_PATCH)
    def test_the_call_carries_the_route(self, mock_completion, monkeypatch):
        monkeypatch.delenv("ALS_APG_BASE_URL", raising=False)
        mock_completion.return_value = ChannelExtractionResult(
            covered_expected_indices=[], extra_recommended=[], reasoning=""
        )

        llm_judge_coverage("response", ["CH:A"], judge=_ROUTE)

        kwargs = mock_completion.call_args.kwargs
        assert kwargs["model_id"] == "claude-sonnet-5"
        assert kwargs["api_key"] == "k"
        assert kwargs["base_url"] == "https://gateway.example.org/v1"
        assert kwargs["max_tokens"] == 2048
        assert kwargs["temperature"] == 0.0
        assert kwargs["output_format"] is ChannelExtractionResult

    @patch(_PATCH)
    def test_a_failing_call_names_the_judge(self, mock_completion):
        upstream = RuntimeError("upstream 500")
        mock_completion.side_effect = upstream

        with pytest.raises(CoverageJudgeError) as info:
            llm_judge_coverage("response", ["CH:A"], judge=_ROUTE)

        assert "als-apg/claude-sonnet-5" in str(info.value)
        assert "upstream 500" in str(info.value)
        assert info.value.__cause__ is upstream


# ---------------------------------------------------------------------------
# resolve_judge
# ---------------------------------------------------------------------------


def _write_config(project_dir: Path, config: dict) -> None:
    (project_dir / "config.yml").write_text(yaml.safe_dump(config))


def _providers(**entries: dict) -> dict:
    return {"api": {"providers": entries}}


_ALS_APG_ENTRY = {
    "api_key": "${ALS_APG_API_KEY}",
    "base_url": "https://gateway.example.org/v1",
    "default_model": "claude-sonnet-5",
}


class TestResolveJudge:
    """The judge's provider, endpoint, key and model come from the project's config."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for name in ("ALS_APG_BASE_URL", "ANTHROPIC_API_KEY", "CBORG_API_KEY"):
            monkeypatch.delenv(name, raising=False)

    def test_a_facility_provider_judges_through_api_providers(self, tmp_path, monkeypatch):
        reg = ProviderRegistry()
        reg.register_provider("facility-gw", "osprey.models.providers.vllm", "VLLMProviderAdapter")
        monkeypatch.setattr(provider_registry, "_registry", reg)
        monkeypatch.setenv("FACILITY_GW_TOKEN", "tok")
        _write_config(
            tmp_path,
            _providers(
                **{
                    "facility-gw": {
                        "base_url": "https://gw.example.org/v1",
                        "api_key": "${FACILITY_GW_TOKEN}",
                        "default_model": "m1",
                        "models": ["m1"],
                    }
                }
            ),
        )

        route = resolve_judge(tmp_path, "facility-gw")
        with patch("osprey.models.providers.vllm.execute_litellm_completion") as mock_completion:
            mock_completion.return_value = ChannelExtractionResult(
                covered_expected_indices=[0], extra_recommended=[], reasoning=""
            )
            covered, _ = llm_judge_coverage("CH:A", ["CH:A"], judge=route)

        assert covered == ["CH:A"]
        kwargs = mock_completion.call_args.kwargs
        assert kwargs["model_id"] == "m1"
        assert kwargs["api_key"] == "tok"
        assert kwargs["base_url"] == "https://gw.example.org/v1"

    def test_an_exported_vendor_key_does_not_choose_the_judge(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        monkeypatch.setenv("CBORG_API_KEY", "cborg-key")
        monkeypatch.setenv("ALS_APG_API_KEY", "als-apg-key")
        _write_config(tmp_path, _providers(**{"als-apg": _ALS_APG_ENTRY}))

        route = resolve_judge(tmp_path, "als-apg")

        assert route.provider == "als-apg"
        assert route.api_key == "als-apg-key"

    def test_the_main_model_answers_when_no_judge_model_is_named(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALS_APG_API_KEY", "k")
        monkeypatch.setenv("OTHER_GW_KEY", "k2")
        config = _providers(
            **{
                "als-apg": _ALS_APG_ENTRY,
                "cborg": {
                    "api_key": "${OTHER_GW_KEY}",
                    "base_url": "https://other.example.org/v1",
                    "default_model": "other-default",
                },
            }
        )
        config["claude_code"] = {
            "provider": "als-apg",
            "default_model": "claude-haiku-4-5-20251001",
        }
        _write_config(tmp_path, config)

        assert resolve_judge(tmp_path, "als-apg").model_id == "claude-haiku-4-5-20251001"
        assert resolve_judge(tmp_path, "cborg").model_id == "other-default"

    def test_a_named_judge_model_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALS_APG_API_KEY", "k")
        _write_config(tmp_path, _providers(**{"als-apg": _ALS_APG_ENTRY}))

        route = resolve_judge(tmp_path, "als-apg", judge_model="claude-opus-5-5")

        assert route.model_id == "claude-opus-5-5"

    def test_the_key_may_come_from_the_project_env_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ALS_APG_API_KEY", raising=False)
        (tmp_path / ".env").write_text("ALS_APG_API_KEY=from-dotenv\n")
        _write_config(tmp_path, _providers(**{"als-apg": _ALS_APG_ENTRY}))

        assert resolve_judge(tmp_path, "als-apg").api_key == "from-dotenv"

    def test_the_endpoint_override_variable_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALS_APG_API_KEY", "k")
        monkeypatch.setenv("ALS_APG_BASE_URL", "https://override.example.org")
        _write_config(tmp_path, _providers(**{"als-apg": _ALS_APG_ENTRY}))

        assert resolve_judge(tmp_path, "als-apg").base_url == "https://override.example.org"

    def test_the_route_keeps_its_key_out_of_its_repr(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALS_APG_API_KEY", "sk-secret-value")
        _write_config(tmp_path, _providers(**{"als-apg": _ALS_APG_ENTRY}))

        route = resolve_judge(tmp_path, "als-apg")

        assert route.api_key == "sk-secret-value"
        assert "sk-secret-value" not in repr(route)

    @pytest.mark.parametrize(
        ("provider", "config", "match"),
        [
            pytest.param("als-apg", None, "config.yml", id="no-config"),
            pytest.param(
                "als-apg",
                _providers(cborg={"api_key": "k", "default_model": "m"}),
                "cborg",
                id="provider-not-configured",
            ),
            pytest.param(
                "no-such-gw",
                _providers(**{"no-such-gw": {"api_key": "k", "default_model": "m"}}),
                "no-such-gw",
                id="no-registered-adapter",
            ),
            pytest.param(
                "als-apg",
                _providers(**{"als-apg": {**_ALS_APG_ENTRY, "api_key": "${UNSET_JUDGE_KEY}"}}),
                r"UNSET_JUDGE_KEY.*\.env",
                id="unset-key-reference",
            ),
            pytest.param(
                "amsc-i2",
                _providers(
                    **{
                        "amsc-i2": {
                            "api_key": "k",
                            "base_url": "${UNSET_GW_URL}",
                            "default_model": "m",
                        }
                    }
                ),
                "base_url",
                id="unset-endpoint-reference",
            ),
        ],
    )
    def test_a_judge_it_cannot_resolve_is_refused_by_name(
        self, tmp_path, monkeypatch, provider, config, match
    ):
        monkeypatch.delenv("UNSET_JUDGE_KEY", raising=False)
        monkeypatch.delenv("UNSET_GW_URL", raising=False)
        monkeypatch.setenv("ALS_APG_API_KEY", "k")
        if config is not None:
            _write_config(tmp_path, config)

        with pytest.raises(CoverageJudgeError, match=match):
            resolve_judge(tmp_path, provider)

    def test_a_malformed_config_is_refused_naming_provider_and_path(self, tmp_path):
        (tmp_path / "config.yml").write_text("- a\n- b\n")

        with pytest.raises(CoverageJudgeError) as info:
            resolve_judge(tmp_path, "als-apg")

        assert "'als-apg'" in str(info.value)
        assert str(tmp_path / "config.yml") in str(info.value)
        assert isinstance(info.value.__cause__, ConfigurationError)


# ---------------------------------------------------------------------------
# evaluate_response
# ---------------------------------------------------------------------------


class TestEvaluateResponse:
    """Tests for the combined two-stage evaluation pipeline."""

    def test_missing_channels_no_judge(self):
        """Without opt-in, missing channels just return Stage 1 found list."""
        text = "Found channel CH:A in the response."
        expected = ["CH:A", "CH:B"]

        predicted, meta = evaluate_response(text, expected)

        assert predicted == ["CH:A"]
        assert meta["stage"] == 1
        assert meta["evaluation"] == "programmatic_recall_fail"
        assert meta["found"] == ["CH:A"]
        assert meta["missing"] == ["CH:B"]

        # Verify precision contract: stage-1 fallback always yields precision=1.0
        # because `found` is a subset of `expected` by construction.
        precision, recall, f1 = compute_f1(predicted, expected)
        assert precision == 1.0
        assert recall < 1.0

    @patch("osprey.services.channel_finder.benchmarks.evaluation.llm_judge_coverage")
    def test_all_found_invokes_judge(self, mock_judge):
        """Opt-in path: judge runs even when Stage 1 found everything."""
        mock_judge.return_value = (["CH:A", "CH:B"], [])
        text = "The channels are CH:A and CH:B in the final answer."
        expected = ["CH:A", "CH:B"]

        predicted, meta = evaluate_response(text, expected, judge=_ROUTE)

        assert predicted == ["CH:A", "CH:B"]
        assert meta["stage"] == 2
        assert meta["evaluation"] == "llm_judge"
        assert meta["llm_covered"] == ["CH:A", "CH:B"]
        assert meta["llm_extras"] == []
        mock_judge.assert_called_once_with(text, expected, judge=_ROUTE)
        assert meta["judge"] == "als-apg/claude-sonnet-5"

    @patch("osprey.services.channel_finder.benchmarks.evaluation.llm_judge_coverage")
    def test_missing_channels_runs_judge_when_opted_in(self, mock_judge):
        """Shorthand recovery: judge runs even when Stage 1 has missing channels."""
        # Agent used shorthand — Stage 1 sees zero literal hits, but the
        # judge interprets the prose and credits both channels as covered.
        mock_judge.return_value = (["CH:A", "CH:B"], [])
        text = "Use the full set of CH channels (both A and B)."
        expected = ["CH:A", "CH:B"]

        predicted, meta = evaluate_response(text, expected, judge=_ROUTE)

        assert predicted == ["CH:A", "CH:B"]
        assert meta["stage"] == 2
        assert meta["evaluation"] == "llm_judge"
        # Stage 1 still reported the literal-only view in meta for debuggability.
        assert meta["found"] == []
        assert meta["missing"] == ["CH:A", "CH:B"]
        mock_judge.assert_called_once_with(text, expected, judge=_ROUTE)
        assert meta["judge"] == "als-apg/claude-sonnet-5"

    @patch("osprey.services.channel_finder.benchmarks.evaluation.llm_judge_coverage")
    def test_judge_reports_extras(self, mock_judge):
        """Over-recommended extras are folded into predicted so precision dings."""
        mock_judge.return_value = (["CH:A"], ["CH:Z", "CH:Y"])
        text = "I recommend CH:A, plus CH:Z and CH:Y as bonus monitors."
        expected = ["CH:A", "CH:B"]

        predicted, meta = evaluate_response(text, expected, judge=_ROUTE)

        assert predicted == ["CH:A", "CH:Z", "CH:Y"]
        precision, recall, f1 = compute_f1(predicted, expected)
        # 1 hit / 3 predicted, 1 hit / 2 expected
        assert precision == pytest.approx(1 / 3)
        assert recall == pytest.approx(1 / 2)
        assert meta["llm_extras"] == ["CH:Z", "CH:Y"]
        mock_judge.assert_called_once_with(text, expected, judge=_ROUTE)
        assert meta["judge"] == "als-apg/claude-sonnet-5"

    @patch("osprey.services.channel_finder.benchmarks.evaluation.llm_judge_coverage")
    def test_a_judge_error_fails_the_query(self, mock_judge):
        """A query the judge could not score is not rescored by substring match."""
        mock_judge.side_effect = CoverageJudgeError("boom")
        text = "Channels CH:A and CH:B are recommended."
        expected = ["CH:A", "CH:B"]

        with pytest.raises(CoverageJudgeError, match="boom"):
            evaluate_response(text, expected, judge=_ROUTE)

    def test_empty_expected_skips_judge(self):
        """Empty expected list short-circuits — no LLM call."""
        with patch(
            "osprey.services.channel_finder.benchmarks.evaluation.llm_judge_coverage"
        ) as mock_judge:
            predicted, meta = evaluate_response("some text", [], judge=_ROUTE)

        assert predicted == []
        assert meta["stage"] == 1
        assert meta["evaluation"] == "programmatic_recall_only"
        mock_judge.assert_not_called()

    def test_no_channels_found(self):
        """No expected channels found in text at all (no judge opt-in)."""
        text = "I could not find any relevant channels."
        expected = ["CH:A", "CH:B"]

        predicted, meta = evaluate_response(text, expected)

        assert predicted == []
        assert meta["stage"] == 1
        assert meta["evaluation"] == "programmatic_recall_fail"
        assert meta["missing"] == ["CH:A", "CH:B"]
