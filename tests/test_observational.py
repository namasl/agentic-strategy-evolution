"""Tests for observational mode — campaigns where the executor probes a live
target system instead of evolving code in a git worktree.
"""
import contextlib
import json
import shutil
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orchestrator.dispatch import StubDispatcher
from orchestrator.engine import Engine
from orchestrator.iteration import IterationOutcome, run_iteration
from orchestrator.llm_dispatch import LLMDispatcher


class _CLIStub(StubDispatcher):
    """StubDispatcher with the CLIDispatcher surface area iteration.py
    needs (override_cwd context manager, model/max_turns attrs).
    """

    def __init__(self, work_dir, **_kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super().__init__(work_dir)
        self.model = "stub"
        self.max_turns = 1

    @contextlib.contextmanager
    def override_cwd(self, _cwd):
        yield


TEMPLATES_DIR = (
    Path(__file__).resolve().parent.parent / "orchestrator" / "templates"
)


def _campaign(observational: bool, repo_path: Path | None = None) -> dict:
    target = {
        "name": "TestSystem",
        "description": "A live target with no code to evolve.",
        "observable_metrics": ["latency_ms"],
        "controllable_knobs": ["config"],
    }
    if observational:
        target["observational"] = True
    if repo_path is not None:
        target["repo_path"] = str(repo_path)
    return {
        "research_question": "Does the live target behave?",
        "target_system": target,
        "prompts": {
            "methodology_layer": "prompts/methodology",
            "domain_adapter_layer": None,
        },
    }


# ---------------------------------------------------------------------------
# _validate_campaign
# ---------------------------------------------------------------------------


class TestCampaignValidation:
    def test_observational_true_accepted(self, tmp_path):
        campaign = _campaign(observational=True)
        # Must not raise.
        LLMDispatcher._validate_campaign(campaign)

    def test_observational_false_accepted(self, tmp_path):
        campaign = _campaign(observational=False)
        LLMDispatcher._validate_campaign(campaign)

    def test_observational_omitted_accepted(self, tmp_path):
        campaign = _campaign(observational=False)
        assert "observational" not in campaign["target_system"]
        LLMDispatcher._validate_campaign(campaign)

    def test_observational_non_bool_rejected(self):
        campaign = _campaign(observational=False)
        campaign["target_system"]["observational"] = "yes"
        with pytest.raises(ValueError, match="observational.*must be a bool"):
            LLMDispatcher._validate_campaign(campaign)


# ---------------------------------------------------------------------------
# _build_context — prompt fragment selection
# ---------------------------------------------------------------------------


class TestPromptFragmentSelection:
    """The execution_environment and worktree_constraint placeholders swap
    based on target_system.observational. The prompt loader will substitute
    them into the design and execute_analyze templates.
    """

    def _dispatcher(self, tmp_path, observational: bool) -> LLMDispatcher:
        # Seed the work_dir with the run_id only — no API key needed because
        # _build_context never calls the LLM.
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "runs" / "iter-1").mkdir(parents=True)
        return LLMDispatcher(
            work_dir=work_dir,
            campaign=_campaign(observational=observational),
            completion_fn=lambda **kw: None,
        )

    def test_default_is_worktree(self, tmp_path):
        d = self._dispatcher(tmp_path, observational=False)
        ctx = d._build_context("planner", "design", iteration=1, perspective=None)
        assert "isolated git worktree" in ctx["execution_environment"]
        assert "Worktree isolation assumed" in ctx["worktree_constraint"]

    def test_observational_swaps_text(self, tmp_path):
        d = self._dispatcher(tmp_path, observational=True)
        ctx = d._build_context("planner", "design", iteration=1, perspective=None)
        assert "Observational" in ctx["worktree_constraint"]
        assert "no per-iteration git isolation" in ctx["execution_environment"]
        assert "git worktree" not in ctx["execution_environment"]

    def test_design_template_renders_with_observational_constraint(self, tmp_path):
        """End-to-end: load the real design.md template with observational
        context and confirm the worktree paragraph is replaced.
        """
        d = self._dispatcher(tmp_path, observational=True)
        ctx = d._build_context("planner", "design", iteration=1, perspective=None)
        rendered = d.loader.load("design", ctx)
        assert "Worktree isolation assumed" not in rendered
        assert "Observational campaign" in rendered

    def test_execute_analyze_template_renders_with_observational_env(self, tmp_path):
        """End-to-end: load execute_analyze.md with observational context.
        The {{iter_dir}} placeholder inside the observational text must also
        be replaced (loader does sequential substitution).
        """
        d = self._dispatcher(tmp_path, observational=True)
        # _build_context for execute-analyze needs a bundle.yaml and handoff.md
        bundle_path = d.work_dir / "runs" / "iter-1" / "bundle.yaml"
        bundle_path.write_text("metadata:\n  iteration: 1\n")
        (d.work_dir / "handoff.md").write_text("(stub handoff)")
        (d.work_dir / "runs" / "iter-1" / "problem.md").write_text("(stub problem)")
        ctx = d._build_context(
            "executor", "execute-analyze", iteration=1, perspective=None,
        )
        rendered = d.loader.load("execute_analyze", ctx)
        assert "isolated git worktree" not in rendered
        assert "no per-iteration git isolation" in rendered
        assert "{{iter_dir}}" not in rendered  # must be substituted


# ---------------------------------------------------------------------------
# Iteration loop: observational mode skips worktree creation
# ---------------------------------------------------------------------------


def _setup_observational_iteration(
    tmp_path: Path, monkeypatch, *, repo_path: Path,
):
    """Prepare a work_dir + campaign that uses observational mode and a
    repo_path that is NOT a git repo. With the observational gate working,
    run_iteration must complete without ever hitting create_experiment_worktree.
    """
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    for t in ("state.json", "ledger.json", "principles.json"):
        shutil.copy(TEMPLATES_DIR / t, work_dir / t)
    state = json.loads((work_dir / "state.json").read_text())
    state["run_id"] = "test"
    (work_dir / "state.json").write_text(json.dumps(state, indent=2))

    campaign = _campaign(observational=True, repo_path=repo_path)

    import orchestrator.iteration as ri

    def stub_factory(work_dir, campaign, model=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return StubDispatcher(work_dir)

    monkeypatch.setattr(ri, "LLMDispatcher", stub_factory)
    # When repo_path is set, iteration.py would normally instantiate a
    # CLIDispatcher. Replace it with a stub that exposes the same surface
    # iteration.py touches (override_cwd, model, max_turns).
    monkeypatch.setattr(
        "orchestrator.cli_dispatch.CLIDispatcher",
        lambda **kw: _CLIStub(kw["work_dir"]),
    )
    monkeypatch.setattr(
        ri, "HumanGate",
        lambda: MagicMock(prompt=MagicMock(return_value=("approve", None))),
    )
    return work_dir, campaign


class TestObservationalIterationFlow:
    def test_runs_without_git_repo(self, tmp_path, monkeypatch):
        """A non-git repo_path + observational=true must not raise
        FileNotFoundError('Not a git repository') and must complete the
        iteration. This is the regression for the magic.yaml campaign.
        """
        repo = tmp_path / "live-target"
        repo.mkdir()  # NOT a git repo — no .git/ here.

        work_dir, campaign = _setup_observational_iteration(
            tmp_path, monkeypatch, repo_path=repo,
        )
        result = run_iteration(campaign, work_dir, iteration=1)
        assert result == IterationOutcome.COMPLETED
        assert Engine(work_dir).phase == "DONE"

    def test_no_experiment_worktree_created(self, tmp_path, monkeypatch):
        repo = tmp_path / "live-target"
        repo.mkdir()
        work_dir, campaign = _setup_observational_iteration(
            tmp_path, monkeypatch, repo_path=repo,
        )

        # Replace create_experiment_worktree with a sentinel that fails the
        # test if it is ever called. The iteration import is local, so patch
        # at the source module.
        called = {"n": 0}

        def must_not_call(*a, **kw):
            called["n"] += 1
            raise AssertionError(
                "create_experiment_worktree must not be called in "
                "observational mode"
            )

        monkeypatch.setattr(
            "orchestrator.worktree.create_experiment_worktree", must_not_call,
        )

        run_iteration(campaign, work_dir, iteration=1)

        assert called["n"] == 0
        # No .experiment_id file should be written in observational mode.
        assert not (work_dir / "runs" / "iter-1" / ".experiment_id").exists()
        # No .nous-experiments/ directory should appear in the target.
        assert not (repo / ".nous-experiments").exists()
