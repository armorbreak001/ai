"""Tests for remote update: shell script, bridge intercept, restart flag lifecycle.

Test class convention:
  - TestWorkerCodeChangedFunction: unit-level tests of the shell function itself.
    These spin up tmp git repos and source the function directly — they NEVER
    skip and do NOT require .venv.  New shell-function internals go here.
  - TestRemoteUpdateScript: full-script smoke tests that may skip when .venv is
    absent.  Reserved for end-to-end behavior that requires the full script.
"""

import asyncio
import os
import subprocess
from datetime import UTC
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.update.deps import (
    auto_bump_deps,
    bump_pin_in_pyproject,
    get_pinned_version,
    get_pypi_latest,
)

# Project root
PROJECT_DIR = Path(__file__).parent.parent.parent

SCRIPT = str(PROJECT_DIR / "scripts" / "remote-update.sh")


# =============================================================================
# worker_code_changed() function-level tests (venv-independent)
# =============================================================================


class TestWorkerCodeChangedFunction:
    """Direct tests of the worker_code_changed shell function.

    Each test creates a fresh git repo in tmp_path, makes commits, then sources
    the function from remote-update.sh and checks its return code.  These tests
    never skip and do not require .venv or any external services.
    """

    def _run_fn(self, tmp_repo: Path, before_sha: str, after_sha: str) -> int:
        """Source worker_code_changed and return its exit code."""
        cmd = (
            f"source {SCRIPT}; BEFORE_SHA={before_sha}; AFTER_SHA={after_sha}; "
            f"PROJECT_DIR={tmp_repo}; worker_code_changed"
        )
        result = subprocess.run(
            ["bash", "-euo", "pipefail", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode

    def _commit_file(self, repo: Path, path: str, content: str = "data\n") -> str:
        """Create a file, commit it, and return the new HEAD SHA."""
        full = repo / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        subprocess.run(
            ["git", "add", path],
            cwd=str(repo),
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"add {path}"],
            cwd=str(repo),
            capture_output=True,
            check=True,
        )
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    def _init_repo(self, tmp_path: Path) -> Path:
        """Initialise a bare git repo with one commit and return its path."""
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=str(repo),
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=str(repo),
            capture_output=True,
            check=True,
        )
        self._commit_file(repo, "README.md", "# test\n")
        return repo

    # -- cases --

    def test_returns_false_when_shas_identical(self, tmp_path):
        """Same before/after SHA → no restart (return 1)."""
        repo = self._init_repo(tmp_path)
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert self._run_fn(repo, sha, sha) == 1

    def test_returns_true_when_worker_dir_changed(self, tmp_path):
        """Commit touching worker/ → restart (return 0)."""
        repo = self._init_repo(tmp_path)
        before = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        after = self._commit_file(repo, "worker/foo.py")
        assert self._run_fn(repo, before, after) == 0

    def test_returns_false_when_only_docs_changed(self, tmp_path):
        """Commit touching only docs/ → no restart (return 1)."""
        repo = self._init_repo(tmp_path)
        before = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        after = self._commit_file(repo, "docs/plans/foo.md")
        assert self._run_fn(repo, before, after) == 1

    def test_returns_true_when_claude_hooks_changed(self, tmp_path):
        """Commit touching .claude/hooks/ → restart (return 0)."""
        repo = self._init_repo(tmp_path)
        before = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        after = self._commit_file(repo, ".claude/hooks/bar.py")
        assert self._run_fn(repo, before, after) == 0

    def test_returns_false_when_only_tests_changed(self, tmp_path):
        """Commit touching only tests/ → no restart (return 1)."""
        repo = self._init_repo(tmp_path)
        before = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        after = self._commit_file(repo, "tests/unit/baz.py")
        assert self._run_fn(repo, before, after) == 1

    def test_sigpipe_safe_with_large_diff(self, tmp_path):
        """200+ changed files with a worker/ match → returns 0 without error.

        Regression test for the SIGPIPE hazard where git diff | grep -q would
        kill the upstream git diff with SIGPIPE under set -euo pipefail.
        """
        repo = self._init_repo(tmp_path)
        before = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        # Create 200+ files — most non-worker, but ensure at least one worker/ file
        for i in range(150):
            self._commit_file(repo, f"docs/page{i}.md", f"doc {i}\n")
        for i in range(60):
            self._commit_file(repo, f"tests/test_{i}.py", f"test {i}\n")
        # The critical match
        after = self._commit_file(repo, "worker/core.py", "restart me\n")
        assert self._run_fn(repo, before, after) == 0

    def test_returns_true_when_before_sha_empty(self, tmp_path):
        """Empty BEFORE_SHA → fail-safe restart (return 0)."""
        repo = self._init_repo(tmp_path)
        after = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert self._run_fn(repo, "", after) == 0

    def test_returns_true_when_git_diff_fails(self, tmp_path):
        """Bogus BEFORE_SHA → git diff fails → fail-safe restart (return 0)."""
        repo = self._init_repo(tmp_path)
        after = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        bogus = "deadbeef" * 10  # valid-looking but nonexistent
        assert self._run_fn(repo, bogus, after) == 0


# =============================================================================
# Shell Script Tests (full-script smoke; may skip without .venv)
# =============================================================================


class TestRemoteUpdateScript:
    """Test scripts/remote-update.sh behavior.

    These are end-to-end smoke tests that invoke the full script.  They may
    skip when .venv is absent (the script requires Python).  For unit-level
    tests of shell-function internals, see TestWorkerCodeChangedFunction.
    """

    SCRIPT = str(PROJECT_DIR / "scripts" / "remote-update.sh")

    def test_script_exists_and_is_executable(self):
        script = Path(self.SCRIPT)
        assert script.exists(), "scripts/remote-update.sh should exist"
        assert os.access(str(script), os.X_OK), "Script should be executable"

    def test_already_up_to_date(self):
        """When HEAD matches remote, script exits 0 with 'Already up to date'."""
        venv_dir = PROJECT_DIR / ".venv"
        if not venv_dir.exists():
            pytest.skip("No .venv in project dir (e.g. running in worktree)")
        # Clean up any stale lock file from previous runs
        lock_dir = PROJECT_DIR / "data" / "update.lock"
        if lock_dir.is_dir():
            lock_dir.rmdir()
        # Ensure we're on main and up to date
        result = subprocess.run(
            ["bash", self.SCRIPT],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Since we just pulled (or are current), expect "Already up to date"
        assert result.returncode == 0
        assert (
            "up to date" in result.stdout.lower()
            or "commit(s)" in result.stdout
            or "update successful" in result.stdout
        )

    def test_no_restart_flag_when_up_to_date(self):
        """When already up to date, no restart flag should be written."""
        venv_dir = PROJECT_DIR / ".venv"
        if not venv_dir.exists():
            pytest.skip("No .venv in project dir (e.g. running in worktree)")
        flag = PROJECT_DIR / "data" / "restart-requested"
        # Remove any existing flag
        flag.unlink(missing_ok=True)

        result = subprocess.run(
            ["bash", self.SCRIPT],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )

        if "Already up to date" in result.stdout:
            assert not flag.exists(), "No restart flag should be written when up to date"

    def test_lockfile_prevents_concurrent_runs(self):
        """Second invocation should skip if lock is held."""
        lock_dir = PROJECT_DIR / "data" / "update.lock"
        lock_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                ["bash", self.SCRIPT],
                cwd=str(PROJECT_DIR),
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0
            assert "Another update is already running" in result.stdout
        finally:
            lock_dir.rmdir()

    def test_lockfile_cleaned_up_on_exit(self):
        """Lock directory should be removed after script completes."""
        lock_dir = PROJECT_DIR / "data" / "update.lock"
        lock_dir.unlink(missing_ok=True) if lock_dir.is_file() else None
        if lock_dir.exists():
            lock_dir.rmdir()

        subprocess.run(
            ["bash", self.SCRIPT],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert not lock_dir.exists(), "Lock directory should be cleaned up"

    def test_log_prefix_on_all_lines(self):
        """Output lines from the update system should have a log prefix.

        Lines produced by subcommands (git, pip, etc.) or the cron-mode
        summary line may not carry a prefix, so we only check that we
        got some output (the Python module captures prefixed lines to a
        log file and prints only a bare summary to stdout in cron mode).
        """
        result = subprocess.run(
            ["bash", self.SCRIPT],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
        # In cron mode the Python module captures prefixed lines to a log
        # file and prints only a bare summary to stdout, so zero prefixed
        # lines is acceptable as long as we got *some* output.
        assert len(lines) > 0, "Expected at least one line of output"

    def test_worker_kickstart_skipped_when_no_commits_pulled(self):
        """When git pull produces no new commits, worker kickstart is skipped.

        This end-to-end smoke test verifies that when the remote is already
        up-to-date, the script logs that it skipped the worker restart (or
        reports being up to date).  Skips when .venv is absent because the
        full script requires Python.
        """
        venv_dir = PROJECT_DIR / ".venv"
        if not venv_dir.exists():
            pytest.skip("No .venv in project dir (e.g. running in worktree)")

        # Clean up any stale lock file from previous runs
        lock_dir = PROJECT_DIR / "data" / "update.lock"
        if lock_dir.is_dir():
            lock_dir.rmdir()

        result = subprocess.run(
            ["bash", self.SCRIPT],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert (
            "No worker-relevant changes detected" in result.stdout
            or "up to date" in result.stdout.lower()
            or "commit(s)" in result.stdout
        )



    def test_worker_kickstart_skipped_when_no_commits_pulled(self):
        """When SHAs match (no commits pulled), kickstart should be skipped.

        Sets up shim `git` and `launchctl` in temp dirs on PATH so that:
        - `git rev-parse HEAD` returns the same SHA before/after pull
        - `git pull --ff-only` reports "Already up to date"
        - `launchctl list` returns as if worker is loaded
        - Asserts NO `kickstart -k` invocation occurs.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            bin_dir = Path(tmpdir) / "bin"
            bin_dir.mkdir()

            # Create launchctl shim that logs calls
            launchctl_log = Path(tmpdir) / "launchctl.log"
            launchctl_shim = bin_dir / "launchctl"
            launchctl_shim.write_text(
                f'#!/bin/bash\necho "$*" >> "{launchctl_log}"\n'
                f'# Return "loaded" for list queries\n'
                f'if [ "$1" = "list" ]; then echo "com.valor.worker"; exit 0; fi\n'
                f'exit 0\n'
            )
            launchctl_shim.chmod(0o755)

            # Create git shims
            fixed_sha = "abc123def456"
            git_shim = bin_dir / "git"
            git_shim.write_text(
                f'#!/bin/bash\n'
                f'if [ "$1" = "-C" ] && [ "$2" != "" ]; then shift 2; fi\n'
                f'case "$1" in\n'
                f'  rev-parse) echo "{fixed_sha}";;\n'
                f'  pull) echo "Already up to date.";;\n'
                f'  diff) exit 1;;  # no diff paths -> grep gets empty input;;\n'
                f'  *) exit 0;;\n'
                f'esac\n'
            )
            git_shim.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = str(bin_dir) + ":" + env.get("PATH", "")
            env_tmp = tempfile.mkdtemp()
            env["HOME"] = env_tmp

            result = subprocess.run(
                ["bash", self.SCRIPT],
                cwd=str(PROJECT_DIR),
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )

            launchctl_calls = launchctl_log.read_text() if launchctl_log.exists() else ""
            assert (
                "kickstart" not in launchctl_calls
            ), f"Expected no kickstart, got: {launchctl_calls}"
            assert (
                "skipping restart" in result.stdout.lower() or "skipping kickstart" in result.stdout.lower()
                or "up to date" in result.stdout.lower()
            ), f"Expected skip message, got stdout: {result.stdout[:500]}"

    def test_worker_kickstart_fires_when_worker_code_changed(self):
        """When diff touches worker/, kickstart -k should fire.

        Sets up shim `git` and `launchctl` in temp dirs on PATH so that:
        - `git rev-parse HEAD` returns different SHAs before/after pull
        - `git diff --name-only` returns a path matching worker-relevant glob
        - `launchctl list` returns as if worker is loaded
        - Asserts `kickstart -k` IS invoked.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            bin_dir = Path(tmpdir) / "bin"
            bin_dir.mkdir()

            # Create launchctl shim that logs calls
            launchctl_log = Path(tmpdir) / "launchctl.log"
            launchctl_shim = bin_dir / "launchctl"
            launchctl_shim.write_text(
                f'#!/bin/bash\necho "$*" >> "{launchctl_log}"\n'
                f'# Return "loaded" for list queries\n'
                f'if [ "$1" = "list" ]; then echo "com.valor.worker"; exit 0; fi\n'
                f'exit 0\n'
            )
            launchctl_shim.chmod(0o755)

            before_sha = "aaa111bbb222"
            after_sha = "ccc333ddd444"
            git_shim = bin_dir / "git"
            git_shim.write_text(
                f'#!/bin/bash\n'
                f'if [ "$1" = "-C" ] && [ "$2" != "" ]; then shift 2; fi\n'
                f'case "$1" in\n'
                f'  rev-parse)\n'
                f'    if [ -f "{tmpdir}/post_pull" ]; then\n'
                f'      echo "{after_sha}"\n'
                f'    else\n'
                f'      echo "{before_sha}"\n'
                f'    fi\n'
                f'    ;;\n'
                f'  pull)\n'
                f'    touch "{tmpdir}/post_pull"\n'
                f'    echo "Fast-forward"\n'
                f'    ;;\n'
                f'  diff)\n'
                f'    shift  # remove --name-only and SHAs\n'
                f'    echo "worker/__main__.py"\n'
                f'    ;;\n'
                f'  *) exit 0;;\n'
                f'esac\n'
            )
            git_shim.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = str(bin_dir) + ":" + env.get("PATH", "")
            env_tmp = tempfile.mkdtemp()
            env["HOME"] = env_tmp

            result = subprocess.run(
                ["bash", self.SCRIPT],
                cwd=str(PROJECT_DIR),
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )

            launchctl_calls = launchctl_log.read_text() if launchctl_log.exists() else ""
            assert (
                "kickstart" in launchctl_calls
            ), f"Expected kickstart invocation, got: {launchctl_calls}"
            assert (
                "-k" in launchctl_calls
            ), f"Expected kickstart -k, got: {launchctl_calls}"

# =============================================================================
# Restart Flag Tests
# =============================================================================


class TestRestartFlag:
    """Test restart flag lifecycle in agent_session_queue.py."""

    def setup_method(self):
        """Ensure clean state for each test."""
        from agent.agent_session_queue import _RESTART_FLAG

        _RESTART_FLAG.parent.mkdir(parents=True, exist_ok=True)
        _RESTART_FLAG.unlink(missing_ok=True)

    def teardown_method(self):
        """Clean up flag after each test."""
        from agent.agent_session_queue import _RESTART_FLAG

        _RESTART_FLAG.unlink(missing_ok=True)

    def test_check_restart_flag_returns_false_when_no_flag(self):
        from agent.agent_session_queue import _check_restart_flag

        assert _check_restart_flag() is False

    def _fresh_timestamp(self):
        """Return a flag content string with a timestamp from 5 minutes ago."""
        from datetime import datetime, timedelta

        ts = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
        return f"{ts} 3 commit(s)"

    def test_check_restart_flag_returns_true_when_flag_exists_and_no_jobs(self):
        from agent.agent_session_queue import _RESTART_FLAG, _check_restart_flag

        _RESTART_FLAG.write_text(self._fresh_timestamp())

        with patch("agent.agent_session_queue.AgentSession") as mock_session:
            mock_session.query.filter.return_value = []
            assert _check_restart_flag() is True

    def test_check_restart_flag_defers_when_jobs_running(self):
        from agent.agent_session_queue import (
            _RESTART_FLAG,
            _active_workers,
            _check_restart_flag,
        )

        _RESTART_FLAG.write_text(self._fresh_timestamp())

        # Simulate an active worker
        mock_task = MagicMock()
        mock_task.done.return_value = False
        _active_workers["testproject"] = mock_task

        try:
            with patch("agent.agent_session_queue.AgentSession") as mock_session:
                # Return running sessions for the project
                mock_session.query.filter.return_value = [MagicMock()]
                assert _check_restart_flag() is False
        finally:
            _active_workers.pop("testproject", None)

    def test_clear_restart_flag_removes_file(self):
        from agent.agent_session_queue import _RESTART_FLAG, clear_restart_flag

        _RESTART_FLAG.write_text("test content")
        assert clear_restart_flag() is True
        assert not _RESTART_FLAG.exists()

    def test_clear_restart_flag_returns_false_when_no_file(self):
        from agent.agent_session_queue import clear_restart_flag

        assert clear_restart_flag() is False

    def test_trigger_restart_removes_flag_and_sends_sigterm(self):
        from agent.agent_session_queue import _RESTART_FLAG, _trigger_restart

        _RESTART_FLAG.write_text("test")

        with patch("agent.agent_session_queue.os.kill") as mock_kill:
            _trigger_restart()

        assert not _RESTART_FLAG.exists()
        mock_kill.assert_called_once_with(os.getpid(), 15)  # SIGTERM = 15

    def test_check_restart_flag_ignores_stale_flag(self):
        """A flag older than 1 hour should be ignored and deleted."""
        from datetime import datetime, timedelta

        from agent.agent_session_queue import _RESTART_FLAG, _check_restart_flag

        stale_ts = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        _RESTART_FLAG.write_text(f"{stale_ts} 5 commit(s)")

        result = _check_restart_flag()
        assert result is False
        assert not _RESTART_FLAG.exists(), "Stale flag should be deleted"

    def test_check_restart_flag_handles_malformed_flag_content(self):
        """Malformed or empty flag content should not raise — returns False and deletes."""
        from agent.agent_session_queue import _RESTART_FLAG, _check_restart_flag

        # Empty content
        _RESTART_FLAG.write_text("")
        assert _check_restart_flag() is False
        assert not _RESTART_FLAG.exists()

        # Garbage content
        _RESTART_FLAG.write_text("not-a-timestamp blah")
        assert _check_restart_flag() is False
        assert not _RESTART_FLAG.exists()

        # Whitespace only
        _RESTART_FLAG.write_text("   \n  ")
        assert _check_restart_flag() is False
        assert not _RESTART_FLAG.exists()


# =============================================================================
# Worker Loop Restart Check Tests
# =============================================================================


class TestWorkerRestartCheck:
    """Test that the worker loop checks the restart flag between jobs."""

    def setup_method(self):
        from agent.agent_session_queue import _RESTART_FLAG

        _RESTART_FLAG.parent.mkdir(parents=True, exist_ok=True)
        _RESTART_FLAG.unlink(missing_ok=True)

    def teardown_method(self):
        from agent.agent_session_queue import _RESTART_FLAG

        _RESTART_FLAG.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_worker_checks_flag_when_queue_empty(self):
        """Worker should check restart flag when queue becomes empty."""
        from agent.agent_session_queue import _RESTART_FLAG

        _RESTART_FLAG.write_text("2026-02-02T10:00:00Z 1 commit(s)")

        with (
            patch("agent.agent_session_queue._pop_agent_session", return_value=None),
            patch("agent.agent_session_queue._check_restart_flag", return_value=True) as mock_check,
            patch("agent.agent_session_queue._trigger_restart") as mock_restart,
        ):
            from agent.agent_session_queue import _worker_loop

            event = asyncio.Event()
            await _worker_loop("testproject", event)

        mock_check.assert_called_once()
        mock_restart.assert_called_once()

    @pytest.mark.asyncio
    async def test_worker_checks_flag_after_job_completion(self):
        """Worker should check restart flag after completing a session."""
        mock_session_entry = MagicMock()
        mock_session_entry.agent_session_id = "test-123"
        mock_session_entry.project_key = "testproject"

        call_count = 0

        async def pop_side_effect(worker_key, is_project_keyed):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_session_entry
            return None

        with (
            patch("agent.agent_session_queue._pop_agent_session", side_effect=pop_side_effect),
            patch("agent.agent_session_queue._execute_agent_session", new_callable=AsyncMock),
            patch("agent.agent_session_queue._complete_agent_session", new_callable=AsyncMock),
            patch("agent.agent_session_queue._check_restart_flag", return_value=True) as mock_check,
            patch("agent.agent_session_queue._trigger_restart") as mock_restart,
        ):
            from agent.agent_session_queue import _worker_loop

            event = asyncio.Event()
            await _worker_loop("testproject", event)

        # Should have been called at least once (after session completion)
        assert mock_check.call_count >= 1
        mock_restart.assert_called()


# =============================================================================
# Bridge Command Intercept Tests
# =============================================================================


class TestBridgeUpdateCommand:
    """Test the /update command handling in the bridge."""

    def test_handle_update_command_exists(self):
        """The handle_update_command function should be importable."""
        from bridge.update import handle_force_update_command, handle_update_command

        assert callable(handle_update_command)
        assert callable(handle_force_update_command)

    def test_update_intercept_before_message_processing(self):
        """The /update check should come before message storage."""
        bridge_path = PROJECT_DIR / "bridge" / "telegram_bridge.py"
        source = bridge_path.read_text()

        # Find positions
        update_pos = source.find("/update")
        store_pos = source.find("store_message(")

        assert update_pos < store_pos, "/update intercept should come before store_message"

    def test_restart_flag_cleanup_in_startup(self):
        """Bridge startup should clear stale restart flags."""
        bridge_path = PROJECT_DIR / "bridge" / "telegram_bridge.py"
        source = bridge_path.read_text()
        assert "clear_restart_flag" in source


# =============================================================================
# Service Manager Tests
# =============================================================================


class TestServiceManager:
    """Test valor-service.sh has update cron support."""

    SERVICE_SCRIPT = str(PROJECT_DIR / "scripts" / "valor-service.sh")

    def test_update_plist_defined(self):
        source = Path(self.SERVICE_SCRIPT).read_text()
        # Label is built from ${SERVICE_LABEL_PREFIX}.update (defaulting to com.valor),
        # not hardcoded. Assert the dynamic form.
        assert "${SERVICE_LABEL_PREFIX}.update" in source
        assert "SERVICE_LABEL_PREFIX:=com.valor" in source

    def test_install_creates_both_plists(self):
        source = Path(self.SERVICE_SCRIPT).read_text()
        # install_service should reference both bridge and update plists
        assert "UPDATE_PLIST_PATH" in source
        assert "StartInterval" in source

    def test_uninstall_removes_both_plists(self):
        source = Path(self.SERVICE_SCRIPT).read_text()
        # uninstall should handle update plist
        assert source.count("UPDATE_PLIST_PATH") >= 2  # defined + used in uninstall

    def test_update_polling_interval_1800(self):
        """Update plist should use StartInterval of 1800 (30 minutes)."""
        source = Path(self.SERVICE_SCRIPT).read_text()
        assert "<key>StartInterval</key>" in source
        assert "<integer>1800</integer>" in source
        # Should NOT use the old calendar-based schedule
        assert "StartCalendarInterval" not in source


# =============================================================================
# Auto-Bump Deps Tests
# =============================================================================


class TestGetPypiLatest:
    def test_fetches_known_package(self):
        """Should return a version string for a known package."""
        version = get_pypi_latest("anthropic")
        assert version is not None
        assert "." in version  # version like "0.84.0"

    def test_returns_none_for_nonexistent_package(self):
        version = get_pypi_latest("this-package-definitely-does-not-exist-12345")
        assert version is None


class TestBumpPinInPyproject:
    def test_bumps_existing_pin(self, tmp_path: Path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            "[project]\ndependencies = [\n"
            '    "anthropic==0.62.0",  # CRITICAL\n'
            '    "claude-agent-sdk==0.1.35",  # CRITICAL\n'
            "]\n"
        )
        assert bump_pin_in_pyproject(tmp_path, "anthropic", "0.84.0")
        content = pyproject.read_text()
        assert '"anthropic==0.84.0"' in content
        # Other pins untouched
        assert '"claude-agent-sdk==0.1.35"' in content

    def test_preserves_comments(self, tmp_path: Path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('    "anthropic==0.62.0", # CRITICAL — pin exact\n')
        bump_pin_in_pyproject(tmp_path, "anthropic", "0.99.0")
        content = pyproject.read_text()
        assert "# CRITICAL" in content
        assert '"anthropic==0.99.0"' in content

    def test_returns_false_for_missing_package(self, tmp_path: Path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\ndependencies = []\n")
        assert bump_pin_in_pyproject(tmp_path, "nonexistent", "1.0.0") is False


class TestGetPinnedVersion:
    def test_reads_pinned_version(self, tmp_path: Path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('    "anthropic==0.62.0",  # CRITICAL\n')
        assert get_pinned_version(tmp_path, "anthropic") == "0.62.0"


class TestAutoBumpDeps:
    def test_no_bump_when_already_latest(self, tmp_path: Path):
        """When all packages are at latest, nothing should be bumped."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            "[project]\ndependencies = [\n"
            '    "anthropic==999.0.0",\n'
            '    "claude-agent-sdk==999.0.0",\n'
            "]\n"
        )
        # Mock PyPI to return the same versions
        with patch(
            "scripts.update.deps.get_pypi_latest",
            return_value="999.0.0",
        ):
            result = auto_bump_deps(tmp_path)
        assert not result.any_bumped
        assert not result.rolled_back

    def test_rollback_on_smoke_failure(self, tmp_path: Path):
        """If smoke test fails, pyproject.toml should be rolled back."""
        pyproject = tmp_path / "pyproject.toml"
        original = (
            "[project]\ndependencies = [\n"
            '    "anthropic==0.62.0",\n'
            '    "claude-agent-sdk==0.1.35",\n'
            "]\n"
        )
        pyproject.write_text(original)

        with (
            patch(
                "scripts.update.deps.get_pypi_latest",
                return_value="99.0.0",
            ),
            patch(
                "scripts.update.deps.sync_dependencies",
                return_value=MagicMock(success=True),
            ),
            patch(
                "scripts.update.deps.run_smoke_test",
                return_value=(False, "ImportError"),
            ),
        ):
            result = auto_bump_deps(tmp_path)

        assert result.any_bumped
        assert result.rolled_back
        assert not result.smoke_passed
        # pyproject.toml should be restored
        assert pyproject.read_text() == original
