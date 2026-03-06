from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_packaged_layout(package_root: Path) -> None:
    assert (package_root / "main.py").exists()
    assert (package_root / "ai.py").exists()
    assert (package_root / "protocol.py").exists()
    assert (package_root / "common.py").exists()
    assert (package_root / "SDK" / "__init__.py").exists()
    assert (package_root / "tools" / "setup_native.py").exists()


def _run_packaging_script(script_name: str, output_dir: Path) -> None:
    subprocess.run(
        ["bash", f"AI/{script_name}", str(output_dir)],
        check=True,
    )


def test_zip_rand_creates_runnable_layout(tmp_path: Path) -> None:
    package_root = tmp_path / "random-package"
    _run_packaging_script("zip_rand.sh", package_root)
    _assert_packaged_layout(package_root)
    sys.path.insert(0, str(package_root))
    try:
        module = _load_module("packaged_random_ai", package_root / "ai.py")
        assert hasattr(module, "AI")
    finally:
        sys.path.remove(str(package_root))


def test_zip_mcts_and_zip_greedy_include_expected_support_files(tmp_path: Path) -> None:
    greedy_root = tmp_path / "greedy-package"
    mcts_root = tmp_path / "mcts-package"
    _run_packaging_script("zip_greedy.sh", greedy_root)
    _run_packaging_script("zip_mcts.sh", mcts_root)
    _assert_packaged_layout(greedy_root)
    _assert_packaged_layout(mcts_root)
    assert (greedy_root / "greedy_runtime.py").exists()
    assert not (greedy_root / "ai_greedy.py").exists()
    assert (mcts_root / "greedy_runtime.py").exists()
    assert (mcts_root / "ai_greedy.py").exists()
    assert not (mcts_root / "AI" / "AI_expert").exists()


def test_gitignore_covers_transient_directories() -> None:
    content = Path(".gitignore").read_text()
    for pattern in ("build/", "__pycache__/", ".pytest_cache/"):
        assert pattern in content


def test_main_entrypoint_uses_supplied_ai_class(monkeypatch) -> None:
    import AI.main as entry

    observed = SimpleNamespace(agent=None)

    class DummyAI:
        pass

    def fake_run_agent(agent) -> None:
        observed.agent = agent

    monkeypatch.setattr(entry, "run_agent", fake_run_agent)
    entry.main(DummyAI)
    assert isinstance(observed.agent, DummyAI)


def test_zip_script_runs_without_explicit_output_dir() -> None:
    completed = subprocess.run(
        ["bash", "AI/zip_rand.sh"],
        check=True,
        capture_output=True,
        text=True,
    )
    package_root = Path(completed.stdout.strip())
    try:
        assert package_root.exists()
        _assert_packaged_layout(package_root)
    finally:
        shutil.rmtree(package_root)
