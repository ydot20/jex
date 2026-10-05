"""Orchestration-only tests; no ML libraries, models, datasets or GPU required.

Run independently of the ML pytest fixtures:
    python -m unittest discover -s tests -p test_colab_runner.py -v
"""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import gc
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner_module = load_module("colab_full_run", "scripts/colab_full_run.py")
builder_module = load_module("build_colab_notebook", "scripts/build_colab_notebook.py")


class ColabRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jex-runner-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runner = runner_module.Runner(self.root)
        self.runner.repo.mkdir()

    def stage(self, name, function, dependencies=(), outputs=(), always=False):
        with redirect_stdout(io.StringIO()):
            self.runner.stage(name, function, dependencies, outputs, always)

    def test_original_scope_and_gpu_kernel_install_order(self):
        self.assertEqual(list(runner_module.SIZES.values()), [400, 50, 150, 400])
        self.assertEqual(len(runner_module.HEADS), 5)
        self.assertEqual(len(runner_module.BENCH_MODELS), 4)
        self.assertEqual(runner_module.CONFIG["lora_max_records"], 3000)
        self.assertEqual(runner_module.BENCH_LIMIT, 30)
        self.assertEqual(runner_module.CONFIG["latency_ks"], [1, 2, 4, 8, 16])
        self.assertEqual(runner_module.CONFIG["async_rates"], [5, 20, 50])
        self.assertEqual(runner_module.CONFIG["async_n"], 200)
        self.runner.make_plan()
        names = [stage[0] for stage in self.runner.plan]
        self.assertEqual(len(names), 31)
        self.assertTrue({"extract", "train-lora", "latency", "async", "compare-jev"} <= set(names))
        self.assertFalse(any("full-eval" in name for name in names))
        self.assertLess(names.index("tests"), names.index("install-fla"))
        self.assertLess(names.index("install-fla"), names.index("bench-qwen3.5-4b-zs"))

    def test_command_failure_blocks_dependencies_but_not_independent_work(self):
        command = [sys.executable, "-c", 'print("visible failure"); raise SystemExit(9)']
        self.stage("bad", lambda: self.runner.command(command))
        self.assertEqual(self.runner.status["bad"]["status"], "failed")
        self.assertIn("visible failure", (self.runner.logs / "bad.log").read_text())
        called = []
        self.stage("dependent", lambda: called.append(True), ("bad",))
        self.assertFalse(called)
        self.assertEqual(self.runner.status["dependent"]["status"], "blocked")
        self.stage("independent", lambda: called.append(True))
        self.assertEqual(called, [True])

    def test_resume_requires_success_and_nonempty_outputs(self):
        path = self.root / "out.txt"
        calls = []

        def write():
            calls.append(True)
            path.write_text("ok")

        self.stage("write", write, outputs=[path])
        self.stage("write", write, outputs=[path])
        self.assertEqual(len(calls), 1)
        path.unlink()
        self.stage("write", write, outputs=[path])
        self.assertEqual(len(calls), 2)
        path.write_text("")
        self.stage("write", write, outputs=[path])
        self.assertEqual(len(calls), 3)

    def test_missing_expected_output_fails(self):
        self.stage("missing", lambda: None, outputs=[self.root / "missing"])
        self.assertEqual(self.runner.status["missing"]["status"], "failed")

    def test_config_mismatch_refused(self):
        (self.root / "run_config.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "differs"):
            runner_module.Runner(self.root)

    def test_jev_license_gate_runs_before_network(self):
        with self.assertRaises(runner_module.Blocked):
            self.runner.download_baseline("Jev")

    def test_bundle_excludes_raw_response_databases_secrets_and_features(self):
        (self.runner.harness / "cache").mkdir(parents=True)
        (self.runner.harness / "cache" / "responses.db").write_text("raw response data")
        (self.root / "token").write_text("secret")
        features = self.runner.repo / "artifacts" / "feats"
        features.mkdir(parents=True)
        (features / "student_train.pt").write_text("large features")
        (self.runner.repo / "artifacts" / "eval.json").write_text("{}")
        self.runner.bundle()
        self.runner.bundle()  # A refreshed manifest must not hash its own prior version.
        with zipfile.ZipFile(self.root / "results.zip") as archive:
            names = archive.namelist()
            self.assertIn("jex/artifacts/eval.json", names)
            self.assertIn("ATTRIBUTION.txt", names)
            self.assertIn(b"https://doi.org/10.5281/zenodo.23039006", archive.read("ATTRIBUTION.txt"))
            self.assertFalse(any("responses.db" in name or "student_train.pt" in name or name == "token" for name in names))
            manifest = json.loads(archive.read("bundle_manifest.json"))
            self.assertNotIn("bundle_manifest.json", [row["path"] for row in manifest])
            for row in manifest:
                self.assertEqual(hashlib.sha256(archive.read(row["path"])).hexdigest(), row["sha256"])

    def test_commands_close_subprocess_stdout(self):
        with warnings.catch_warnings(record=True) as observed, redirect_stdout(io.StringIO()):
            warnings.simplefilter("always", ResourceWarning)
            self.runner.command([sys.executable, "-c", 'print("ok")'])
            with self.assertRaises(subprocess.CalledProcessError):
                self.runner.command([sys.executable, "-c", "raise SystemExit(1)"])
            gc.collect()
        self.assertFalse([warning for warning in observed if issubclass(warning.category, ResourceWarning)])

    def test_notebook_embeds_exact_source_compiles_and_defaults_license_false(self):
        notebook = json.loads(builder_module.NOTEBOOK_PATH.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "notebook", "exec")
        tree = ast.parse("".join(notebook["cells"][1]["source"]))
        assignments = {
            node.targets[0].id: node.value
            for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
        source = builder_module.RUNNER_PATH.read_text()
        self.assertEqual(ast.literal_eval(assignments["RUNNER_SOURCE"]), source)
        self.assertFalse(ast.literal_eval(assignments["JEV_RESPONSE_LICENSE_ACK"]))
        self.assertTrue(ast.literal_eval(assignments["COLAB_OPTIONAL_PACKAGE_CLEANUP"]))
        self.assertNotIn("user confirmed", "".join(notebook["cells"][0]["source"]).lower())

    def package_lookup(self, versions):
        def lookup(name):
            if versions.get(name) is None:
                raise runner_module.package_metadata.PackageNotFoundError(name)
            return versions[name]
        return lookup

    def test_torchao_stable_version_classification_is_conservative(self):
        for version in ("0.10.0", "0.15.9", "0.10.0+cu128"):
            self.assertTrue(runner_module.torchao_is_incompatible(version))
        for version in ("0.16.0", "0.16", "0.17.0", "1.0.0", "0.16.0+cu128"):
            self.assertFalse(runner_module.torchao_is_incompatible(version))
        for version in ("0.16.0rc1", "0.16.0.dev0", "unrecognized"):
            self.assertIsNone(runner_module.torchao_is_incompatible(version))

    def test_colab_runtime_guard_requires_all_three_hosted_indicators(self):
        with patch.dict(runner_module.os.environ, {"COLAB_RELEASE_TAG": "release"}), \
             patch.object(runner_module.import_util, "find_spec", return_value=object()), \
             patch.object(Path, "is_dir", return_value=True):
            self.assertTrue(all(self.runner.colab_runtime_evidence().values()))
        with patch.dict(runner_module.os.environ, {}, clear=True), \
             patch.object(runner_module.import_util, "find_spec", return_value=object()), \
             patch.object(Path, "is_dir", return_value=True):
            self.assertFalse(all(self.runner.colab_runtime_evidence().values()))
        with patch.dict(runner_module.os.environ, {"COLAB_RELEASE_TAG": "release"}), \
             patch.object(runner_module.import_util, "find_spec", side_effect=ModuleNotFoundError), \
             patch.object(Path, "is_dir", return_value=True):
            self.assertFalse(all(self.runner.colab_runtime_evidence().values()))
        with patch.dict(runner_module.os.environ, {"COLAB_RELEASE_TAG": "release"}), \
             patch.object(runner_module.import_util, "find_spec", return_value=object()), \
             patch.object(Path, "is_dir", return_value=False):
            self.assertFalse(all(self.runner.colab_runtime_evidence().values()))

    def test_absent_or_compatible_torchao_is_unchanged(self):
        for version in (None, "0.16.0", "0.17.1"):
            versions = {"torch": "2.10.0", "peft": "0.21.2", "transformers": "5.14.1", "torchao": version}
            with patch.object(runner_module.package_metadata, "version", side_effect=self.package_lookup(versions)), \
                 patch.object(self.runner, "command") as command, redirect_stdout(io.StringIO()):
                self.runner.prepare_optional_backends()
                command.assert_not_called()

    def test_old_torchao_cleanup_requires_opt_in_and_hosted_colab(self):
        versions = {"torch": "2.10.0", "peft": "0.21.2", "transformers": "5.14.1", "torchao": "0.10.0"}
        for opt_in, hosted in ((False, True), (True, False), (False, False)):
            self.runner.allow_colab_optional_package_cleanup = opt_in
            with patch.object(runner_module.package_metadata, "version", side_effect=self.package_lookup(versions)), \
                 patch.object(self.runner, "colab_runtime_evidence", return_value={"hosted": hosted}), \
                 patch.object(self.runner, "command") as command, redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "fresh isolated virtual environment"):
                    self.runner.prepare_optional_backends()
                command.assert_not_called()
            record = json.loads((self.root / "optional_backend_cleanup.json").read_text())
            self.assertEqual(record["status"], "blocked")

    def test_opted_in_colab_removes_only_old_unused_torchao_and_logs_versions(self):
        versions = {"torch": "2.10.0", "peft": "0.21.2", "transformers": "5.14.1", "torchao": "0.10.0"}
        self.runner.allow_colab_optional_package_cleanup = True

        def uninstall(args, **kwargs):
            self.assertEqual(args, [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
            versions["torchao"] = None

        with patch.object(runner_module.package_metadata, "version", side_effect=self.package_lookup(versions)), \
             patch.object(self.runner, "colab_runtime_evidence", return_value={"hosted": True}), \
             patch.object(self.runner, "command", side_effect=uninstall) as command, redirect_stdout(io.StringIO()):
            self.runner.prepare_optional_backends()
            command.assert_called_once()
        record = json.loads((self.root / "optional_backend_cleanup.json").read_text())
        self.assertEqual(record["status"], "removed")
        self.assertEqual(record["before"]["torchao"], "0.10.0")
        self.assertIsNone(record["after"]["torchao"])
        self.assertEqual(record["before"]["torch"], record["after"]["torch"])

    def test_cleanup_refuses_unrecognized_versions_or_unexpected_peft(self):
        for peft_version, torchao_version in (("0.21.2", "0.16.0rc1"), ("0.22.0", "0.10.0")):
            versions = {"torch": "2.10.0", "peft": peft_version, "transformers": "5.14.1", "torchao": torchao_version}
            self.runner.allow_colab_optional_package_cleanup = True
            with patch.object(runner_module.package_metadata, "version", side_effect=self.package_lookup(versions)), \
                 patch.object(self.runner, "colab_runtime_evidence", return_value={"hosted": True}), \
                 patch.object(self.runner, "command") as command, redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    self.runner.prepare_optional_backends()
                command.assert_not_called()

    def test_cleanup_detects_unexpected_core_package_changes(self):
        versions = {"torch": "2.10.0", "peft": "0.21.2", "transformers": "5.14.1", "torchao": "0.10.0"}
        self.runner.allow_colab_optional_package_cleanup = True

        def unexpected_change(*args, **kwargs):
            versions.update(torchao=None, torch="changed")

        with patch.object(runner_module.package_metadata, "version", side_effect=self.package_lookup(versions)), \
             patch.object(self.runner, "colab_runtime_evidence", return_value={"hosted": True}), \
             patch.object(self.runner, "command", side_effect=unexpected_change), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "core package versions must remain unchanged"):
                self.runner.prepare_optional_backends()
        record = json.loads((self.root / "optional_backend_cleanup.json").read_text())
        self.assertEqual(record["status"], "failed")

    def test_notebook_streams_child_stdout_stderr_and_failure_status(self):
        script = self.root / "fake_runner.py"
        script.write_text('import sys; print("visible stdout", flush=True); print("visible stderr", file=sys.stderr, flush=True); sys.exit(3)')
        output = io.StringIO()
        context = {"RUNNER": script, "RUN_ROOT": self.root, "JEV_RESPONSE_LICENSE_ACK": False,
                   "COLAB_OPTIONAL_PACKAGE_CLEANUP": False}
        with redirect_stdout(output):
            exec(compile(builder_module.RUN_CELL, "notebook-run", "exec"), context)
        text = output.getvalue()
        self.assertIn("visible stdout", text)
        self.assertIn("visible stderr", text)
        self.assertIn("Runner exit code: 3", text)
        self.assertIn("INCOMPLETE", text)

    def test_builder_is_deterministic_and_repository_relative(self):
        source = builder_module.RUNNER_PATH.read_text()
        expected = builder_module.build_notebook(source)
        self.assertEqual(expected, json.loads(builder_module.NOTEBOOK_PATH.read_text()))
        self.assertEqual(builder_module.RUNNER_PATH, ROOT / "scripts" / "colab_full_run.py")
        self.assertEqual(builder_module.NOTEBOOK_PATH, ROOT / "notebooks" / "jex_gpu_full.ipynb")


if __name__ == "__main__":
    unittest.main(verbosity=2)
