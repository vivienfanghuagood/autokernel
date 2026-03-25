#!/usr/bin/env python3
"""
KernelBench Evaluation Harness -- Correctness + Performance for KernelBench problems.

Fully compatible with the KernelBench evaluation protocol:
  - 5 random input trials for correctness (atol=1e-2, rtol=1e-2)
  - 3 warmup + 100 timed runs for performance
  - Speedup vs PyTorch reference (Model)

Additional AutoKernel features:
  - Numerical stability probing (NaN/Inf detection)
  - Determinism checks (3 runs with same seed must be bitwise identical)
  - VRAM monitoring
  - Timeout protection (30s per kernel call)
  - hipBLASLt auto-tuning for fair BLAS baselines (ROCm)
  - Pure GPU kernel latency profiling (per-call CUDA event, percentile stats)
  - Greppable summary output (for log parsing by the agent)

Usage:
    uv run kernelbench/bench_kb.py                     # Full evaluation
    uv run kernelbench/bench_kb.py --quick              # Quick (3 trials, 30 timed)
    uv run kernelbench/bench_kb.py --correctness-only   # Skip performance
    uv run kernelbench/bench_kb.py --n-timed 200        # More timing iterations
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = PROJECT_DIR / "workspace"
KB_ACTIVE_DIR = WORKSPACE_DIR / "kb_active"
KERNEL_PY = PROJECT_DIR / "kernel.py"

# Ensure project root is on sys.path (for `from kernels.cuda._compile import ...`)
sys.path.insert(0, str(PROJECT_DIR))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2
DEFAULT_N_CORRECTNESS = 5
DEFAULT_N_WARMUP = 3
DEFAULT_N_TIMED = 100
TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------


class _Timeout:
    """Context manager that raises TimeoutError after `seconds`."""

    def __init__(self, seconds: float, msg: str = "timeout"):
        self.seconds = seconds
        self.msg = msg
        self._use_signal = hasattr(signal, "SIGALRM")  # Unix only

    def __enter__(self):
        if self._use_signal:
            self._prev = signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(int(self.seconds))
        else:
            self._timer = threading.Timer(self.seconds, self._thread_raise)
            self._timer.start()
        return self

    def __exit__(self, *_):
        if self._use_signal:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._prev)
        else:
            self._timer.cancel()

    def _handler(self, signum, frame):
        raise TimeoutError(self.msg)

    def _thread_raise(self):
        # Thread-based fallback for Windows -- limited, can't interrupt GPU ops
        pass


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------


def _load_module_from_path(path: Path, module_name: str):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_reference():
    """Load the reference Model, get_inputs, get_init_inputs from workspace."""
    ref_path = KB_ACTIVE_DIR / "reference.py"
    if not ref_path.exists():
        print("ERROR: No active KernelBench problem.")
        print("       Run: uv run kernelbench/bridge.py setup --level 1 --problem 1")
        sys.exit(1)
    mod = _load_module_from_path(ref_path, "_kb_reference")
    Model = getattr(mod, "Model", None)
    get_inputs = getattr(mod, "get_inputs", None)
    get_init_inputs = getattr(mod, "get_init_inputs", None)
    if Model is None or get_inputs is None:
        print("ERROR: reference.py must define Model, get_inputs(), get_init_inputs().")
        sys.exit(1)
    if get_init_inputs is None:
        get_init_inputs = lambda: []
    return Model, get_inputs, get_init_inputs


def load_kernel():
    """Load ModelNew from kernel.py."""
    if not KERNEL_PY.exists():
        print("ERROR: kernel.py not found.")
        print("       Run: uv run kernelbench/bridge.py setup --level 1 --problem 1")
        sys.exit(1)
    mod = _load_module_from_path(KERNEL_PY, "_kb_kernel")
    ModelNew = getattr(mod, "ModelNew", None)
    if ModelNew is None:
        print("ERROR: kernel.py must define a ModelNew class.")
        sys.exit(1)
    get_inputs = getattr(mod, "get_inputs", None)
    get_init_inputs = getattr(mod, "get_init_inputs", None)
    problem_meta = getattr(mod, "KERNELBENCH_PROBLEM", {})
    return ModelNew, get_inputs, get_init_inputs, problem_meta


def load_metadata() -> Dict[str, Any]:
    """Load active problem metadata."""
    meta_path = KB_ACTIVE_DIR / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------
# Tensor comparison
# ---------------------------------------------------------------------------


def _has_nan_inf(t) -> bool:
    import torch

    return bool(torch.isnan(t).any() or torch.isinf(t).any())


def _compare(output, expected, atol: float, rtol: float) -> Dict[str, Any]:
    """Compare two tensors. Returns match info."""
    import torch

    result: Dict[str, Any] = {
        "match": False,
        "reason": "",
        "max_abs_error": float("inf"),
        "mean_abs_error": float("inf"),
    }

    # Shape check
    if output.shape != expected.shape:
        result["reason"] = f"shape mismatch: {output.shape} vs {expected.shape}"
        return result

    # Cast to float32 for comparison
    out_f = output.detach().float().cpu()
    exp_f = expected.detach().float().cpu()

    # NaN/Inf symmetry
    out_nan = torch.isnan(out_f)
    exp_nan = torch.isnan(exp_f)
    if out_nan.any() or exp_nan.any():
        if not torch.equal(out_nan, exp_nan):
            result["reason"] = "NaN position mismatch"
            return result
        mask = ~out_nan
        if mask.any():
            out_f = out_f[mask]
            exp_f = exp_f[mask]
        else:
            result["match"] = True
            result["reason"] = "all NaN (matching)"
            result["max_abs_error"] = 0.0
            result["mean_abs_error"] = 0.0
            return result

    abs_err = (out_f - exp_f).abs()
    result["max_abs_error"] = float(abs_err.max())
    result["mean_abs_error"] = float(abs_err.mean())

    if torch.allclose(out_f, exp_f, atol=atol, rtol=rtol):
        result["match"] = True
        result["reason"] = "PASS"
    else:
        result["reason"] = (
            f"tolerance exceeded: max_abs={result['max_abs_error']:.6e}, "
            f"mean_abs={result['mean_abs_error']:.6e} (atol={atol}, rtol={rtol})"
        )

    return result


def _compare_outputs(output, expected, atol: float, rtol: float) -> Dict[str, Any]:
    """Compare outputs: tensors, tuples/lists of tensors, or scalars."""
    import torch

    if isinstance(output, torch.Tensor) and isinstance(expected, torch.Tensor):
        return _compare(output, expected, atol, rtol)

    if isinstance(output, (tuple, list)) and isinstance(expected, (tuple, list)):
        if len(output) != len(expected):
            return {
                "match": False,
                "reason": f"output count mismatch: {len(output)} vs {len(expected)}",
            }
        worst = {
            "match": True,
            "reason": "PASS",
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
        }
        for i, (o, e) in enumerate(zip(output, expected)):
            r = _compare_outputs(o, e, atol, rtol)
            if not r["match"]:
                return {"match": False, "reason": f"output[{i}]: {r['reason']}"}
            worst["max_abs_error"] = max(
                worst["max_abs_error"], r.get("max_abs_error", 0)
            )
        return worst

    # Scalar comparison
    try:
        diff = abs(float(output) - float(expected))
        if diff <= atol:
            return {
                "match": True,
                "reason": "PASS",
                "max_abs_error": diff,
                "mean_abs_error": diff,
            }
    except (TypeError, ValueError):
        pass

    return {
        "match": False,
        "reason": f"incomparable types: {type(output)} vs {type(expected)}",
    }


# ---------------------------------------------------------------------------
# Correctness evaluation
# ---------------------------------------------------------------------------


def run_correctness(
    model_new,
    model_ref,
    get_inputs_fn: Callable,
    n_trials: int = DEFAULT_N_CORRECTNESS,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    KernelBench-compatible correctness checks.

    Generate n_trials random input sets, run both models, compare within (atol, rtol).
    """
    import torch

    results: Dict[str, Any] = {
        "correctness": "FAIL",
        "trials_passed": 0,
        "trials_total": n_trials,
        "worst_max_abs_error": 0.0,
        "worst_mean_abs_error": 0.0,
        "details": [],
    }

    for trial in range(n_trials):
        trial_info: Dict[str, Any] = {"trial": trial, "status": "FAIL"}
        try:
            with _Timeout(TIMEOUT_SECONDS, f"trial {trial} timed out"):
                inputs = get_inputs_fn()
                inputs_dev = [
                    inp.to(device) if isinstance(inp, torch.Tensor) else inp
                    for inp in inputs
                ]

                with torch.no_grad():
                    expected = model_ref(*inputs_dev)
                with torch.no_grad():
                    output = model_new(*inputs_dev)

                cmp = _compare_outputs(output, expected, atol, rtol)
                trial_info["max_abs_error"] = cmp.get("max_abs_error", float("inf"))
                trial_info["mean_abs_error"] = cmp.get("mean_abs_error", float("inf"))

                if cmp["match"]:
                    trial_info["status"] = "PASS"
                    results["trials_passed"] += 1
                    results["worst_max_abs_error"] = max(
                        results["worst_max_abs_error"], cmp.get("max_abs_error", 0)
                    )
                    results["worst_mean_abs_error"] = max(
                        results["worst_mean_abs_error"], cmp.get("mean_abs_error", 0)
                    )
                else:
                    trial_info["reason"] = cmp["reason"]

        except TimeoutError as e:
            trial_info["status"] = "TIMEOUT"
            trial_info["reason"] = str(e)
        except Exception as e:
            trial_info["status"] = "ERROR"
            trial_info["reason"] = f"{type(e).__name__}: {e}"

        results["details"].append(trial_info)

        # Early exit on failure
        if trial_info["status"] != "PASS":
            break

    if results["trials_passed"] == n_trials:
        results["correctness"] = "PASS"

    return results


def run_stability(
    model_new,
    get_inputs_fn: Callable,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Test numerical stability: check for NaN/Inf on normal inputs."""
    import torch

    result: Dict[str, Any] = {"stability": "PASS", "details": []}

    for trial in range(3):
        try:
            inputs = get_inputs_fn()
            inputs_dev = [
                inp.to(device) if isinstance(inp, torch.Tensor) else inp
                for inp in inputs
            ]
            with torch.no_grad():
                output = model_new(*inputs_dev)

            if isinstance(output, torch.Tensor):
                if _has_nan_inf(output):
                    result["stability"] = "WARN"
                    result["details"].append(f"trial {trial}: output contains NaN/Inf")
            elif isinstance(output, (tuple, list)):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor) and _has_nan_inf(o):
                        result["stability"] = "WARN"
                        result["details"].append(
                            f"trial {trial}: output[{i}] contains NaN/Inf"
                        )

        except Exception as e:
            result["stability"] = "FAIL"
            result["details"].append(f"trial {trial}: {type(e).__name__}: {e}")

    return result


def run_determinism(
    model_new,
    get_inputs_fn: Callable,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run 3 times with identical inputs, check bitwise reproducibility."""
    import torch

    result: Dict[str, Any] = {"determinism": "PASS", "details": []}

    try:
        torch.manual_seed(42)
        inputs = get_inputs_fn()
        inputs_dev = [
            inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in inputs
        ]
        inputs_copies = [
            [
                inp.clone() if isinstance(inp, torch.Tensor) else copy.deepcopy(inp)
                for inp in inputs_dev
            ]
            for _ in range(3)
        ]

        outputs = []
        for i in range(3):
            with torch.no_grad():
                out = model_new(*inputs_copies[i])
            if isinstance(out, torch.Tensor):
                outputs.append(out.clone())
            elif isinstance(out, (tuple, list)):
                outputs.append(
                    tuple(o.clone() if isinstance(o, torch.Tensor) else o for o in out)
                )
            else:
                outputs.append(out)

        for i in range(1, 3):
            if isinstance(outputs[0], torch.Tensor):
                if not torch.equal(outputs[0], outputs[i]):
                    result["determinism"] = "WARN"
                    result["details"].append(f"run 0 vs run {i}: not bitwise identical")
            elif isinstance(outputs[0], (tuple, list)):
                for j, (a, b) in enumerate(zip(outputs[0], outputs[i])):
                    if isinstance(a, torch.Tensor) and not torch.equal(a, b):
                        result["determinism"] = "WARN"
                        result["details"].append(
                            f"run 0 vs run {i}, output[{j}]: not bitwise identical"
                        )

    except Exception as e:
        result["determinism"] = "ERROR"
        result["details"].append(f"{type(e).__name__}: {e}")

    return result


# ---------------------------------------------------------------------------
# Performance evaluation
# ---------------------------------------------------------------------------


def run_performance(
    model_new,
    model_ref,
    get_inputs_fn: Callable,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_timed: int = DEFAULT_N_TIMED,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    KernelBench-compatible performance benchmarking via CUDA event timing.
    Returns speedup = ref_time / kernel_time.
    """
    import torch

    result: Dict[str, Any] = {
        "kernel_time_ms": 0.0,
        "reference_time_ms": 0.0,
        "speedup": 0.0,
        "kernel_times": [],
        "reference_times": [],
    }

    torch.manual_seed(0)
    inputs = get_inputs_fn()
    inputs_dev = [
        inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in inputs
    ]

    def _time_model(model, inputs_list, n_warm, n_iter):
        """Time a model using CUDA events."""
        for _ in range(n_warm):
            with torch.no_grad():
                model(*inputs_list)
        torch.cuda.synchronize()

        times = []
        for _ in range(n_iter):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                model(*inputs_list)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        return times

    try:
        ref_times = _time_model(model_ref, inputs_dev, n_warmup, n_timed)
        result["reference_times"] = ref_times
        result["reference_time_ms"] = _robust_median(ref_times)

        kernel_times = _time_model(model_new, inputs_dev, n_warmup, n_timed)
        result["kernel_times"] = kernel_times
        result["kernel_time_ms"] = _robust_median(kernel_times)

        if result["kernel_time_ms"] > 0:
            result["speedup"] = result["reference_time_ms"] / result["kernel_time_ms"]

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def _robust_median(times: List[float]) -> float:
    """Trimmed median: median of the middle 80% of measurements."""
    if not times:
        return 0.0
    s = sorted(times)
    n = len(s)
    trim = max(1, n // 10)
    trimmed = s[trim : n - trim] if n > 2 * trim else s
    mid = len(trimmed) // 2
    if len(trimmed) % 2 == 0:
        return (trimmed[mid - 1] + trimmed[mid]) / 2
    return trimmed[mid]


# ---------------------------------------------------------------------------
# hipBLASLt auto-tuning (ROCm)
# ---------------------------------------------------------------------------


TUNING_N_WARMUP = 20
TUNING_FILE_NAME = "hipblaslt_tuning.csv"


def _try_enable_hipblaslt_tuning(
    model_ref,
    get_inputs_fn: Callable,
    device: str = "cuda",
) -> bool:
    """Auto-tune hipBLASLt for the reference model to establish a fair baseline.

    On ROCm, ``torch._scaled_mm`` and ``torch.mm`` use hipBLASLt whose
    default heuristic picks a mediocre algorithm.  Running the tunable-ops
    warmup phase selects the fastest algorithm for the actual problem shape,
    which can be 1.5-2x faster.  Without this, any custom Triton kernel
    appears to "win" by comparing against an unfairly slow baseline.

    The tuning result is persisted to ``hipblaslt_tuning.csv`` so subsequent
    runs skip the search.

    NOTE: max_tuning_iterations is kept low (5) because higher values can
    trigger memory access faults in hipBLASLt on certain shape/dtype combos
    (ROCm bug where later candidate algorithms access out-of-bounds memory).
    5 iterations is sufficient to find a near-optimal algorithm.
    """
    try:
        import torch
        import torch.cuda.tunable as tunable
    except (ImportError, AttributeError):
        return False

    tuning_path = PROJECT_DIR / TUNING_FILE_NAME
    already_tuned = tuning_path.exists()

    # Load cached tuning if file exists and is non-empty
    if already_tuned and tuning_path.stat().st_size > 0:
        tunable.enable(True)
        tunable.set_filename(str(tuning_path))
        tunable.read_file(str(tuning_path))
        print(f"  Loaded hipBLASLt tuning from {tuning_path.name}")
        return True
    elif already_tuned:
        # Empty/corrupt file from a previous crashed tuning — remove it
        tuning_path.unlink(missing_ok=True)

    # Check for a marker file indicating that tuning has previously crashed
    # on this GPU.  If so, skip entirely to avoid GPU state corruption
    # (a crashed tuning subprocess can degrade performance on the same GPU
    # even for the parent process).
    tuning_skip_marker = PROJECT_DIR / ".hipblaslt_tuning_skip"
    if tuning_skip_marker.exists():
        print("  hipBLASLt tuning skipped (previously crashed on this GPU).")
        print("  Delete .hipblaslt_tuning_skip to retry.")
        return False

    # Run tuning in a subprocess.  hipBLASLt tuning can SIGABRT on certain
    # shape/dtype combos (ROCm bug: some candidate algorithms access
    # out-of-bounds GPU memory).  Subprocess isolation keeps the main
    # benchmark process alive.  When it crashes, we create a skip marker
    # so we never try again on this GPU.
    print("  Running hipBLASLt auto-tuning in subprocess (one-time)...")

    ref_path = str(KB_ACTIVE_DIR / "reference.py")
    tuning_script = (
        "import torch, torch.cuda.tunable as tunable, importlib.util\n"
        "tunable.enable(True)\n"
        "tunable.tuning_enable(True)\n"
        "tunable.set_max_tuning_iterations(5)\n"
        "tunable.set_max_tuning_duration(10)\n"
        f'tunable.set_filename("{tuning_path}")\n'
        f'spec = importlib.util.spec_from_file_location("_ref", "{ref_path}")\n'
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "model = mod.Model(*getattr(mod, 'get_init_inputs', lambda:[])()).to('cuda').eval()\n"
        "inputs = [t.to('cuda') if isinstance(t, __import__('torch').Tensor) else t for t in mod.get_inputs()]\n"
        f"for _ in range({TUNING_N_WARMUP}):\n"
        "    with __import__('torch').no_grad(): model(*inputs)\n"
        "    __import__('torch').cuda.synchronize()\n"
        "tunable.tuning_enable(False)\n"
        f'tunable.write_file("{tuning_path}")\n'
        'print("TUNING_OK")\n'
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", tuning_script],
            capture_output=True,
            text=True,
            timeout=120,
            env=dict(os.environ),
        )
        if result.returncode == 0 and "TUNING_OK" in result.stdout:
            tunable.enable(True)
            tunable.set_filename(str(tuning_path))
            tunable.read_file(str(tuning_path))
            print(f"  hipBLASLt tuning saved to {tuning_path.name}")
            return True
        else:
            stderr_tail = (result.stderr or "").strip().split("\n")[-1][:120]
            # Clean up empty/corrupt tuning file left by crashed subprocess
            if tuning_path.exists():
                tuning_path.unlink(missing_ok=True)
            # Write skip marker so we never try again on this GPU
            tuning_skip_marker.write_text(
                f"hipBLASLt tuning crashed (rc={result.returncode})\n"
                f"{stderr_tail}\n"
                "Delete this file to retry tuning.\n"
            )
            print(f"  hipBLASLt tuning crashed (rc={result.returncode}): {stderr_tail}")
            print("  Created .hipblaslt_tuning_skip marker to prevent future attempts.")
            print("  Using default heuristic. Delete .hipblaslt_tuning_skip to retry.")
            return False
    except subprocess.TimeoutExpired:
        print("  hipBLASLt tuning timed out. Using default heuristic.")
        return False
    except Exception as e:
        print(f"  hipBLASLt tuning failed: {e}")
        return False

    tuning_path = PROJECT_DIR / TUNING_FILE_NAME
    already_tuned = tuning_path.exists()

    # Always enable + load if file exists
    if already_tuned:
        tunable.enable(True)
        tunable.set_filename(str(tuning_path))
        tunable.read_file(str(tuning_path))
        print(f"  Loaded hipBLASLt tuning from {tuning_path.name}")
        return True

    # Run tuning in a subprocess to isolate potential crashes (SIGABRT from
    # hipBLASLt trying invalid algorithms on certain shapes).
    print("  Running hipBLASLt auto-tuning in subprocess (one-time)...")

    tuning_script = f'''
import torch, torch.cuda.tunable as tunable, importlib.util, sys
tunable.enable(True)
tunable.tuning_enable(True)
tunable.set_max_tuning_iterations(30)
tunable.set_max_tuning_duration(15)
tunable.set_filename("{tuning_path}")

spec = importlib.util.spec_from_file_location("_ref", "{KB_ACTIVE_DIR / "reference.py"}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
Model = mod.Model
get_inputs = mod.get_inputs
get_init_inputs = getattr(mod, "get_init_inputs", lambda: [])

model = Model(*get_init_inputs()).to("cuda").eval()
inputs = [t.to("cuda") if isinstance(t, torch.Tensor) else t for t in get_inputs()]
for i in range({TUNING_N_WARMUP}):
    with torch.no_grad():
        model(*inputs)
    torch.cuda.synchronize()
tunable.write_file("{tuning_path}")
print("TUNING_OK")
'''
    try:
        env = dict(os.environ)
        result = subprocess.run(
            [sys.executable, "-c", tuning_script],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        if result.returncode == 0 and "TUNING_OK" in result.stdout:
            # Load the tuning results in this process
            tunable.enable(True)
            tunable.set_filename(str(tuning_path))
            tunable.read_file(str(tuning_path))
            print(f"  hipBLASLt tuning saved to {tuning_path.name}")
            return True
        else:
            stderr_tail = (result.stderr or "")[-200:]
            print(f"  hipBLASLt tuning subprocess failed (rc={result.returncode})")
            if stderr_tail:
                print(f"    {stderr_tail}")
            print("  Continuing without tuning (using default heuristic).")
            return False
    except subprocess.TimeoutExpired:
        print("  hipBLASLt tuning subprocess timed out. Continuing without tuning.")
        return False
    except Exception as e:
        print(f"  hipBLASLt tuning failed (non-fatal): {e}")
        return False

    tuning_path = PROJECT_DIR / TUNING_FILE_NAME
    already_tuned = tuning_path.exists()

    # Always enable + load if file exists
    if already_tuned:
        tunable.enable(True)
        tunable.set_filename(str(tuning_path))
        tunable.read_file(str(tuning_path))
        print(f"  Loaded hipBLASLt tuning from {tuning_path.name}")
        return True

    # Otherwise, run tuning warmup
    print("  Running hipBLASLt auto-tuning (one-time)...")
    tunable.enable(True)
    tunable.tuning_enable(True)
    tunable.set_max_tuning_iterations(100)
    tunable.set_max_tuning_duration(30)
    tunable.set_filename(str(tuning_path))

    try:
        inputs = get_inputs_fn()
        inputs_dev = [
            inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in inputs
        ]
        for _ in range(TUNING_N_WARMUP):
            with torch.no_grad():
                model_ref(*inputs_dev)
            torch.cuda.synchronize()
        tunable.write_file(str(tuning_path))
        print(f"  hipBLASLt tuning saved to {tuning_path.name}")
        return True
    except Exception as e:
        print(f"  hipBLASLt tuning failed (non-fatal): {e}")
        tunable.enable(False)
        return False


# ---------------------------------------------------------------------------
# Pure GPU kernel latency profiling
# ---------------------------------------------------------------------------


def run_pure_latency(
    model,
    get_inputs_fn: Callable,
    n_warmup: int = 50,
    n_iter: int = 2000,
    device: str = "cuda",
    label: str = "model",
) -> Dict[str, Any]:
    """Measure pure GPU kernel latency with per-call CUDA event timing.

    Unlike ``run_performance`` which includes Python dispatch overhead in
    the loop, this function pre-creates all inputs and events, then runs
    the tightest possible timing loop:

        start.record()
        model(*inputs)
        end.record()
        sync()

    Returns min/p50/p90/p99/mean/std in **microseconds**.
    """
    import torch

    result: Dict[str, Any] = {
        "label": label,
        "min_us": 0.0,
        "p50_us": 0.0,
        "p90_us": 0.0,
        "p99_us": 0.0,
        "mean_us": 0.0,
        "std_us": 0.0,
        "n_iter": n_iter,
    }

    try:
        # Pre-create inputs on device once
        inputs = get_inputs_fn()
        inputs_dev = [
            inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in inputs
        ]

        # Warmup (JIT compilation, cache fill)
        for _ in range(n_warmup):
            with torch.no_grad():
                model(*inputs_dev)
        torch.cuda.synchronize()

        # Timed: per-call CUDA event timing
        times_us: List[float] = []
        for _ in range(n_iter):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                model(*inputs_dev)
            end.record()
            torch.cuda.synchronize()
            times_us.append(start.elapsed_time(end) * 1000.0)  # ms -> us

        times_us.sort()
        n = len(times_us)
        result["min_us"] = times_us[0]
        result["p50_us"] = times_us[n // 2]
        result["p90_us"] = times_us[int(n * 0.90)]
        result["p99_us"] = times_us[int(n * 0.99)]
        result["mean_us"] = sum(times_us) / n
        result["std_us"] = statistics.stdev(times_us) if n > 1 else 0.0
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def _print_latency_table(
    ref_lat: Dict[str, Any],
    kern_lat: Dict[str, Any],
) -> None:
    """Print a side-by-side pure GPU latency comparison table."""
    print()
    print("-" * 65)
    print("Pure GPU Kernel Latency (per-call CUDA event, 2000 samples)")
    print("-" * 65)
    header = f"{'':18s} {'min':>8s} {'p50':>8s} {'p90':>8s} {'p99':>8s} {'mean':>8s} {'std':>8s}"
    print(header)
    for lat in [ref_lat, kern_lat]:
        if "error" in lat:
            print(f"  {lat['label']:16s}  ERROR: {lat['error']}")
            continue
        print(
            f"  {lat['label']:16s}"
            f" {lat['min_us']:7.1f}  {lat['p50_us']:7.1f}"
            f"  {lat['p90_us']:7.1f}  {lat['p99_us']:7.1f}"
            f"  {lat['mean_us']:7.1f}  {lat['std_us']:7.1f}  us"
        )
    # Speedup at p50
    if ref_lat.get("p50_us", 0) > 0 and kern_lat.get("p50_us", 0) > 0:
        sp = ref_lat["p50_us"] / kern_lat["p50_us"]
        print(f"  {'p50 speedup':16s}  {sp:.3f}x")
    print("-" * 65)


# ---------------------------------------------------------------------------
# VRAM monitoring
# ---------------------------------------------------------------------------


def get_vram_usage() -> Dict[str, float]:
    """Get current GPU VRAM usage in MB."""
    import torch

    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "peak_mb": 0}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "reserved_mb": torch.cuda.memory_reserved() / 1e6,
        "peak_mb": torch.cuda.max_memory_allocated() / 1e6,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KernelBench Evaluation -- Correctness + Performance",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: 3 trials, 30 timed runs"
    )
    parser.add_argument(
        "--correctness-only", action="store_true", help="Skip performance benchmarking"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help=f"Correctness trials (default: {DEFAULT_N_CORRECTNESS})",
    )
    parser.add_argument(
        "--n-timed",
        type=int,
        default=None,
        help=f"Timed iterations (default: {DEFAULT_N_TIMED})",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=DEFAULT_N_WARMUP,
        help=f"Warmup iterations (default: {DEFAULT_N_WARMUP})",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help=f"Absolute tolerance (default: {DEFAULT_ATOL})",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help=f"Relative tolerance (default: {DEFAULT_RTOL})",
    )
    parser.add_argument(
        "--skip-stability", action="store_true", help="Skip stability test"
    )
    parser.add_argument(
        "--skip-determinism", action="store_true", help="Skip determinism test"
    )

    args = parser.parse_args()
    n_trials = args.n_trials or (3 if args.quick else DEFAULT_N_CORRECTNESS)
    n_timed = args.n_timed or (30 if args.quick else DEFAULT_N_TIMED)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No CUDA GPU detected. Results on CPU are not meaningful.")

    meta = load_metadata()
    uid = meta.get("uid", "unknown")
    name = meta.get("name", "unknown")
    level = meta.get("level", "?")

    print("=" * 65)
    print(f"KernelBench Evaluation: {uid} -- {name}")
    print(f"Level: {level} | Device: {device}")
    print("=" * 65)
    print()

    # ---- Load reference ----
    print("Loading reference model...")
    Model, ref_get_inputs, ref_get_init_inputs = load_reference()
    ref_init_args = ref_get_init_inputs()
    model_ref = Model(*ref_init_args)
    if hasattr(model_ref, "to"):
        model_ref = model_ref.to(device)
    if hasattr(model_ref, "eval"):
        model_ref = model_ref.eval()

    # ---- Load kernel ----
    print("Loading ModelNew from kernel.py...")
    ModelNew, kern_get_inputs, kern_get_init_inputs, problem_meta = load_kernel()

    if kern_get_init_inputs is not None:
        try:
            init_args = kern_get_init_inputs()
        except Exception:
            init_args = ref_init_args
    else:
        init_args = ref_init_args

    try:
        model_new = ModelNew(*init_args)
    except Exception as e:
        print(f"\nFATAL: ModelNew instantiation failed: {e}")
        traceback.print_exc()
        _print_summary("FAIL", 0.0, uid, name, {"correctness": "FAIL"})
        sys.exit(1)

    # Sync weights: copy reference model parameters into ModelNew
    # This is standard KernelBench protocol -- ModelNew must be evaluated
    # with the same weights as Model to ensure correctness comparison is fair.
    import torch

    ref_sd = model_ref.state_dict()
    new_sd = model_new.state_dict()
    if ref_sd and new_sd:
        try:
            # Try strict load first (same parameter names)
            model_new.load_state_dict(ref_sd, strict=False)
            print("  Synced weights from reference model to ModelNew")
        except Exception as e:
            print(f"  WARNING: Could not sync weights: {e}")

    if hasattr(model_new, "to"):
        model_new = model_new.to(device)
    if hasattr(model_new, "eval"):
        model_new = model_new.eval()

    get_inputs_fn = kern_get_inputs or ref_get_inputs

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- hipBLASLt auto-tuning (ROCm) ----
    if device == "cuda":
        _try_enable_hipblaslt_tuning(model_ref, get_inputs_fn, device=device)

    # ---- Stage 1: Correctness ----
    print(
        f"\n--- Stage 1: Correctness ({n_trials} trials, atol={args.atol}, rtol={args.rtol}) ---"
    )
    correctness = run_correctness(
        model_new,
        model_ref,
        get_inputs_fn,
        n_trials=n_trials,
        atol=args.atol,
        rtol=args.rtol,
        device=device,
    )
    status = correctness["correctness"]
    print(
        f"  Trials passed: {correctness['trials_passed']}/{correctness['trials_total']}"
    )
    if status == "PASS":
        print(f"  Worst max_abs_error: {correctness['worst_max_abs_error']:.6e}")
        print(f"  Worst mean_abs_error: {correctness['worst_mean_abs_error']:.6e}")
    else:
        for d in correctness["details"]:
            if d["status"] != "PASS":
                print(f"  Trial {d['trial']}: {d['status']} -- {d.get('reason', '?')}")
    print(f"  Result: {status}")

    # ---- Stage 2: Stability ----
    stability = {"stability": "SKIP"}
    if not args.skip_stability and status == "PASS":
        print("\n--- Stage 2: Numerical Stability ---")
        stability = run_stability(model_new, get_inputs_fn, device=device)
        print(f"  Result: {stability['stability']}")
        for d in stability.get("details", []):
            print(f"  {d}")

    # ---- Stage 3: Determinism ----
    determinism = {"determinism": "SKIP"}
    if not args.skip_determinism and status == "PASS":
        print("\n--- Stage 3: Determinism ---")
        determinism = run_determinism(model_new, get_inputs_fn, device=device)
        print(f"  Result: {determinism['determinism']}")
        for d in determinism.get("details", []):
            print(f"  {d}")

    # ---- Stage 4: Performance ----
    perf: Dict[str, Any] = {"speedup": 0.0}
    if not args.correctness_only and status == "PASS":
        print(
            f"\n--- Stage 4: Performance ({args.n_warmup} warmup + {n_timed} timed) ---"
        )
        perf = run_performance(
            model_new,
            model_ref,
            get_inputs_fn,
            n_warmup=args.n_warmup,
            n_timed=n_timed,
            device=device,
        )
        print(f"  Reference time: {perf['reference_time_ms']:.4f} ms")
        print(f"  Kernel time:    {perf['kernel_time_ms']:.4f} ms")
        print(f"  Speedup:        {perf['speedup']:.3f}x")
        if "error" in perf:
            print(f"  Error:          {perf['error']}")
        if perf.get("kernel_times"):
            kt = perf["kernel_times"]
            print(
                f"  Kernel stats:   median={statistics.median(kt):.4f}ms, "
                f"std={statistics.stdev(kt) if len(kt) > 1 else 0:.4f}ms, "
                f"min={min(kt):.4f}ms, max={max(kt):.4f}ms"
            )
    elif status != "PASS":
        print("\n--- Stage 4: Performance SKIPPED (correctness failed) ---")

    # ---- Stage 5: Pure GPU Kernel Latency ----
    ref_lat: Dict[str, Any] = {}
    kern_lat: Dict[str, Any] = {}
    if not args.correctness_only and status == "PASS" and device == "cuda":
        print(f"\n--- Stage 5: Pure GPU Kernel Latency (2000 per-call samples) ---")
        ref_lat = run_pure_latency(
            model_ref,
            get_inputs_fn,
            n_warmup=50,
            n_iter=2000,
            device=device,
            label="reference",
        )
        kern_lat = run_pure_latency(
            model_new,
            get_inputs_fn,
            n_warmup=50,
            n_iter=2000,
            device=device,
            label="kernel",
        )
        _print_latency_table(ref_lat, kern_lat)

    vram = get_vram_usage() if device == "cuda" else {}

    _print_summary(
        status,
        perf.get("speedup", 0.0),
        uid,
        name,
        {
            **correctness,
            **stability,
            **determinism,
            **perf,
            **vram,
            **{f"ref_{k}": v for k, v in ref_lat.items() if k != "label"},
            **{f"kern_{k}": v for k, v in kern_lat.items() if k != "label"},
        },
    )
    _save_results(uid, correctness, stability, determinism, perf, vram, meta)


def _print_summary(
    correctness_status: str,
    speedup: float,
    uid: str,
    name: str,
    data: Dict[str, Any],
) -> None:
    """Print greppable summary for agent log parsing."""
    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"problem: {uid}")
    print(f"name: {name}")
    print(f"correctness: {correctness_status}")
    print(f"speedup: {speedup:.3f}x")
    print(f"kernel_time_ms: {data.get('kernel_time_ms', 0):.4f}")
    print(f"reference_time_ms: {data.get('reference_time_ms', 0):.4f}")
    print(f"stability: {data.get('stability', 'SKIP')}")
    print(f"determinism: {data.get('determinism', 'SKIP')}")
    print(f"peak_vram_mb: {data.get('peak_mb', 0):.1f}")
    print(f"worst_max_abs_error: {data.get('worst_max_abs_error', 0):.6e}")
    # Pure GPU latency (if available)
    ref_p50 = data.get("ref_p50_us", 0)
    kern_p50 = data.get("kern_p50_us", 0)
    if ref_p50 > 0 and kern_p50 > 0:
        print(f"gpu_latency_ref_p50_us: {ref_p50:.1f}")
        print(f"gpu_latency_kern_p50_us: {kern_p50:.1f}")
        print(f"gpu_latency_speedup: {ref_p50 / kern_p50:.3f}x")
    for threshold in [1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0]:
        passes = correctness_status == "PASS" and speedup >= threshold
        print(f"fast_{threshold}: {'PASS' if passes else 'FAIL'}")
    print("=" * 65)


def _save_results(uid, correctness, stability, determinism, perf, vram, meta):
    """Append results to workspace/kb_active/results.json."""
    results_path = KB_ACTIVE_DIR / "results.json"
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "uid": uid,
        "correctness": correctness.get("correctness", "FAIL"),
        "speedup": perf.get("speedup", 0.0),
        "kernel_time_ms": perf.get("kernel_time_ms", 0.0),
        "reference_time_ms": perf.get("reference_time_ms", 0.0),
        "worst_max_abs_error": correctness.get("worst_max_abs_error", 0.0),
        "stability": stability.get("stability", "SKIP"),
        "determinism": determinism.get("determinism", "SKIP"),
        "peak_vram_mb": vram.get("peak_mb", 0.0),
    }
    history = []
    if results_path.exists():
        try:
            history = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []
    history.append(entry)
    results_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
