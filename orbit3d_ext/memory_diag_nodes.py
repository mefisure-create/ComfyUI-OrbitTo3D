"""Orbit3D Patch 3.4: diagnostic CPU heap trim node.

Purpose:
- distinguish live anonymous allocations from freed allocations retained by glibc
- do NOT unload ComfyUI models
- do NOT touch CUDA caches/model management
"""

from __future__ import annotations

import ctypes
import gc
import os
from pathlib import Path


_GIB = 1024 ** 3


def _read_status() -> dict[str, int]:
    out = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            parts = val.strip().split()
            if parts and parts[0].isdigit():
                # /proc/self/status memory values are kB.
                out[key] = int(parts[0]) * 1024
    except Exception:
        pass
    return out


def _read_meminfo() -> dict[str, int]:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            parts = val.strip().split()
            if parts and parts[0].isdigit():
                out[key] = int(parts[0]) * 1024
    except Exception:
        pass
    return out


def _gib(v: int | None) -> str:
    if v is None:
        return "n/a"
    return f"{v / _GIB:.2f} GiB"


def _malloc_trim() -> tuple[bool, str]:
    """Call glibc malloc_trim(0) if available."""
    try:
        libc = ctypes.CDLL(None)
        fn = getattr(libc, "malloc_trim", None)
        if fn is None:
            return False, "malloc_trim unavailable in process libc"
        fn.argtypes = [ctypes.c_size_t]
        fn.restype = ctypes.c_int
        rc = int(fn(0))
        return bool(rc), f"malloc_trim rc={rc}"
    except Exception as exc:
        return False, f"malloc_trim error: {type(exc).__name__}: {exc}"


class Orbit3DCPUMemoryTrimDiagnostic:
    """Diagnostic only: Python GC + glibc malloc_trim(0)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "after_report": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "tooltip": (
                            "Connect Generic Cache Ref Report.report here so "
                            "this node runs after the Marigold loop completes."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Diagnostics"
    OUTPUT_NODE = True

    def run(self, after_report):
        before = _read_status()
        mem_before = _read_meminfo()

        collected = gc.collect()

        after_gc = _read_status()
        mem_after_gc = _read_meminfo()

        trim_effective, trim_status = _malloc_trim()

        after_trim = _read_status()
        mem_after_trim = _read_meminfo()

        rss0 = before.get("VmRSS")
        rss1 = after_gc.get("VmRSS")
        rss2 = after_trim.get("VmRSS")

        anon0 = before.get("RssAnon")
        anon2 = after_trim.get("RssAnon")

        def delta(a, b):
            if a is None or b is None:
                return "n/a"
            return f"{(b - a) / _GIB:+.2f} GiB"

        lines = [
            "Orbit3D CPU Memory Trim Diagnostic",
            f"PID: {os.getpid()}",
            f"gc.collect(): {collected} objects",
            trim_status,
            "",
            f"VmRSS before:       {_gib(rss0)}",
            f"VmRSS after GC:     {_gib(rss1)}   delta {delta(rss0, rss1)}",
            f"VmRSS after trim:   {_gib(rss2)}   delta {delta(rss0, rss2)}",
            f"RssAnon before:     {_gib(anon0)}",
            f"RssAnon after trim: {_gib(anon2)}   delta {delta(anon0, anon2)}",
            "",
            f"MemAvailable before:     {_gib(mem_before.get('MemAvailable'))}",
            f"MemAvailable after GC:   {_gib(mem_after_gc.get('MemAvailable'))}",
            f"MemAvailable after trim: {_gib(mem_after_trim.get('MemAvailable'))}",
            "",
            "Interpretation:",
            "- Large RSS drop after GC: live Python reference cycles were the main issue.",
            "- Large additional drop after malloc_trim: freed heap arenas were retained by glibc.",
            "- Little/no drop: most anonymous memory is still genuinely live.",
            "",
            "No ComfyUI model unload, CUDA cache clear, or reconstruction state change was requested.",
        ]

        report = "\n".join(lines)
        print("\n[Orbit3D CPU Memory Trim]\n" + report + "\n", flush=True)
        return (report,)


NODE_CLASS_MAPPINGS = {
    "Orbit3DCPUMemoryTrimDiagnostic": Orbit3DCPUMemoryTrimDiagnostic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Orbit3DCPUMemoryTrimDiagnostic": "Orbit3D - CPU Memory Trim Diagnostic",
}
