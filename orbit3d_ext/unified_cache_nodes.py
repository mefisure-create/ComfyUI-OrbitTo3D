"""Orbit3D unified cache/session utilities."""
from __future__ import annotations

import ctypes
import datetime as _dt
import gc
import json
from pathlib import Path
import shutil
import uuid

from safetensors import safe_open

_GIB = 1024 ** 3
_CACHE_VERSION = 1

def _default_cache_root() -> Path:
    try:
        import folder_paths
        return Path(folder_paths.get_temp_directory()) / "orbit3d_cache"
    except Exception:
        return Path("/home/ai/ComfyUI/temp/orbit3d_cache")

def _resolve_cache_root(cache_root: str) -> Path:
    value = str(cache_root or "").strip()
    return Path(value).expanduser().resolve() if value else _default_cache_root().resolve()

# ORBIT3D PERSISTENT SNAPSHOT V1
# A synchronous snapshot is taken at Unified Cache Prepare, after every view has
# been spilled and validated and before Cleanup Cache Session can be scheduled.
def _orbit3d_snapshot_persistent(cache_session: dict) -> None:
    root, sid = _validate_session(cache_session)
    disk = json.loads((root / "session.json").read_text(encoding="utf-8"))
    if disk.get("storage_mode", "temporary") != "persistent":
        return

    target_parent = Path(str(disk.get("persistent_path", ""))).expanduser().resolve()
    if not str(disk.get("persistent_path", "")).strip():
        raise ValueError("Persistent storage requires a nonempty directory")
    destination = target_parent / sid
    if destination.exists():
        raise FileExistsError(f"Orbit3D persistent archive already exists: {destination}")
    target_parent.mkdir(parents=True, exist_ok=True)
    # Avoid recursion and the temp-cleanup tree. A persistent root cannot be
    # inside the temporary session, nor may it contain that session.
    if target_parent == root or root in target_parent.parents or target_parent in root.parents:
        raise ValueError("Persistent path must be separate from the temporary cache tree")

    source_files = [p for p in root.rglob("*") if p.is_file()]
    source_bytes = sum(p.stat().st_size for p in source_files)
    free_bytes = shutil.disk_usage(target_parent).free
    if free_bytes < source_bytes:
        raise OSError(f"Not enough free space for persistent cache: need {source_bytes / _GIB:.2f} GiB, free {free_bytes / _GIB:.2f} GiB")

    staging = target_parent / f".{sid}.incomplete_{uuid.uuid4().hex[:8]}"
    print(f"[Orbit3D Cache] persistent snapshot BEGIN: {root} -> {destination}; {source_bytes/_GIB:.2f} GiB", flush=True)
    try:
        shutil.copytree(root, staging, symlinks=False)

        # Manifests contain absolute paths. Rebase *only* paths pointing inside
        # this session; paths outside the session are left untouched.
        src_prefix = str(root)
        dst_prefix = str(destination)
        def rebase(value):
            if isinstance(value, str):
                if value == src_prefix:
                    return dst_prefix
                if value.startswith(src_prefix + "/"):
                    return dst_prefix + value[len(src_prefix):]
                return value
            if isinstance(value, list):
                return [rebase(x) for x in value]
            if isinstance(value, dict):
                return {k: rebase(v) for k, v in value.items()}
            return value

        for jpath in staging.rglob("*.json"):
            try:
                obj = json.loads(jpath.read_text(encoding="utf-8"))
            except (UnicodeError, json.JSONDecodeError):
                continue
            updated = rebase(obj)
            if updated != obj:
                tmp = jpath.with_name(jpath.name + ".rewrite")
                tmp.write_text(json.dumps(updated, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
                tmp.replace(jpath)

        # Count and size checks after copy. JSON metadata can grow/shrink due to
        # path rebasing, so only require identical byte sizes for non-JSON files.
        copied_files = [p for p in staging.rglob("*") if p.is_file()]
        src_rel = {p.relative_to(root) for p in source_files}
        dst_rel = {p.relative_to(staging) for p in copied_files}
        if src_rel != dst_rel:
            raise RuntimeError("Persistent cache file-set verification failed")
        for rel in src_rel:
            if rel.suffix != ".json" and (root / rel).stat().st_size != (staging / rel).stat().st_size:
                raise RuntimeError(f"Persistent cache size mismatch: {rel}")
        mirrored_session = json.loads((staging / "session.json").read_text(encoding="utf-8"))
        if mirrored_session.get("root") != str(destination) or mirrored_session.get("session_id") != sid:
            raise RuntimeError("Persistent cache session metadata was not rebased")

        staging.rename(destination)  # atomic publish on the destination filesystem
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"[Orbit3D Cache] persistent snapshot COMPLETE: {destination}; files={len(src_rel)}", flush=True)

def _read_kb_file(path: str) -> dict[str, int]:
    out = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            parts = val.strip().split()
            if parts and parts[0].isdigit():
                out[key] = int(parts[0]) * 1024
    except Exception:
        pass
    return out

def _gib(v):
    return "n/a" if v is None else f"{v/_GIB:.2f} GiB"

def _malloc_trim():
    try:
        libc = ctypes.CDLL(None)
        fn = getattr(libc, "malloc_trim", None)
        if fn is None:
            return False, "malloc_trim unavailable"
        fn.argtypes = [ctypes.c_size_t]
        fn.restype = ctypes.c_int
        rc = int(fn(0))
        return bool(rc), f"malloc_trim rc={rc}"
    except Exception as exc:
        return False, f"malloc_trim error: {type(exc).__name__}: {exc}"

def _validate_session(cache_session: dict) -> tuple[Path, str]:
    if not isinstance(cache_session, dict):
        raise TypeError("cache_session must come from Orbit3D - Cache Session")
    if cache_session.get("orbit3d_cache_type") != "orbit3d_session":
        raise ValueError("Expected a unified Orbit3D cache session")
    session_id = str(cache_session.get("session_id", "")).strip()
    root = Path(str(cache_session.get("root", ""))).resolve()
    session_json = root / "session.json"
    if not session_id or not session_json.is_file():
        raise ValueError("Invalid Orbit3D cache session")
    disk = json.loads(session_json.read_text(encoding="utf-8"))
    if disk.get("orbit3d_cache_type") != "orbit3d_session":
        raise ValueError("session.json is not a unified Orbit3D session")
    if str(disk.get("session_id", "")) != session_id:
        raise ValueError("Orbit3D cache session_id mismatch")
    if Path(str(disk.get("root", ""))).resolve() != root:
        raise ValueError("Orbit3D cache root mismatch")
    return root, session_id

class Orbit3DCacheSession:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cache_root": ("STRING", {
                "default": "", "multiline": False,
                "tooltip": "Empty = ComfyUI temp/orbit3d_cache. Temporary working cache is unchanged.",
            }),
            "storage_mode": (["temporary", "persistent"], {"default": "temporary"}),
            "persistent_path": ("STRING", {
                "default": "/home/ai/orbit3d-cache", "multiline": False,
                "tooltip": "Persistent mode: save a separate copy here when Unified Cache Prepare runs.",
            }),
        }}
    RETURN_TYPES = ("ORBIT3D_CACHE_SESSION", "STRING")
    RETURN_NAMES = ("cache_session", "session_path")
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
    def run(self, cache_root="", storage_mode="temporary", persistent_path="/home/ai/orbit3d-cache"):
        if storage_mode not in ("temporary", "persistent"):
            raise ValueError(f"Unknown cache storage mode: {storage_mode!r}")
        if storage_mode == "persistent":
            if not str(persistent_path).strip():
                raise ValueError("persistent_path cannot be empty")
            temp_root = _resolve_cache_root(cache_root)
            permanent = Path(persistent_path).expanduser().resolve()
            if permanent == temp_root or permanent in temp_root.parents or temp_root in permanent.parents:
                raise ValueError("Permanent directory must be separate from the temporary cache root")
        parent = _resolve_cache_root(cache_root)
        parent.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sid = f"orbit3d_{stamp}_{uuid.uuid4().hex[:8]}"
        root = parent / sid
        root.mkdir(parents=True, exist_ok=False)
        session = {
            "orbit3d_cache_type": "orbit3d_session",
            "version": _CACHE_VERSION,
            "session_id": sid,
            "root": str(root),
            "storage_mode": storage_mode,
            "persistent_path": str(Path(persistent_path).expanduser().resolve()) if storage_mode == "persistent" else "",
        }
        (root/"session.json").write_text(json.dumps(session, indent=2, sort_keys=True), encoding="utf-8")
        free = shutil.disk_usage(root).free/_GIB
        print(f"[Orbit3D Cache] session {sid} at {root}; free={free:.2f} GiB", flush=True)
        return session, str(root)

# --- ORBIT3D PATCH 3.6.6 MALLINFO2 DIAGNOSTICS BEGIN ---
class _Orbit3DMallinfo2(ctypes.Structure):
    # glibc malloc.h: struct mallinfo2. All fields are size_t.
    _fields_ = [
        ("arena", ctypes.c_size_t),
        ("ordblks", ctypes.c_size_t),
        ("smblks", ctypes.c_size_t),
        ("hblks", ctypes.c_size_t),
        ("hblkhd", ctypes.c_size_t),
        ("usmblks", ctypes.c_size_t),
        ("fsmblks", ctypes.c_size_t),
        ("uordblks", ctypes.c_size_t),
        ("fordblks", ctypes.c_size_t),
        ("keepcost", ctypes.c_size_t),
    ]


def _orbit3d_mallinfo2_snapshot():
    """Return glibc allocator counters without changing allocator policy."""
    try:
        libc = ctypes.CDLL(None)
        fn = getattr(libc, "mallinfo2", None)
        if fn is None:
            return {"available": False, "error": "mallinfo2 unavailable"}
        fn.argtypes = []
        fn.restype = _Orbit3DMallinfo2
        mi = fn()
        return {
            "available": True,
            "arena": int(mi.arena),
            "ordblks": int(mi.ordblks),
            "smblks": int(mi.smblks),
            "hblks": int(mi.hblks),
            "hblkhd": int(mi.hblkhd),
            "fsmblks": int(mi.fsmblks),
            "uordblks": int(mi.uordblks),
            "fordblks": int(mi.fordblks),
            "keepcost": int(mi.keepcost),
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _orbit3d_mallinfo2_report(label, snap):
    if not snap.get("available"):
        return f"mallinfo2 {label}: unavailable ({snap.get('error', 'unknown error')})"
    # arena ~= non-mmapped heap reserved by glibc. uordblks/fordblks split its
    # currently allocated/free chunks. hblkhd is the bytes in mmap allocations.
    return (
        f"mallinfo2 {label}: "
        f"arena={_gib(snap['arena'])}; "
        f"used(uordblks)={_gib(snap['uordblks'])}; "
        f"free(fordblks)={_gib(snap['fordblks'])}; "
        f"top(keepcost)={_gib(snap['keepcost'])}; "
        f"mmap(hblkhd)={_gib(snap['hblkhd'])}; "
        f"free_chunks(ordblks)={snap['ordblks']}; "
        f"mmap_regions(hblks)={snap['hblks']}; "
        f"fastbin_free(fsmblks)={_gib(snap['fsmblks'])}"
    )
# --- ORBIT3D PATCH 3.6.6 MALLINFO2 DIAGNOSTICS END ---

class Orbit3DReleaseCPUMemory:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "after": ("STRING", {"forceInput": True, "multiline": True}),
            "python_gc": ("BOOLEAN", {"default": True}),
            "malloc_trim": ("BOOLEAN", {"default": True}),
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"
    def run(self, after, python_gc=True, malloc_trim=True):
        # IMPORTANT: diagnostics only. Cleanup order is deliberately unchanged:
        #   snapshot -> gc.collect() -> snapshot -> malloc_trim(0) -> snapshot.
        before = _read_kb_file("/proc/self/status")
        mem0 = _read_kb_file("/proc/meminfo")
        mi0 = _orbit3d_mallinfo2_snapshot()

        collected = gc.collect() if python_gc else None
        after_gc = _read_kb_file("/proc/self/status")
        mi1 = _orbit3d_mallinfo2_snapshot()

        trim_status = "malloc_trim disabled"
        if malloc_trim:
            _, trim_status = _malloc_trim()

        final = _read_kb_file("/proc/self/status")
        mem1 = _read_kb_file("/proc/meminfo")
        mi2 = _orbit3d_mallinfo2_snapshot()

        rss0, rss1, rss2 = before.get("VmRSS"), after_gc.get("VmRSS"), final.get("VmRSS")
        anon0, anon1, anon2 = before.get("RssAnon"), after_gc.get("RssAnon"), final.get("RssAnon")

        def delta(a, b):
            return "n/a" if a is None or b is None else f"{(b-a)/_GIB:+.2f} GiB"

        report = "\n".join([
            "Orbit3D Release CPU Memory",
            f"gc.collect(): {collected if collected is not None else 'disabled'}",
            trim_status,
            f"VmRSS: {_gib(rss0)} -> {_gib(rss1)} after GC -> {_gib(rss2)} final ({delta(rss0,rss2)})",
            f"RssAnon: {_gib(anon0)} -> {_gib(anon1)} after GC -> {_gib(anon2)} final ({delta(anon0,anon2)})",
            f"MemAvailable: {_gib(mem0.get('MemAvailable'))} -> {_gib(mem1.get('MemAvailable'))}",
            _orbit3d_mallinfo2_report("before GC", mi0),
            _orbit3d_mallinfo2_report("after GC", mi1),
            _orbit3d_mallinfo2_report("after trim", mi2),
            "mallinfo2 describes glibc malloc state only; it does not account for every Python/PyTorch mmap or external allocator.",
            "No model unload or CUDA cache operation was requested.",
        ])
        print("\n[Orbit3D Release CPU Memory]\n" + report + "\n", flush=True)
        return (report,)

def _decode_generic_dict(tree: dict) -> dict[str, dict]:
    if not isinstance(tree, dict) or tree.get("kind") != "bundle":
        raise ValueError("Generic spill item tree is not a bundle")
    vals = list(tree.get("values") or [])
    if len(vals) != 1 or not isinstance(vals[0], dict) or vals[0].get("kind") != "dict":
        raise ValueError("Generic MoGe spill must contain exactly one MOGE_GEOMETRY dict")
    out = {}
    for pair in vals[0].get("items") or []:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        k, v = pair
        if isinstance(k, dict) and k.get("kind") == "scalar" and isinstance(k.get("value"), str):
            out[k["value"]] = v
    return out

class Orbit3DGenericToMoGeLazyCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cache_manifest": ("ORBIT3D_GENERIC_CACHE_MANIFEST",),
            "after_release": ("STRING", {"forceInput": True, "multiline": True}),
        }}
    RETURN_TYPES = ("ORBIT3D_MOGE_CACHE", "STRING")
    RETURN_NAMES = ("moge_cache", "report")
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"
    def run(self, cache_manifest, after_release):
        if not isinstance(cache_manifest, dict) or cache_manifest.get("orbit3d_cache_type") != "generic_spill_manifest":
            raise TypeError("cache_manifest must come from Generic Cache Ref Report")
        if str(cache_manifest.get("stream", "")).lower() != "moge":
            raise ValueError(f"Expected stream='MoGe', got {cache_manifest.get('stream')!r}")
        if not str(after_release or "").strip():
            raise ValueError("after_release dependency is empty")
        required = ("points","depth","normal","mask","intrinsics")
        items = []
        for i, ref in enumerate(cache_manifest.get("items") or []):
            item_dir = Path(str(ref.get("path",""))).resolve()
            item_json = item_dir/"item.json"
            tensor_file = item_dir/"tensors.safetensors"
            if not item_json.is_file() or not tensor_file.is_file():
                raise FileNotFoundError(f"Incomplete generic MoGe item {i}: {item_dir}")
            payload = json.loads(item_json.read_text(encoding="utf-8"))
            logical = _decode_generic_dict(payload.get("tree"))
            missing = [x for x in required if x not in logical]
            if missing:
                raise ValueError(f"Generic MoGe item {i} missing fields: {missing}")
            schema, keymap = {}, {}
            with safe_open(str(tensor_file), framework="pt", device="cpu") as f:
                disk_keys = set(f.keys())
                for name in required:
                    node = logical[name]
                    if node.get("kind") != "torch":
                        raise TypeError(f"MoGe field {name!r} in item {i} is not torch")
                    key = str(node.get("key",""))
                    if key not in disk_keys:
                        raise KeyError(f"Missing tensor key {key!r} for {name!r} in item {i}")
                    keymap[name] = key
                    schema[name] = {"shape": list(node.get("shape") or []), "dtype": str(node.get("dtype",""))}
            items.append({
                "index": i,
                "path": str(tensor_file),
                "file_size": int(ref.get("file_size") or tensor_file.stat().st_size),
                "tensor_schema": schema,
                "tensor_key_map": keymap,
                "metadata": {},
            })
        if not items:
            raise ValueError("Generic MoGe manifest contains no items")
        out = {
            "orbit3d_cache_type": "moge_cache",
            "version": 1,
            "format": "generic_spill_adapter_v1",
            "session_id": str(cache_manifest.get("session_id","")),
            "root": str(cache_manifest.get("root","")),
            "view_count": len(items),
            "total_file_size": int(cache_manifest.get("total_file_size") or 0),
            "items": items,
            "generic_manifest_path": str(cache_manifest.get("manifest_path","")),
        }
        report = f"Orbit3D generic MoGe cache adapted lazily: V={len(items)}; no tensors repacked"
        print(f"[Orbit3D Cache] {report}", flush=True)
        return out, report

class Orbit3DCleanupCacheSession:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cache_session": ("ORBIT3D_CACHE_SESSION",),
            "after_export": ("STRING", {"forceInput": True, "multiline": True}),
            "delete_cache": ("BOOLEAN", {"default": True}),
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"
    OUTPUT_NODE = True
    def run(self, cache_session, after_export, delete_cache=True):
        root, sid = _validate_session(cache_session)
        marker = str(after_export or "").strip()
        if not marker:
            raise ValueError("Cleanup requires a completed export/report dependency")
        if not sid.startswith("orbit3d_") or root.name != sid:
            raise ValueError(f"Refusing unsafe cleanup target: {root}")
        if not delete_cache:
            report = f"Orbit3D cache kept: {root}; export={marker[:160]}"
            print(f"[Orbit3D Cache] {report}", flush=True)
            return (report,)
        total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        shutil.rmtree(root)
        report = f"Orbit3D cache session deleted: {root}; freed~{total/_GIB:.2f} GiB; export={marker[:160]}"
        print(f"[Orbit3D Cache] {report}", flush=True)
        return (report,)

# --- ORBIT3D PATCH 3.6.12 MV INTEGRITY BEGIN ---
def validate_unified_cache_integrity(view_cache, moge_cache, marigold_cache, expected_views):
    """Fail-closed structural integrity check for the disk-backed Unified cache.

    This deliberately reads only JSON metadata and safetensors headers/keys.
    Tensor payloads are NOT materialized, so the check does not allocate a full
    view or full batch merely to verify completeness.
    """
    import json as _json
    from pathlib import Path as _Path
    from safetensors import safe_open as _safe_open

    expected = int(expected_views)
    if expected < 1:
        raise RuntimeError(f"Orbit3D MV integrity expected_views must be >=1, got {expected}")

    specs = (
        ("view_cache", view_cache, "view_cache"),
        ("moge_cache", moge_cache, "moge_cache"),
        ("marigold_cache", marigold_cache, "marigold_cache"),
    )
    for label, cache, kind in specs:
        if not isinstance(cache, dict):
            raise RuntimeError(f"Orbit3D MV integrity: {label} is missing or is not a dict")
        if cache.get("orbit3d_cache_type") != kind:
            raise RuntimeError(
                f"Orbit3D MV integrity: {label} type={cache.get('orbit3d_cache_type')!r}, "
                f"expected {kind!r}"
            )

    sessions = {
        str(view_cache.get("session_id", "")),
        str(moge_cache.get("session_id", "")),
        str(marigold_cache.get("session_id", "")),
    }
    if "" in sessions or len(sessions) != 1:
        raise RuntimeError(f"Orbit3D MV integrity: cache session mismatch: {sorted(sessions)!r}")
    session_id = next(iter(sessions))

    manifests = {
        str(view_cache.get("generic_manifest_path", "")),
        str(moge_cache.get("generic_manifest_path", "")),
        str(marigold_cache.get("generic_manifest_path", "")),
    }
    if "" in manifests or len(manifests) != 1:
        raise RuntimeError("Orbit3D MV integrity: caches do not reference one common Unified manifest")
    manifest_path = _Path(next(iter(manifests))).resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"Orbit3D MV integrity: Unified manifest missing: {manifest_path}")

    manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("orbit3d_cache_type") != "generic_spill_manifest":
        raise RuntimeError("Orbit3D MV integrity: invalid generic spill manifest type")
    if str(manifest.get("stream", "")).strip().lower() != "unified":
        raise RuntimeError(
            f"Orbit3D MV integrity: expected Unified stream, got {manifest.get('stream')!r}"
        )
    if str(manifest.get("session_id", "")) != session_id:
        raise RuntimeError("Orbit3D MV integrity: manifest/cache session_id mismatch")

    refs = list(manifest.get("items") or [])
    manifest_count = int(manifest.get("item_count", len(refs)))
    if manifest_count != len(refs):
        raise RuntimeError(
            f"Orbit3D MV integrity: manifest item_count={manifest_count}, refs={len(refs)}"
        )

    view_items = list(view_cache.get("items") or [])
    moge_items = list(moge_cache.get("items") or [])
    marigold_items = list(marigold_cache.get("items") or [])
    counts = {
        "manifest": len(refs),
        "view_cache": len(view_items),
        "moge_cache": len(moge_items),
        "marigold_cache": len(marigold_items),
        "view_cache.view_count": int(view_cache.get("view_count", -1)),
        "moge_cache.view_count": int(moge_cache.get("view_count", -1)),
        "marigold_cache.view_count": int(marigold_cache.get("view_count", -1)),
    }
    bad_counts = {k: v for k, v in counts.items() if v != expected}
    if bad_counts:
        raise RuntimeError(
            f"Orbit3D MV integrity: expected {expected} views, count mismatch: {bad_counts}"
        )

    for label, shape in (
        ("image_shape", view_cache.get("image_shape")),
        ("mask_shape", view_cache.get("mask_shape")),
        ("normal_shape", marigold_cache.get("normal_shape")),
        ("depth_shape", marigold_cache.get("depth_shape")),
    ):
        if not isinstance(shape, (list, tuple)) or not shape or int(shape[0]) != expected:
            raise RuntimeError(
                f"Orbit3D MV integrity: {label} must start with V={expected}, got {shape!r}"
            )

    ref_indices = [int(r.get("iteration_index", -1)) for r in refs]
    expected_indices = list(range(expected))
    if ref_indices != expected_indices:
        raise RuntimeError(
            "Orbit3D MV integrity: Unified manifest iteration indices are not exactly "
            f"0..{expected - 1}; got {ref_indices[:20]}"
            + ("..." if len(ref_indices) > 20 else "")
        )

    required_moge = ("points", "depth", "normal", "mask", "intrinsics")
    component_counts = {
        "image": 0,
        "mask": 0,
        "marigold_normal": 0,
        "marigold_depth": 0,
        "moge_points": 0,
        "moge_depth": 0,
        "moge_normal": 0,
        "moge_mask": 0,
        "moge_intrinsics": 0,
    }
    errors = []
    complete_views = 0
    first_moge_schema = None

    for i in range(expected):
        try:
            ref = refs[i]
            vi = view_items[i]
            gi = moge_items[i]
            mi = marigold_items[i]

            for label, item in (("view", vi), ("moge", gi), ("marigold", mi)):
                if int(item.get("index", -1)) != i:
                    raise RuntimeError(f"{label} cache index={item.get('index')!r}, expected {i}")

            paths = {
                str(vi.get("path", "")),
                str(gi.get("path", "")),
                str(mi.get("path", "")),
            }
            if "" in paths or len(paths) != 1:
                raise RuntimeError(f"view {i} cache paths disagree: {sorted(paths)!r}")
            tensor_file = _Path(next(iter(paths))).resolve()
            if not tensor_file.is_file() or tensor_file.stat().st_size <= 0:
                raise RuntimeError(f"view {i} tensor file missing/empty: {tensor_file}")

            item_json = tensor_file.parent / "item.json"
            if not item_json.is_file():
                raise RuntimeError(f"view {i} item.json missing: {item_json}")
            item_meta = _json.loads(item_json.read_text(encoding="utf-8"))
            if item_meta.get("orbit3d_cache_type") != "generic_spill_item":
                raise RuntimeError(f"view {i} invalid spill item type")
            if str(item_meta.get("session_id", "")) != session_id:
                raise RuntimeError(f"view {i} session_id mismatch")
            if str(item_meta.get("stream", "")).strip().lower() != "unified":
                raise RuntimeError(f"view {i} stream is not Unified")
            if int(item_meta.get("iteration_index", -1)) != i:
                raise RuntimeError(
                    f"view {i} item.json iteration_index={item_meta.get('iteration_index')!r}"
                )

            ref_path = _Path(str(ref.get("path", ""))).resolve()
            if ref_path != tensor_file.parent:
                raise RuntimeError(
                    f"view {i} manifest ref path {ref_path} != cache item {tensor_file.parent}"
                )

            ikey = str(vi.get("image_key", ""))
            mask_key = str(vi.get("mask_key", ""))
            nkey = str(mi.get("normal_key", ""))
            dkey = str(mi.get("depth_key", ""))
            keymap = dict(gi.get("tensor_key_map") or {})
            schema = dict(gi.get("tensor_schema") or {})

            missing_moge_meta = [name for name in required_moge if not keymap.get(name) or name not in schema]
            if missing_moge_meta:
                raise RuntimeError(f"view {i} MoGe metadata missing {missing_moge_meta}")

            required_keys = {
                "image": ikey,
                "mask": mask_key,
                "marigold_normal": nkey,
                "marigold_depth": dkey,
                **{f"moge_{name}": str(keymap[name]) for name in required_moge},
            }
            empty = [name for name, key in required_keys.items() if not key]
            if empty:
                raise RuntimeError(f"view {i} empty tensor-key metadata: {empty}")

            with _safe_open(str(tensor_file), framework="pt", device="cpu") as f:
                disk_keys = set(f.keys())
            missing_disk = [name for name, key in required_keys.items() if key not in disk_keys]
            if missing_disk:
                raise RuntimeError(f"view {i} safetensors missing fields: {missing_disk}")

            # MoGe schema consistency is validated from metadata only; no tensor body is loaded.
            current_schema = {
                name: (tuple(schema[name].get("shape") or ()), str(schema[name].get("dtype", "")))
                for name in required_moge
            }
            if first_moge_schema is None:
                first_moge_schema = current_schema
            elif current_schema != first_moge_schema:
                raise RuntimeError(f"view {i} MoGe tensor schema differs from view 0")

            pshape = current_schema["points"][0]
            dshape = current_schema["depth"][0]
            nshape = current_schema["normal"][0]
            mshape = current_schema["mask"][0]
            kshape = current_schema["intrinsics"][0]
            if len(pshape) != 4 or pshape[0] != 1 or pshape[-1] != 3:
                raise RuntimeError(f"view {i} MoGe points shape invalid: {pshape}")
            if len(nshape) != 4 or nshape[0] != 1 or nshape[-1] != 3:
                raise RuntimeError(f"view {i} MoGe normal shape invalid: {nshape}")
            if len(dshape) != 3 or dshape[0] != 1:
                raise RuntimeError(f"view {i} MoGe depth shape invalid: {dshape}")
            if len(mshape) != 3 or mshape[0] != 1:
                raise RuntimeError(f"view {i} MoGe mask shape invalid: {mshape}")
            if kshape != (1, 3, 3):
                raise RuntimeError(f"view {i} MoGe intrinsics shape invalid: {kshape}")
            if tuple(pshape[1:3]) != tuple(dshape[1:3]) or tuple(pshape[1:3]) != tuple(nshape[1:3]) or tuple(pshape[1:3]) != tuple(mshape[1:3]):
                raise RuntimeError(f"view {i} MoGe HxW mismatch between points/depth/normal/mask")

            for name in required_keys:
                component_counts[name] += 1
            complete_views += 1
        except Exception as exc:
            errors.append(f"view {i}: {type(exc).__name__}: {exc}")

    if errors:
        preview = "\n  - ".join(errors[:12])
        suffix = "" if len(errors) <= 12 else f"\n  ... and {len(errors) - 12} more"
        raise RuntimeError(
            f"Orbit3D MV integrity FAILED: complete={complete_views}/{expected}; "
            f"bad_views={len(errors)}\n  - {preview}{suffix}"
        )

    pct = 100.0 * complete_views / expected
    lines = [
        "[Orbit3D MV Integrity]",
        f"expected_views={expected}",
        f"manifest_refs={len(refs)}/{expected} OK",
        f"indices=0..{expected - 1} contiguous OK",
        f"view_cache={len(view_items)}/{expected} OK",
        f"moge_cache={len(moge_items)}/{expected} OK",
        f"marigold_cache={len(marigold_items)}/{expected} OK",
    ]
    for name in (
        "image", "mask", "marigold_normal", "marigold_depth",
        "moge_points", "moge_depth", "moge_normal", "moge_mask", "moge_intrinsics",
    ):
        lines.append(f"{name}={component_counts[name]}/{expected} OK")
    lines.append(f"RESULT={complete_views}/{expected} COMPLETE — {pct:.1f}%")
    report = "; ".join(lines[1:])
    print("\n" + "\n".join(lines) + "\n", flush=True)
    return report
# --- ORBIT3D PATCH 3.6.12 MV INTEGRITY END ---


NODE_CLASS_MAPPINGS = {
    "Orbit3DCacheSession": Orbit3DCacheSession,
    "Orbit3DReleaseCPUMemory": Orbit3DReleaseCPUMemory,
    "Orbit3DGenericToMoGeLazyCache": Orbit3DGenericToMoGeLazyCache,
    "Orbit3DCleanupCacheSession": Orbit3DCleanupCacheSession,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Orbit3DCacheSession": "Orbit3D - Cache Session",
    "Orbit3DReleaseCPUMemory": "Orbit3D - Release CPU Memory",
    "Orbit3DGenericToMoGeLazyCache": "Orbit3D - Generic Cache → MoGe Lazy Cache",
    "Orbit3DCleanupCacheSession": "Orbit3D - Cleanup Cache Session",
}

# --- ORBIT3D PATCH 3.6.1 SINGLE LOOP UNIFIED VIEW CACHE BEGIN ---
import torch as _orbit3d_torch


def _orbit3d_encoded_dict(node):
    if not isinstance(node, dict) or node.get("kind") != "dict":
        raise ValueError("Expected encoded dict in unified cache item")
    out = {}
    for pair in node.get("items") or []:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        k, v = pair
        if (
            isinstance(k, dict)
            and k.get("kind") == "scalar"
            and isinstance(k.get("value"), str)
        ):
            out[k["value"]] = v
    return out


def _orbit3d_unified_item_nodes(ref):
    item_dir = Path(str(ref.get("path", ""))).resolve()
    item_json = item_dir / "item.json"
    tensor_file = item_dir / "tensors.safetensors"
    if not item_json.is_file() or not tensor_file.is_file():
        raise FileNotFoundError(f"Incomplete unified cache item: {item_dir}")

    payload = json.loads(item_json.read_text(encoding="utf-8"))
    tree = payload.get("tree")
    if not isinstance(tree, dict) or tree.get("kind") != "bundle":
        raise ValueError(f"Unified item has invalid bundle tree: {item_dir}")
    values = list(tree.get("values") or [])
    if len(values) != 1:
        raise ValueError(
            f"Unified item must contain exactly one packed payload, got {len(values)}"
        )

    fields = _orbit3d_encoded_dict(values[0])
    required = ("moge_geometry", "normal_map", "depth_map", "image", "mask")
    missing = [k for k in required if k not in fields]
    if missing:
        raise ValueError(f"Unified cache item missing fields {missing}: {item_dir}")

    moge = _orbit3d_encoded_dict(fields["moge_geometry"])
    required_moge = ("points", "depth", "normal", "mask", "intrinsics")
    missing_moge = [k for k in required_moge if k not in moge]
    if missing_moge:
        raise ValueError(
            f"Unified cache item MoGe packet missing {missing_moge}: {item_dir}"
        )

    return item_dir, tensor_file, fields, moge


def _orbit3d_torch_node(node, label):
    if not isinstance(node, dict) or node.get("kind") != "torch":
        raise TypeError(f"{label} is not an encoded torch tensor")
    key = str(node.get("key", ""))
    shape = tuple(int(x) for x in (node.get("shape") or []))
    dtype = str(node.get("dtype", ""))
    if not key or not shape:
        raise ValueError(f"{label} has invalid encoded tensor schema")
    return key, shape, dtype


class Orbit3DDecodeUnifiedCacheFrame:
    """Decode a single view from an existing Unified generic-spill manifest."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cache_manifest": ("ORBIT3D_GENERIC_CACHE_MANIFEST",),
            "frame_index": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
        }}

    RETURN_TYPES = ("MOGE_GEOMETRY", "IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("moge_geometry", "image", "normals", "depth", "mask")
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"

    def run(self, cache_manifest, frame_index):
        if not isinstance(cache_manifest, dict) or cache_manifest.get("orbit3d_cache_type") != "generic_spill_manifest":
            raise ValueError("Expected a generic spill cache_manifest")
        if str(cache_manifest.get("stream", "")).lower() != "unified":
            raise ValueError("Cache manifest must be stream='Unified'")
        refs = list(cache_manifest.get("items") or [])
        i = int(frame_index)
        if i < 0 or i >= len(refs):
            raise IndexError(f"Frame {i} outside Unified cache range 0..{len(refs)-1}")

        item_dir, tensor_file, fields, moge_fields = _orbit3d_unified_item_nodes(refs[i])
        keys = {}
        expected_shapes = {}
        for name, encoded in (
            ("image", fields["image"]),
            ("normals", fields["normal_map"]),
            ("depth", fields["depth_map"]),
            ("mask", fields["mask"]),
        ):
            key, shape, _ = _orbit3d_torch_node(encoded, name)
            keys[name], expected_shapes[name] = key, shape
        for name in ("points", "depth", "normal", "mask", "intrinsics"):
            key, shape, _ = _orbit3d_torch_node(moge_fields[name], "moge." + name)
            keys["moge." + name], expected_shapes["moge." + name] = key, shape

        with safe_open(str(tensor_file), framework="pt", device="cpu") as f:
            present = set(f.keys())
            missing = [name for name, key in keys.items() if key not in present]
            if missing:
                raise KeyError(f"Missing tensors in {item_dir}: {missing}")
            data = {name: f.get_tensor(key) for name, key in keys.items()}

        for name, tensor in data.items():
            if tuple(tensor.shape) != expected_shapes[name]:
                raise ValueError(f"{name}: shape {tuple(tensor.shape)} != {expected_shapes[name]}")

        moge = {name: data["moge." + name] for name in ("points", "depth", "normal", "mask", "intrinsics")}
        print(f"[Orbit3D Cache] Decoded Unified frame {i}/{len(refs)-1}: {tensor_file}", flush=True)
        return moge, data["image"], data["normals"], data["depth"], data["mask"]



class Orbit3DUnifiedViewPayload:
    """Pack one synchronized view without copying tensors.

    Generic Spill performs the actual CPU/disk representation change.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "moge_geometry": ("MOGE_GEOMETRY",),
                "normal_map": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("ORBIT3D_UNIFIED_VIEW",)
    RETURN_NAMES = ("view_payload",)
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"

    def run(self, moge_geometry, normal_map, depth_map, image, mask):
        return ({
            "orbit3d_cache_type": "unified_view_payload",
            "version": 1,
            "moge_geometry": moge_geometry,
            "normal_map": normal_map,
            "depth_map": depth_map,
            "image": image,
            "mask": mask,
        },)


class _Orbit3DMarigoldReader:
    def __init__(self, cache):
        self.items = list(cache["items"])
        self._index = None
        self._fields = {}

    def get(self, index, field):
        index = int(index)
        if index < 0:
            index += len(self.items)
        if not 0 <= index < len(self.items):
            raise IndexError(index)

        if self._index != index:
            self._fields.clear()
            self._index = index

        if field not in self._fields:
            item = self.items[index]
            path = Path(item["path"])
            key = item[f"{field}_key"]
            if not path.is_file():
                raise FileNotFoundError(path)
            with safe_open(str(path), framework="pt", device="cpu") as f:
                if key not in f.keys():
                    raise KeyError(
                        f"Unified Marigold view {index}: key {key!r} missing in {path}"
                    )
                self._fields[field] = f.get_tensor(key)
        return self._fields[field]


class _Orbit3DLazyDepthStack:
    """Lazy logical stack derived from depth IMAGE[..., 0].

    Supports exactly the tensor operations used by Orbit's memory-lean
    attach_geometry_confidence path: detach/cpu/float/contiguous, unsqueeze,
    shape/ndim and explicit single-view indexing.
    """

    def __init__(self, reader, shape, view_axis, cast_dtype=None):
        self._reader = reader
        self._shape = tuple(int(x) for x in shape)
        self._view_axis = int(view_axis)
        self._cast_dtype = cast_dtype

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def device(self):
        return _orbit3d_torch.device("cpu")

    @property
    def dtype(self):
        return self._cast_dtype or _orbit3d_torch.float32

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def float(self):
        return _Orbit3DLazyDepthStack(
            self._reader, self._shape, self._view_axis, _orbit3d_torch.float32
        )

    def unsqueeze(self, dim):
        dim = int(dim)
        if dim < 0:
            dim += len(self._shape) + 1
        if not 0 <= dim <= len(self._shape):
            raise IndexError(dim)
        shape = list(self._shape)
        shape.insert(dim, 1)
        view_axis = self._view_axis + (1 if dim <= self._view_axis else 0)
        return _Orbit3DLazyDepthStack(
            self._reader, shape, view_axis, self._cast_dtype
        )

    @staticmethod
    def _expand_key(key, ndim):
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            p = key.index(Ellipsis)
            explicit = len(key) - 1
            fill = max(0, ndim - explicit)
            key = key[:p] + (slice(None),) * fill + key[p + 1:]
        if len(key) < ndim:
            key = key + (slice(None),) * (ndim - len(key))
        return key

    def __getitem__(self, key):
        key = self._expand_key(key, self.ndim)
        if len(key) != self.ndim:
            raise RuntimeError(
                f"Lazy Marigold depth expected {self.ndim} indices, got {key!r}"
            )

        view_key = key[self._view_axis]
        if not isinstance(view_key, int):
            raise RuntimeError(
                "Lazy Marigold depth supports one explicit view at a time; "
                f"got {view_key!r}"
            )
        view = view_key
        if view < 0:
            view += self._shape[self._view_axis]

        t = self._reader.get(view, "depth")
        if t.ndim != 4 or t.shape[0] != 1:
            raise ValueError(
                f"Cached Marigold depth view must be [1,H,W,C], got {tuple(t.shape)}"
            )
        t = t[0, ..., 0]
        if self._cast_dtype is not None and t.dtype != self._cast_dtype:
            t = t.to(self._cast_dtype)

        target_shape = list(self._shape)
        del target_shape[self._view_axis]
        if t.numel() != int(_orbit3d_torch.tensor(target_shape).prod().item()):
            raise RuntimeError(
                f"Lazy depth logical shape {target_shape} does not match "
                f"cached view {tuple(t.shape)}"
            )
        t = t.reshape(target_shape)

        rest = key[:self._view_axis] + key[self._view_axis + 1:]
        return t[rest]


class _Orbit3DLazyMarigoldImageBatch:
    def __init__(self, reader, field, shape):
        self._reader = reader
        self._field = str(field)
        self._shape = tuple(int(x) for x in shape)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def device(self):
        return _orbit3d_torch.device("cpu")

    @property
    def dtype(self):
        return _orbit3d_torch.float32

    @staticmethod
    def _expand_key(key, ndim):
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            p = key.index(Ellipsis)
            explicit = len(key) - 1
            fill = max(0, ndim - explicit)
            key = key[:p] + (slice(None),) * fill + key[p + 1:]
        if len(key) < ndim:
            key = key + (slice(None),) * (ndim - len(key))
        return key

    def __getitem__(self, key):
        key = self._expand_key(key, self.ndim)
        if len(key) != self.ndim:
            raise RuntimeError(
                f"Lazy Marigold IMAGE expected {self.ndim} indices, got {key!r}"
            )

        view_key = key[0]
        if isinstance(view_key, int):
            view = view_key
            if view < 0:
                view += self._shape[0]
            t = self._reader.get(view, self._field)
            if t.ndim != 4 or t.shape[0] != 1:
                raise ValueError(
                    f"Cached Marigold {self._field} view must be [1,H,W,C], "
                    f"got {tuple(t.shape)}"
                )
            return t[0][key[1:]]

        # The memory-lean Orbit Marigold path accesses depth_maps[..., 0].
        if (
            self._field == "depth"
            and isinstance(view_key, slice)
            and view_key == slice(None)
            and all(isinstance(k, slice) and k == slice(None) for k in key[1:-1])
            and key[-1] == 0
        ):
            return _Orbit3DLazyDepthStack(
                self._reader,
                (self._shape[0], self._shape[1], self._shape[2]),
                view_axis=0,
                cast_dtype=None,
            )

        raise RuntimeError(
            "Unified Marigold cache was asked to materialize multiple views at "
            f"once ({key!r}). Patch 3.6 requires Orbit's memory-lean "
            "per-view Marigold path."
        )


def _orbit3d_validate_marigold_cache(cache):
    if not isinstance(cache, dict):
        raise TypeError("marigold_cache must come from Unified Cache Prepare")
    if cache.get("orbit3d_cache_type") != "marigold_cache":
        raise ValueError("Invalid ORBIT3D_MARIGOLD_CACHE handle")
    items = list(cache.get("items") or [])
    if not items:
        raise ValueError("ORBIT3D_MARIGOLD_CACHE is empty")
    return items


def make_marigold_lazy_tensors(cache):
    items = _orbit3d_validate_marigold_cache(cache)
    reader = _Orbit3DMarigoldReader(cache)
    nshape = tuple(cache["normal_shape"])
    dshape = tuple(cache["depth_shape"])
    normals = _Orbit3DLazyMarigoldImageBatch(reader, "normal", nshape)
    depths = _Orbit3DLazyMarigoldImageBatch(reader, "depth", dshape)
    return normals, depths



# --- ORBIT3D PATCH 3.6.3 LAZY MULTIVIEW IMAGES/MASKS BEGIN ---
class _Orbit3DUnifiedViewReader:
    """Read one clay render/mask pair from the Unified spill at a time."""

    def __init__(self, cache):
        self.items = list(cache["items"])
        self._index = None
        self._fields = {}

    def get(self, index, field):
        index = int(index)
        if index < 0:
            index += len(self.items)
        if not 0 <= index < len(self.items):
            raise IndexError(index)

        if self._index != index:
            self._fields.clear()
            self._index = index

        if field not in self._fields:
            item = self.items[index]
            path = Path(item["path"])
            key = item[f"{field}_key"]
            if not path.is_file():
                raise FileNotFoundError(path)
            with safe_open(str(path), framework="pt", device="cpu") as f:
                if key not in f.keys():
                    raise KeyError(
                        f"Unified view {index}: key {key!r} missing in {path}"
                    )
                self._fields[field] = f.get_tensor(key)
        return self._fields[field]


class _Orbit3DLazyViewImageBatch:
    """Logical Comfy IMAGE [V,H,W,C], loading only images[i]."""

    _orbit3d_lazy_view_batch = True

    def __init__(self, reader, shape):
        self._reader = reader
        self._shape = tuple(int(x) for x in shape)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def device(self):
        return _orbit3d_torch.device("cpu")

    @property
    def dtype(self):
        return _orbit3d_torch.float32

    def __len__(self):
        return self._shape[0]

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise RuntimeError(
                "Lazy Orbit3D IMAGE supports one explicit view at a time; "
                f"got {key!r}"
            )
        view = key
        if view < 0:
            view += self._shape[0]
        if not 0 <= view < self._shape[0]:
            raise IndexError(view)
        t = self._reader.get(view, "image")
        if t.ndim != 4 or t.shape[0] != 1:
            raise ValueError(
                f"Cached clay IMAGE view must be [1,H,W,C], got {tuple(t.shape)}"
            )
        return t[0]


class _Orbit3DLazyViewMaskBatch:
    """Logical Comfy MASK [V,H,W], loading only masks[i]."""

    _orbit3d_lazy_view_batch = True

    def __init__(self, reader, shape):
        self._reader = reader
        self._shape = tuple(int(x) for x in shape)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def device(self):
        return _orbit3d_torch.device("cpu")

    @property
    def dtype(self):
        return _orbit3d_torch.float32

    def __len__(self):
        return self._shape[0]

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise RuntimeError(
                "Lazy Orbit3D MASK supports one explicit view at a time; "
                f"got {key!r}"
            )
        view = key
        if view < 0:
            view += self._shape[0]
        if not 0 <= view < self._shape[0]:
            raise IndexError(view)
        t = self._reader.get(view, "mask")
        if t.ndim != 3 or t.shape[0] != 1:
            raise ValueError(
                f"Cached MASK view must be [1,H,W], got {tuple(t.shape)}"
            )
        return t[0]


def _orbit3d_validate_view_cache(cache):
    if not isinstance(cache, dict):
        raise TypeError("view_cache must come from Unified Cache Prepare")
    if cache.get("orbit3d_cache_type") != "view_cache":
        raise ValueError("Invalid ORBIT3D_VIEW_CACHE handle")
    items = list(cache.get("items") or [])
    if not items:
        raise ValueError("ORBIT3D_VIEW_CACHE is empty")
    return items


def make_view_lazy_tensors(cache):
    _orbit3d_validate_view_cache(cache)
    reader = _Orbit3DUnifiedViewReader(cache)
    images = _Orbit3DLazyViewImageBatch(reader, tuple(cache["image_shape"]))
    masks = _Orbit3DLazyViewMaskBatch(reader, tuple(cache["mask_shape"]))
    return images, masks
# --- ORBIT3D PATCH 3.6.3 LAZY MULTIVIEW IMAGES/MASKS END ---



class Orbit3DUnifiedCachePrepare:
    """Prepare lightweight disk-backed handles for MultiView.

    Clay renders, masks, MoGe and Marigold all stay in the Unified spill.
    No full [V,H,W,C] IMAGE or [V,H,W] MASK batch is allocated here.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_manifest": ("ORBIT3D_GENERIC_CACHE_MANIFEST",),
                "after_release": (
                    "STRING",
                    {"forceInput": True, "multiline": True},
                ),
            }
        }

    RETURN_TYPES = (
        "ORBIT3D_VIEW_CACHE",
        "ORBIT3D_MOGE_CACHE",
        "ORBIT3D_MARIGOLD_CACHE",
        "ORBIT3D_CACHE_SESSION",
        "STRING",
    )
    RETURN_NAMES = (
        "view_cache",
        "moge_cache",
        "marigold_cache",
        "cache_session",
        "report",
    )
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"

    def run(self, cache_manifest, after_release, snapshot_persistent=True):
        if not isinstance(cache_manifest, dict):
            raise TypeError("cache_manifest must come from Generic Cache Ref Report")
        if cache_manifest.get("orbit3d_cache_type") != "generic_spill_manifest":
            raise ValueError("Expected generic_spill_manifest")
        if str(cache_manifest.get("stream", "")).strip().lower() != "unified":
            raise ValueError(
                f"Unified Cache Prepare requires stream='Unified', got "
                f"{cache_manifest.get('stream')!r}"
            )
        if not str(after_release or "").strip():
            raise ValueError("after_release dependency is empty")

        refs = list(cache_manifest.get("items") or [])
        if not refs:
            raise ValueError("Unified cache manifest has no items")

        view_items = []
        moge_items = []
        marigold_items = []
        image_shape = mask_shape = normal_per_shape = depth_per_shape = None
        image_dtype = mask_dtype = None

        for i, ref in enumerate(refs):
            _, tensor_file, fields, moge_nodes = _orbit3d_unified_item_nodes(ref)

            ikey, ishape, idtype = _orbit3d_torch_node(fields["image"], f"image[{i}]")
            mkey, mshape, mdtype = _orbit3d_torch_node(fields["mask"], f"mask[{i}]")
            nkey, nshape, ndtype = _orbit3d_torch_node(
                fields["normal_map"], f"normal_map[{i}]"
            )
            dkey, dshape, ddtype = _orbit3d_torch_node(
                fields["depth_map"], f"depth_map[{i}]"
            )

            if len(ishape) != 4 or ishape[0] != 1:
                raise ValueError(f"image[{i}] must be [1,H,W,C], got {ishape}")
            if len(mshape) != 3 or mshape[0] != 1:
                raise ValueError(f"mask[{i}] must be [1,H,W], got {mshape}")
            if len(nshape) != 4 or nshape[0] != 1:
                raise ValueError(f"normal_map[{i}] must be [1,H,W,C], got {nshape}")
            if len(dshape) != 4 or dshape[0] != 1:
                raise ValueError(f"depth_map[{i}] must be [1,H,W,C], got {dshape}")

            if i == 0:
                image_shape, mask_shape = ishape, mshape
                normal_per_shape, depth_per_shape = nshape, dshape
                image_dtype, mask_dtype = idtype, mdtype
            else:
                if ishape != image_shape or idtype != image_dtype:
                    raise ValueError("Unified clay IMAGE schema changes between views")
                if mshape != mask_shape or mdtype != mask_dtype:
                    raise ValueError("Unified MASK schema changes between views")
                if nshape != normal_per_shape or dshape != depth_per_shape:
                    raise ValueError("Unified Marigold schema changes between views")

            moge_schema, moge_keymap = {}, {}
            with safe_open(str(tensor_file), framework="pt", device="cpu") as f:
                disk_keys = set(f.keys())
                for name in ("points", "depth", "normal", "mask", "intrinsics"):
                    key, shape, dtype = _orbit3d_torch_node(
                        moge_nodes[name], f"moge.{name}[{i}]"
                    )
                    if key not in disk_keys:
                        raise KeyError(f"{tensor_file}: missing {key!r}")
                    moge_keymap[name] = key
                    moge_schema[name] = {"shape": list(shape), "dtype": dtype}

                for key in (ikey, mkey, nkey, dkey):
                    if key not in disk_keys:
                        raise KeyError(f"{tensor_file}: missing {key!r}")

            view_items.append({
                "index": i,
                "path": str(tensor_file),
                "image_key": ikey,
                "mask_key": mkey,
            })
            moge_items.append({
                "index": i,
                "path": str(tensor_file),
                "file_size": int(ref.get("file_size") or tensor_file.stat().st_size),
                "tensor_schema": moge_schema,
                "tensor_key_map": moge_keymap,
                "metadata": {},
            })
            marigold_items.append({
                "index": i,
                "path": str(tensor_file),
                "normal_key": nkey,
                "depth_key": dkey,
            })

        V = len(view_items)
        view_cache = {
            "orbit3d_cache_type": "view_cache",
            "version": 1,
            "format": "unified_spill_adapter_v1",
            "session_id": str(cache_manifest.get("session_id", "")),
            "root": str(cache_manifest.get("root", "")),
            "view_count": V,
            "image_shape": [V, *image_shape[1:]],
            "mask_shape": [V, *mask_shape[1:]],
            "image_dtype": image_dtype,
            "mask_dtype": mask_dtype,
            "items": view_items,
            "generic_manifest_path": str(cache_manifest.get("manifest_path", "")),
        }
        moge_cache = {
            "orbit3d_cache_type": "moge_cache",
            "version": 1,
            "format": "unified_spill_adapter_v1",
            "session_id": str(cache_manifest.get("session_id", "")),
            "root": str(cache_manifest.get("root", "")),
            "view_count": V,
            "total_file_size": int(cache_manifest.get("total_file_size") or 0),
            "items": moge_items,
            "generic_manifest_path": str(cache_manifest.get("manifest_path", "")),
        }
        marigold_cache = {
            "orbit3d_cache_type": "marigold_cache",
            "version": 1,
            "format": "unified_spill_adapter_v1",
            "session_id": str(cache_manifest.get("session_id", "")),
            "root": str(cache_manifest.get("root", "")),
            "view_count": V,
            "normal_shape": [V, *normal_per_shape[1:]],
            "depth_shape": [V, *depth_per_shape[1:]],
            "items": marigold_items,
            "generic_manifest_path": str(cache_manifest.get("manifest_path", "")),
        }
        cache_session = {
            "orbit3d_cache_type": "orbit3d_session",
            "version": 1,
            "session_id": str(cache_manifest.get("session_id", "")),
            "root": str(cache_manifest.get("root", "")),
        }
        _validate_session(cache_session)
        # A newly generated session is snapshotted as before. Reopening an
        # existing archive must never try to publish the same archive again.
        if snapshot_persistent:
            _orbit3d_snapshot_persistent(cache_session)

        report = (
            f"Orbit3D unified cache prepared: V={V}; "
            f"view_cache images={tuple(view_cache['image_shape'])} "
            f"masks={tuple(view_cache['mask_shape'])}; "
            "images+masks+MoGe+Marigold remain lazy/disk-backed"
        )
        print(f"[Orbit3D Cache] {report}", flush=True)
        return view_cache, moge_cache, marigold_cache, cache_session, report


NODE_CLASS_MAPPINGS.update({
    "Orbit3DUnifiedViewPayload": Orbit3DUnifiedViewPayload,
    "Orbit3DUnifiedCachePrepare": Orbit3DUnifiedCachePrepare,
    "Orbit3DDecodeUnifiedCacheFrame": Orbit3DDecodeUnifiedCacheFrame,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "Orbit3DUnifiedViewPayload": "Orbit3D - Pack Unified View",
    "Orbit3DUnifiedCachePrepare": "Orbit3D - Prepare Unified Cache for MultiView",
    "Orbit3DDecodeUnifiedCacheFrame": "Orbit3D - Decode Unified Cache Frame",
})
# --- ORBIT3D PATCH 3.6.1 SINGLE LOOP UNIFIED VIEW CACHE END ---
