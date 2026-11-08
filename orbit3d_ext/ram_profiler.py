from __future__ import annotations

import csv
import datetime as _dt
import functools
import inspect
import json
import os
from pathlib import Path
import threading
import time
import traceback

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None

_ENABLE_FILE = Path('/home/ai/ComfyUI/.orbit3d_ram_profile_enabled')
_ENV_ENABLE = 'ORBIT3D_RAM_PROFILE'
_ENV_INTERVAL = 'ORBIT3D_RAM_PROFILE_INTERVAL_MS'
_ENV_DIR = 'ORBIT3D_RAM_PROFILE_DIR'
_DEFAULT_INTERVAL_MS = 100

_LOCK = threading.RLock()
_STAGE_STACK = []
_SESSION = None
_EXTERNAL_WRAPPERS_ATTEMPTED = set()


def _enabled():
    value = os.environ.get(_ENV_ENABLE, '').strip().lower()
    return value in {'1', 'true', 'yes', 'on'} or _ENABLE_FILE.exists()


def _proc_status_bytes(key):
    try:
        with open('/proc/self/status', 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(key + ':'):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except Exception:
        pass
    return None


def _system_available():
    try:
        with open('/proc/meminfo', 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _torch_cuda_snapshot():
    out = dict(torch_allocated=None, torch_reserved=None, cuda_free=None, cuda_total=None)
    if torch is None or not torch.cuda.is_available():
        return out
    try:
        dev = torch.cuda.current_device()
        out['torch_allocated'] = int(torch.cuda.memory_allocated(dev))
        out['torch_reserved'] = int(torch.cuda.memory_reserved(dev))
        free_b, total_b = torch.cuda.mem_get_info(dev)
        out['cuda_free'] = int(free_b)
        out['cuda_total'] = int(total_b)
    except Exception:
        pass
    return out


_NVML = None
_NVML_HANDLE = None

def _nvml_used():
    global _NVML, _NVML_HANDLE
    try:
        if _NVML is None:
            import pynvml
            pynvml.nvmlInit()
            _NVML = pynvml
            index = 0
            if torch is not None and torch.cuda.is_available():
                try:
                    index = int(torch.cuda.current_device())
                except Exception:
                    pass
            _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(index)
        return int(_NVML.nvmlDeviceGetMemoryInfo(_NVML_HANDLE).used)
    except Exception:
        return None


def _snapshot():
    s = {
        'wall_time': time.time(),
        'monotonic': time.monotonic(),
        'rss': _proc_status_bytes('VmRSS'),
        'rss_hwm': _proc_status_bytes('VmHWM'),
        'system_available': _system_available(),
        'nvml_used': _nvml_used(),
    }
    s.update(_torch_cuda_snapshot())
    return s


def _current_stage():
    with _LOCK:
        return _STAGE_STACK[-1][1] if _STAGE_STACK else 'BETWEEN_PROFILED_STAGES'


def _push_stage(label):
    token = object()
    with _LOCK:
        _STAGE_STACK.append((token, str(label)))
    return token


def _pop_stage(token):
    with _LOCK:
        for i in range(len(_STAGE_STACK) - 1, -1, -1):
            if _STAGE_STACK[i][0] is token:
                del _STAGE_STACK[i]
                return


def _pil_estimate_bytes(im):
    try:
        bands = len(im.getbands())
        w, h = im.size
        mode = str(getattr(im, 'mode', ''))
        bpb = 2 if mode.startswith('I;16') else (4 if mode in {'I', 'F'} else 1)
        return int(w) * int(h) * int(bands) * bpb
    except Exception:
        return 0


def _payload_summary(obj, root='root', max_leaves=24):
    seen_containers = set()
    torch_storages = {}
    numpy_storages = {}
    pil_objects = {}
    leaves = []

    def leaf(path, kind, logical, device):
        leaves.append({'path': path, 'kind': kind, 'logical_bytes': int(logical), 'device': device})

    def walk(x, path, depth):
        if depth > 10:
            return 0
        if torch is not None and isinstance(x, torch.Tensor):
            logical = int(x.numel()) * int(x.element_size())
            dev = str(x.device)
            try:
                st = x.untyped_storage()
                torch_storages.setdefault((dev, int(st.data_ptr()), int(st.nbytes())), int(st.nbytes()))
            except Exception:
                torch_storages.setdefault((dev, int(x.data_ptr()), logical), logical)
            leaf(path, 'torch.Tensor', logical, dev)
            return logical
        if np is not None and isinstance(x, np.ndarray):
            logical = int(x.nbytes)
            base = x
            try:
                while isinstance(getattr(base, 'base', None), np.ndarray):
                    base = base.base
                key = (int(base.__array_interface__['data'][0]), int(base.nbytes))
                numpy_storages.setdefault(key, int(base.nbytes))
            except Exception:
                numpy_storages.setdefault((id(base), logical), logical)
            leaf(path, 'numpy.ndarray', logical, 'cpu')
            return logical
        if PILImage is not None:
            try:
                if isinstance(x, PILImage.Image):
                    logical = _pil_estimate_bytes(x)
                    pil_objects.setdefault(id(x), logical)
                    leaf(path, f'PIL.{getattr(x, "mode", "?")}', logical, 'cpu')
                    return logical
            except Exception:
                pass
        if isinstance(x, dict):
            oid = id(x)
            if oid in seen_containers:
                return 0
            seen_containers.add(oid)
            return sum(walk(v, f'{path}[{k!r}]', depth + 1) for k, v in x.items())
        if isinstance(x, (list, tuple)):
            oid = id(x)
            if oid in seen_containers:
                return 0
            seen_containers.add(oid)
            return sum(walk(v, f'{path}[{i}]', depth + 1) for i, v in enumerate(x))
        return 0

    logical = walk(obj, root, 0)
    tcpu = sum(n for (dev, _p, _s), n in torch_storages.items() if not dev.startswith('cuda'))
    tcuda = sum(n for (dev, _p, _s), n in torch_storages.items() if dev.startswith('cuda'))
    ncpu = sum(numpy_storages.values())
    pcpu = sum(pil_objects.values())
    leaves.sort(key=lambda d: d['logical_bytes'], reverse=True)
    return {
        'logical_bytes': int(logical),
        'unique_cpu_bytes': int(tcpu + ncpu + pcpu),
        'unique_cuda_bytes': int(tcuda),
        'unique_torch_cpu_bytes': int(tcpu),
        'unique_numpy_bytes': int(ncpu),
        'estimated_pil_bytes': int(pcpu),
        'top_leaves': leaves[:max_leaves],
    }


class _ProfileSession:
    def __init__(self):
        interval_ms = int(os.environ.get(_ENV_INTERVAL, _DEFAULT_INTERVAL_MS))
        self.interval = max(20, interval_ms) / 1000.0
        try:
            import folder_paths
            base = Path(folder_paths.get_output_directory())
        except Exception:
            base = Path('/home/ai/ComfyUI/output')
        override = os.environ.get(_ENV_DIR, '').strip()
        if override:
            base = Path(override)
        root = base / 'orbit3d_ram_profiles'
        root.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.dir = root / f'session_{stamp}_pid{os.getpid()}'
        self.dir.mkdir(parents=True, exist_ok=False)
        (root / 'LATEST').write_text(str(self.dir), encoding='utf-8')
        self.samples_path = self.dir / 'samples.csv'
        self.events_path = self.dir / 'events.jsonl'
        self.meta_path = self.dir / 'meta.json'
        self._stop = threading.Event()
        self._thread = None
        self._sample_file = open(self.samples_path, 'w', encoding='utf-8', newline='', buffering=1)
        self._sample_writer = csv.DictWriter(self._sample_file, fieldnames=[
            'wall_time', 'iso_time', 'monotonic', 'stage', 'rss', 'rss_hwm',
            'system_available', 'torch_allocated', 'torch_reserved', 'cuda_free',
            'cuda_total', 'nvml_used'])
        self._sample_writer.writeheader()
        self._events_file = open(self.events_path, 'a', encoding='utf-8', buffering=1)
        self.meta_path.write_text(json.dumps({
            'pid': os.getpid(), 'started_iso': _dt.datetime.now().isoformat(),
            'interval_ms': int(self.interval * 1000),
            'note': 'Instrumentation only: no gc.collect, empty_cache, CUDA synchronization, tensor moves/copies, or reconstruction changes are performed.'
        }, indent=2), encoding='utf-8')
        print(f'[Orbit3D RAM Profile] session: {self.dir}', flush=True)
        print(f'[Orbit3D RAM Profile] interval: {int(self.interval * 1000)} ms', flush=True)

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name='Orbit3DRAMSampler', daemon=True)
            self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                s = _snapshot()
                row = dict(s)
                row['iso_time'] = _dt.datetime.fromtimestamp(s['wall_time']).isoformat()
                row['stage'] = _current_stage()
                self._sample_writer.writerow(row)
            except Exception as e:
                try:
                    self.event({'event': 'sampler_error', 'error': repr(e), 'traceback': traceback.format_exc()})
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def event(self, payload):
        rec = {'iso_time': _dt.datetime.now().isoformat(), 'wall_time': time.time(), 'monotonic': time.monotonic(), **payload}
        self._events_file.write(json.dumps(rec, ensure_ascii=False) + '\n')


def _ensure_session():
    global _SESSION
    if not _enabled():
        return None
    with _LOCK:
        if _SESSION is None:
            _SESSION = _ProfileSession()
            _SESSION.start()
    _install_external_wrappers()
    return _SESSION


def _sync_call(label, fn, args, kwargs):
    session = _ensure_session()
    if session is None:
        return fn(*args, **kwargs)
    token = _push_stage(label)
    start = _snapshot(); t0 = time.perf_counter()
    session.event({'event': 'stage_enter', 'stage': label, 'snapshot': start,
                   'input_payload': _payload_summary({'args': args[1:] if args else args, 'kwargs': kwargs}, root='input')})
    try:
        result = fn(*args, **kwargs)
    except Exception as e:
        session.event({'event': 'stage_error', 'stage': label, 'duration_s': time.perf_counter()-t0,
                       'snapshot': _snapshot(), 'error': repr(e), 'traceback': traceback.format_exc()})
        raise
    else:
        session.event({'event': 'stage_exit', 'stage': label, 'duration_s': time.perf_counter()-t0,
                       'snapshot': _snapshot(), 'output_payload': _payload_summary(result, root='output')})
        return result
    finally:
        _pop_stage(token)


async def _async_call(label, fn, args, kwargs):
    session = _ensure_session()
    if session is None:
        return await fn(*args, **kwargs)
    token = _push_stage(label)
    start = _snapshot(); t0 = time.perf_counter()
    session.event({'event': 'stage_enter', 'stage': label, 'snapshot': start,
                   'input_payload': _payload_summary({'args': args[1:] if args else args, 'kwargs': kwargs}, root='input')})
    try:
        result = await fn(*args, **kwargs)
    except Exception as e:
        session.event({'event': 'stage_error', 'stage': label, 'duration_s': time.perf_counter()-t0,
                       'snapshot': _snapshot(), 'error': repr(e), 'traceback': traceback.format_exc()})
        raise
    else:
        session.event({'event': 'stage_exit', 'stage': label, 'duration_s': time.perf_counter()-t0,
                       'snapshot': _snapshot(), 'output_payload': _payload_summary(result, root='output')})
        return result
    finally:
        _pop_stage(token)


def _raw_descriptor(cls, method_name):
    for base in cls.__mro__:
        if method_name in base.__dict__:
            return base.__dict__[method_name]
    return None


def _wrap_method(cls, method_name, label):
    if cls is None or not method_name:
        return False
    raw = _raw_descriptor(cls, method_name)
    if raw is None:
        return False

    descriptor_kind = 'instance'
    if isinstance(raw, classmethod):
        descriptor_kind = 'class'
        original = raw.__func__
    elif isinstance(raw, staticmethod):
        descriptor_kind = 'static'
        original = raw.__func__
    else:
        original = raw

    if getattr(original, '_orbit3d_ram_profile_wrapper', False):
        return True

    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def wrapped(*args, **kwargs):
            return await _async_call(label, original, args, kwargs)
    else:
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            return _sync_call(label, original, args, kwargs)

    wrapped._orbit3d_ram_profile_wrapper = True
    wrapped._orbit3d_ram_profile_original = original

    if descriptor_kind == 'class':
        setattr(cls, method_name, classmethod(wrapped))
    elif descriptor_kind == 'static':
        setattr(cls, method_name, staticmethod(wrapped))
    else:
        setattr(cls, method_name, wrapped)
    return True

def _method_for_node_class(cls):
    name = getattr(cls, 'FUNCTION', None)
    if isinstance(name, str) and hasattr(cls, name):
        return name
    for candidate in ('execute', 'process', 'run'):
        if hasattr(cls, candidate):
            return candidate
    return None


_EXTERNAL_TARGETS = {
    'MarigoldV2Predict': 'EXT.MarigoldV2Predict',
    'ImageListToImageBatch': 'EXT.ImageListToImageBatch',
    'MaskListToMaskBatch': 'EXT.MaskListToMaskBatch',
    'ImageCompositeMasked': 'EXT.ImageCompositeMasked',
    'Trellis2DecodeLatents': 'EXT.DecodeLatents',
    'Trellis2ReconstructMeshWithQuad': 'EXT.ReconstructMeshWithQuad',
    'Trellis2PostProcessMesh': 'EXT.PostProcessMesh',
    'Trellis2MeshWithVoxelToTrimesh': 'EXT.MeshToTrimesh',
}


def _install_external_wrappers():
    try:
        import nodes as comfy_nodes
        mappings = getattr(comfy_nodes, 'NODE_CLASS_MAPPINGS', {})
    except Exception:
        return
    for node_id, label in _EXTERNAL_TARGETS.items():
        if node_id in _EXTERNAL_WRAPPERS_ATTEMPTED:
            continue
        cls = mappings.get(node_id)
        if cls is None:
            continue
        method = _method_for_node_class(cls)
        if method and _wrap_method(cls, method, label):
            _EXTERNAL_WRAPPERS_ATTEMPTED.add(node_id)
            print(f'[Orbit3D RAM Profile] wrapped external node {node_id}.{method} as {label}', flush=True)


_ORBIT_TARGETS = {
    'Orbit3DLoadModel': 'O00.LoadModel',
    'Orbit3DVGGTViewSelector': 'O10.VGGT',
    'Orbit3DRealBasicVSR': 'O20.RealBasicVSR',
    'Orbit3DCacheSession': 'O05.CacheSession',
    'Orbit3DVSRBatchCache': 'O21.VSRBatchCache',
    'Orbit3DVSRViewAtIndex': 'O22.VSRView',
    'Orbit3DLoopContextPack': 'O23.LoopContextPack',
    'Orbit3DLoopContextUnpack': 'O24.LoopContextUnpack',
    'Orbit3DUnifiedViewSpill': 'O43.UnifiedViewSpill',
    'Orbit3DUnifiedCacheFinalize': 'O44.UnifiedCacheFinalize',
    'Orbit3DUnifiedCacheManifestLoad': 'O48.LoadUnifiedManifest',
    'Orbit3DMultiViewCacheOpen': 'O49.OpenMultiViewCache',
    'Orbit3DCleanupCacheSession': 'O99.CleanupCacheSession',
'Orbit3DMoGeInference': 'O40.MoGeInference',
'Orbit3DMoGeGeometryListToBatch': 'O50.MoGeListToBatch',
    'Orbit3DMultiViewConfig': 'O60.MultiViewConfig',
    'Orbit3DSparseGenerator': 'O70.Sparse',
    'Orbit3DShapeGenerator': 'O80.Shape512',
    'Orbit3DShapeCascadeGenerator': 'O90.ShapeHR',
    'Orbit3DMeshLibFillHolesNicelyPack': 'O110.MeshLibRepair',
}


def install_profiling(node_class_mappings):
    if not _enabled():
        return False
    installed = []
    for node_id, label in _ORBIT_TARGETS.items():
        cls = node_class_mappings.get(node_id)
        if cls is None:
            continue
        method = _method_for_node_class(cls)
        if method and _wrap_method(cls, method, label):
            installed.append(f'{node_id}.{method}')
    print('[Orbit3D RAM Profile] ENABLED; wrapped Orbit methods: ' + ', '.join(installed), flush=True)
    return True
