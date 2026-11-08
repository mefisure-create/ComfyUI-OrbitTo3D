"""Orbit3D mesh-socket component cleaning and isolated WTiVo integration.

This module is additive. It does not modify Trellis2 / Pixal3D / WTiVo algorithms.
ComfyUI owns the lifetime of the transient, resource-limited WTiVo worker.
The native CUDA WTiVo solver is not a torch.nn.Module and cannot be ModelPatched.
"""
from __future__ import annotations

import csv
import gc
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import threading
import time
import uuid

import numpy as np
import torch
import folder_paths

from ..trellis2.representations import MeshWithVoxel

_GIB = 1024 ** 3
_RUN_SUBDIR = 'orbit3d_wtivo_runs'
_DEFAULT_COMPONENT_BIN = os.environ.get('ORBIT3D_COMPONENT_TOOL', 'orbit_raw_component_tool')
_DEFAULT_LAUNCHER = os.environ.get('ORBIT3D_WTIVO_LAUNCHER', '/usr/local/sbin/orbit3d-wtivo-systemd-launch')


def _rss_bytes(pid: int):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def _mem_available_bytes():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def _gpu_memory(device_index: int):
    """Observed total GPU use (not per-process allocation); no CUDA initialization."""
    try:
        p = subprocess.run(
            ['/usr/bin/nvidia-smi', '--query-gpu=index,memory.used,memory.free',
             '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=3,
            check=True,
        )
        for ln in p.stdout.splitlines():
            cols = [c.strip() for c in ln.split(',')]
            if len(cols) >= 3 and int(cols[0]) == device_index:
                return int(cols[1]) * 1024**2, int(cols[2]) * 1024**2
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return None, None


def _json_report(path: Path, payload: dict):
    tmp = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex[:8])
    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _report_text(label: str, d: dict):
    return label + '\n' + '\n'.join(f'{k}: {v}' for k, v in d.items())


def _run_root() -> Path:
    root = Path(folder_paths.get_output_directory()).resolve() / _RUN_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def _clean_label(s: str):
    s = re.sub(r'[^a-zA-Z0-9_.-]+', '_', str(s)).strip('._-')
    return (s[:70] or 'run')


def _run_dir(input_path: Path, label: str) -> Path:
    root = _run_root()
    parent = input_path.resolve().parent
    try:
        parent.relative_to(root)
        # A path from earlier steps of this run; keep the entire run together.
        d = parent
    except ValueError:
        d = root / (time.strftime('%Y%m%d_%H%M%S') + '_' + _clean_label(label) + '_' + uuid.uuid4().hex[:8])
        d.mkdir(parents=True, exist_ok=False)
    return d


def _select_ply_path(ply_path: str, input_ply_path=None) -> Path:
    """Prefer connected STRING input; retain manual path widget for standalone runs."""
    from_socket = str(input_ply_path or '').strip()
    manual = str(ply_path or '').strip()
    selected = from_socket or manual
    if not selected:
        raise ValueError('PLY path is empty: connect input_ply_path or fill ply_path')
    return _check_ply_input(selected)


def _resolve_component_binary(path: str) -> Path:
    path = str(path or '').strip()
    if not path:
        raise ValueError('component_tool_path is empty; specify the compiled C++ cleaner')
    candidate = shutil.which(path) if '/' not in path else path
    if not candidate:
        raise RuntimeError(f'Component cleaner executable not found in PATH: {path}')
    binary = Path(candidate).expanduser().resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f'Component cleaner missing or not executable: {binary}')
    return binary


def _check_ply_input(s: str) -> Path:
    path = Path(str(s).strip()).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != '.ply' or path.stat().st_size <= 100:
        raise ValueError(f'Expected nonempty .ply file: {path}')
    return path


def _comfy_interrupt():
    """Observe ComfyUI prompt cancellation, including while a worker is running."""
    import comfy.model_management as mm
    fn = getattr(mm, 'throw_exception_if_processing_interrupted', None)
    if callable(fn):
        fn()


def _prepare_comfy_memory_for_external_worker():
    """Offload managed ComfyUI models before an external native CUDA worker.

    WTiVo uses CUDA extensions outside ComfyUI's torch process.  It is not a
    ModelPatcher and cannot be offloaded with unload_model_and_clones().
    """
    import comfy.model_management as mm
    _comfy_interrupt()
    unload = getattr(mm, 'unload_all_models', None)
    if not callable(unload):
        raise RuntimeError('ComfyUI model_management.unload_all_models unavailable; refusing uncoordinated WTiVo startup')
    t0 = time.monotonic()
    mem_before = _mem_available_bytes()
    unload()
    gc.collect()
    empty = getattr(mm, 'soft_empty_cache', None)
    if callable(empty):
        empty()
    _comfy_interrupt()
    return dict(comfy_model_offload_done=True,
                comfy_model_offload_seconds=round(time.monotonic()-t0,3),
                host_available_before_comfy_offload_bytes=mem_before,
                host_available_after_comfy_offload_bytes=_mem_available_bytes())


def _unit_state(unit: str) -> str:
    """Observe the transient systemd job; 'missing' is a successful final state."""
    try:
        p = subprocess.run(['/usr/bin/systemctl', 'is-active', unit+'.service'],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, timeout=5, check=False)
        state = p.stdout.strip()
        return state or ('unknown' if p.returncode else 'active')
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def _stop_wtivo_unit(unit: str):
    """Ask the narrow root-owned launcher to stop only this transient WTiVo unit."""
    p = subprocess.run(['/usr/bin/sudo', '-n', _DEFAULT_LAUNCHER, 'stop', unit],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, timeout=25, check=False)
    if p.returncode:
        raise RuntimeError(f'Cannot stop WTiVo unit {unit}: {p.stdout.strip()} {p.stderr.strip()}')
    return p.stdout.strip() or 'stopped'


def _end_process_group(proc):
    """Kill only the command's dedicated process group; never the ComfyUI group."""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)


def _stream_process(cmd, logfile: Path, monitor_gpu_index=None, timeout_sec=None,
                    check_interrupt=None, abort_callback=None):
    """Drain child stdout, sample resources, and own cancellation/teardown.

    For WTiVo, abort_callback stops the *systemd unit*, not just systemd-run.
    For the C++ cleaner, terminating its process group suffices.
    """
    t0 = time.monotonic()
    baseline_used, _ = _gpu_memory(monitor_gpu_index) if monitor_gpu_index is not None else (None, None)
    peak_gpu_used = baseline_used
    peak_rss = None
    with logfile.open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, start_new_session=True)
        def drain():
            try:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    print('[Orbit3D External] ' + line.rstrip(), flush=True)
            except (OSError, ValueError):
                pass
            finally:
                try:
                    proc.stdout.close()
                except OSError:
                    pass

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        rc = None
        try:
            while proc.poll() is None:
                if check_interrupt is not None:
                    check_interrupt()
                if timeout_sec is not None and time.monotonic() - t0 > timeout_sec:
                    raise TimeoutError(f'External process timed out after {timeout_sec} s; log={logfile}')
                rss = _rss_bytes(proc.pid)
                if rss is not None:
                    peak_rss = max(peak_rss or 0, rss)
                if monitor_gpu_index is not None:
                    used, _ = _gpu_memory(monitor_gpu_index)
                    if used is not None:
                        peak_gpu_used = max(peak_gpu_used or 0, used)
                time.sleep(0.5)
            rc = proc.wait()
        except BaseException:
            # Prompt cancellation / timeout: external CUDA memory must be freed.
            if abort_callback is not None:
                try:
                    abort_callback()
                except Exception as e:
                    print(f'[Orbit3D WTiVo] CRITICAL: unit cancellation failed: {e}', flush=True)
            _end_process_group(proc)
            # A second stop closes the rare cancel-before-systemd-unit-created race.
            if abort_callback is not None:
                try:
                    abort_callback()
                except Exception as e:
                    print(f'[Orbit3D WTiVo] CRITICAL: post-termination unit stop failed: {e}', flush=True)
            raise
        finally:
            reader.join(timeout=15)
            if reader.is_alive():
                try:
                    proc.stdout.close()
                except OSError:
                    pass
                reader.join(timeout=1)
        if rc != 0:
            tail = ''.join(logfile.read_text(encoding='utf-8', errors='replace').splitlines(keepends=True)[-20:])
            raise RuntimeError(f'External process exit={rc}. Log: {logfile}\n--- LOG TAIL ---\n{tail}')
    return dict(elapsed_s=round(time.monotonic()-t0, 3),
                launcher_peak_rss_bytes=peak_rss,
                gpu_total_used_baseline_bytes=baseline_used,
                gpu_total_used_observed_peak_bytes=peak_gpu_used,
                gpu_observed_delta_bytes=(max(0, peak_gpu_used - baseline_used)
                                          if peak_gpu_used is not None and baseline_used is not None else None))


def _component_summary(csv_path: Path, max_faces: int, max_area: float):
    comps = 0
    rem_comps = 0
    total_faces = 0
    total_area = 0.0
    removed_faces = 0
    removed_area = 0.0
    largest_id = None
    largest_area = -1.0
    rows = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            faces, area = int(r['faces']), float(r['area'])
            rows.append((str(r['component_root_id']), faces, area))
            total_faces += faces
            total_area += area
            comps += 1
            if area > largest_area:
                largest_area, largest_id = area, str(r['component_root_id'])
    for root, faces, area in rows:
        if root != largest_id and faces <= max_faces and area <= max_area:
            rem_comps += 1
            removed_faces += faces
            removed_area += area
    return dict(components_before=comps, components_removed=rem_comps,
                components_after=comps-rem_comps,
                faces_before=total_faces, faces_removed=removed_faces,
                faces_after=total_faces-removed_faces,
                area_before=total_area, area_removed=removed_area,
                removed_face_fraction=removed_faces/total_faces if total_faces else 0.0,
                removed_area_fraction=removed_area/total_area if total_area else 0.0)


class _GeometryOnlyMeshWithVoxel(MeshWithVoxel):
    """MeshWithVoxel socket compatibility without inventing lost voxel attributes."""
    def to(self, device, non_blocking=False):
        return _GeometryOnlyMeshWithVoxel(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            origin=self.origin.detach().cpu().tolist(),
            voxel_size=self.voxel_size,
            coords=None, attrs=None, voxel_shape=None, layout={},
        )

    def query_attrs(self, xyz):
        raise RuntimeError('PLY import is geometry-only; voxel attrs cannot be reconstructed. '
                           'Use an Orbit3D native MESHWITHVOXEL checkpoint for voxel operations.')


# Required PLY geometry layout: scalar float/double XYZ vertices, triangle index list faces;
# all other attributes ignored (not silently reinterpreted). No rotations, processing, welding.
_PLY_TYPE = {
    'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1',
    'short': '<i2', 'int16': '<i2', 'ushort': '<u2', 'uint16': '<u2',
    'int': '<i4', 'int32': '<i4', 'uint': '<u4', 'uint32': '<u4',
    'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
}


def _ply_layout(path: Path):
    vcount = fcount = None
    vprops = []
    face_list = None
    current = None
    with path.open('rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError('Not a PLY file')
        if f.readline().strip() != b'format binary_little_endian 1.0':
            raise ValueError('PLY import requires binary_little_endian 1.0')
        for _ in range(512):
            line = f.readline()
            if not line:
                raise ValueError('PLY header not terminated')
            t = line.decode('ascii').strip().split()
            if not t:
                continue
            if t[0] == 'end_header':
                header_size = f.tell()
                break
            if t[0] == 'element':
                if t[1] == 'vertex':
                    if vcount is not None or fcount is not None:
                        raise ValueError('PLY requires vertex first, face second')
                    vcount = int(t[2]); current = 'vertex'
                elif t[1] == 'face':
                    if vcount is None or fcount is not None:
                        raise ValueError('PLY requires vertex first, face second')
                    fcount = int(t[2]); current = 'face'
                else:
                    raise ValueError('PLY extra elements not supported')
            elif t[0] == 'property':
                if current == 'vertex' and len(t) == 3 and t[1] in _PLY_TYPE:
                    vprops.append((t[2], _PLY_TYPE[t[1]]))
                elif current == 'face' and len(t) == 5 and t[1] == 'list':
                    if face_list is not None or t[4] not in ('vertex_indices', 'vertex_index'):
                        raise ValueError('Only a single triangle vertex_indices list is supported')
                    face_list = (t[2], t[3])
                elif t[0] != 'comment' and t[0] != 'obj_info':
                    raise ValueError('Unsupported PLY property declaration: ' + line.decode('ascii', errors='replace'))
        else:
            raise ValueError('PLY header too long')
    if not vcount or not fcount or not face_list or not vprops:
        raise ValueError('PLY missing vertices / faces / properties')
    if len(set(x for x, _ in vprops)) != len(vprops):
        raise ValueError('Duplicate vertex property')
    vdtype = np.dtype(vprops, align=False)
    for axis in ('x','y','z'):
        if axis not in vdtype.names or vdtype[axis].kind != 'f':
            raise ValueError('PLY requires float/double XYZ properties')
    ct, it = face_list
    if ct not in ('uchar','uint8','ushort','uint16') or it not in ('int','int32','uint','uint32'):
        raise ValueError('PLY face list must use uint8/uint16 count and int32/uint32 indices')
    face_dtype = np.dtype([('count', _PLY_TYPE[ct]), ('indices', np.dtype(_PLY_TYPE[it]), (3,))], align=False)
    expected = header_size + vdtype.itemsize*vcount + face_dtype.itemsize*fcount
    if expected != path.stat().st_size:
        raise ValueError(f'PLY byte-size mismatch: expected {expected}, actual {path.stat().st_size}')
    return header_size, vcount, fcount, vdtype, face_dtype


def _read_ply_geometry(path: Path):
    header_size, nv, nf, vdt, fdt = _ply_layout(path)
    vertex_rec = np.memmap(path, mode='c', offset=header_size, dtype=vdt, shape=(nv,))
    offset_faces = header_size + nv*vdt.itemsize
    face_rec = np.memmap(path, mode='c', offset=offset_faces, dtype=fdt, shape=(nf,))
    for begin in range(0, nf, 1_000_000):
        end = min(nf, begin + 1_000_000)
        if not np.all(face_rec['count'][begin:end] == 3):
            raise ValueError('PLY contains non-triangle faces')
        ids = face_rec['indices'][begin:end]
        if ids.dtype.kind == 'i' and np.any(ids < 0):
            raise ValueError('PLY contains negative vertex indices')
        if np.any(ids >= nv):
            raise ValueError('PLY face index out of range')
    # A single contiguous float32 XYZ layout is directly tensor-backed by the
    # copy-on-write mmap. Otherwise copy only XYZ (no mesh/voxel reconstruction).
    if (vdt.itemsize == 12 and vdt.names == ('x','y','z')
            and all(vdt.fields[axis][0] == np.dtype('<f4') for axis in ('x','y','z'))):
        xyz = vertex_rec.view(np.float32).reshape(nv, 3)
        precision = 'float32 exact'
    else:
        xyz = np.empty((nv, 3), dtype=np.float32)
        for i, axis in enumerate(('x','y','z')):
            xyz[:, i] = vertex_rec[axis]
        precision = 'float64 -> float32 conversion' if any(vdt[axis].itemsize == 8 for axis in ('x','y','z')) else 'float32 XYZ copied'
    faces = np.empty((nf, 3), dtype=np.int32)
    for begin in range(0, nf, 1_000_000):
        end = min(nf, begin+1_000_000)
        faces[begin:end] = face_rec['indices'][begin:end]
    if not np.all(np.isfinite(xyz)):
        raise ValueError('PLY has nonfinite coordinates')
    return xyz, faces, precision


class Orbit3DImportMeshPLY:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'ply_path': ('STRING', {'default':'', 'multiline':False}),
            'load_device': (['cpu','cuda'], {'default':'cpu'}),
        }}
    RETURN_TYPES = ('MESHWITHVOXEL', 'STRING')
    RETURN_NAMES = ('mesh', 'report')
    FUNCTION = 'load'
    CATEGORY = 'Orbit3D/Mesh/WTiVo'
    DESCRIPTION = 'Direct PLY triangle geometry import without remesh, rotations or voxel synthesis. No attrs/coords.'

    def load(self, ply_path, load_device):
        t = time.monotonic()
        src = _select_ply_path(ply_path)
        before = _rss_bytes(os.getpid())
        xyz, faces, precision = _read_ply_geometry(src)
        mesh = _GeometryOnlyMeshWithVoxel(
            torch.from_numpy(xyz), torch.from_numpy(faces),
            origin=[0.0,0.0,0.0], voxel_size=1.0,
            coords=None, attrs=None, voxel_shape=None, layout={},
        )
        if load_device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; use CPU')
            mesh = mesh.to('cuda')
        after = _rss_bytes(os.getpid())
        info = dict(source=str(src), vertices=int(len(mesh.vertices)), faces=int(len(mesh.faces)),
                    device=str(mesh.device), xyz_precision=precision,
                    geometry_changes='none (except float64 PLY precision -> float32 required by MeshWithVoxel)',
                    voxel_payload='absent; geometry-only',
                    elapsed_s=round(time.monotonic()-t, 3),
                    parent_process_rss_before_bytes=before,
                    parent_process_rss_after_bytes=after)
        root = _run_dir(src, 'import')
        path = root / ('import_' + uuid.uuid4().hex[:8] + '.report.json')
        _json_report(path, info)
        info['report_json_path'] = str(path)
        report = _report_text('Orbit3D PLY Import (geometry-only)', info)
        print(report, flush=True)
        return mesh, report



def _mesh_run_directory(label: str) -> Path:
    root = _run_root()
    dest = root / (time.strftime('%Y%m%d_%H%M%S') + '_' +
                   _clean_label(label) + '_' + uuid.uuid4().hex[:8])
    dest.mkdir(parents=True, exist_ok=False)
    return dest


def _export_mesh_ply_streamed(mesh: MeshWithVoxel, destination: Path):
    """Write exact float32 XYZ and int32 triangle indices, no geometry processing.

    Bounded GPU-to-CPU chunks instead of materializing another 40M-face mesh.
    The output PLY is compatible with the supplied C++ cleaner and WTiVo.
    """
    if not isinstance(mesh, MeshWithVoxel):
        raise TypeError('Expected MESHWITHVOXEL / MeshWithVoxel input')
    vertices, faces = mesh.vertices, mesh.faces
    if not isinstance(vertices, torch.Tensor) or not isinstance(faces, torch.Tensor):
        raise TypeError('Mesh vertices/faces must be torch tensors')
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('Mesh vertices must be [V,3] and triangle faces [F,3]')
    nv, nf = int(vertices.shape[0]), int(faces.shape[0])
    if not nv or not nf or nv >= 2**31 or nf >= 2**32:
        raise ValueError('Mesh is empty or too large for PLY 32-bit face indices')
    t0 = time.monotonic()
    temp = destination.with_name(destination.name + '.partial.' + uuid.uuid4().hex[:8])
    face_dtype = np.dtype([('count','u1'),('indices','<i4',(3,))], align=False)
    assert face_dtype.itemsize == 13
    try:
        with temp.open('wb') as f:
            f.write((f'ply\nformat binary_little_endian 1.0\n'
                     f'element vertex {nv}\nproperty float x\nproperty float y\n'
                     f'property float z\nelement face {nf}\n'
                     f'property list uchar int vertex_indices\nend_header\n').encode('ascii'))
            for i in range(0, nv, 262_144):
                part = vertices[i:i+262_144].detach().to(device='cpu', dtype=torch.float32).contiguous().numpy()
                if not np.all(np.isfinite(part)):
                    raise ValueError(f'Non-finite vertex coordinates near index {i}')
                f.write(part.astype('<f4', copy=False).tobytes(order='C'))
            for i in range(0, nf, 262_144):
                part = faces[i:i+262_144].detach().to(device='cpu', dtype=torch.int64).contiguous().numpy()
                if np.any(part < 0) or np.any(part >= nv):
                    raise ValueError(f'Invalid face index near face {i}')
                records = np.empty((len(part),), dtype=face_dtype)
                records['count'] = 3
                records['indices'] = part.astype(np.int32, copy=False)
                f.write(records.tobytes(order='C'))
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)
    return dict(serialized_vertices=nv, serialized_faces=nf,
                serialization_s=round(time.monotonic()-t0,3),
                serialization_source_device=str(vertices.device),
                serialized_ply_bytes=destination.stat().st_size,
                serialization='streamed XYZ float32 + triangles int32; no geometric transformation')


def _geometry_mesh_from_ply(path: Path, load_device: str='cpu'):
    xyz, faces, precision = _read_ply_geometry(path)
    mesh = _GeometryOnlyMeshWithVoxel(
        torch.from_numpy(xyz), torch.from_numpy(faces),
        origin=[0.0,0.0,0.0], voxel_size=1.0,
        coords=None, attrs=None, voxel_shape=None, layout={},
    )
    if load_device == 'cuda':
        mesh = mesh.to('cuda')
    return mesh, precision


class Orbit3DComponentCleanerPLY:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'mesh': ('MESHWITHVOXEL',),
            'stage': (['pre','post'], {'default':'pre'}),
            'max_faces': ('INT', {'default':512, 'min':1, 'max':10_000_000}),
            'max_area': ('FLOAT', {'default':0.00003, 'min':0.0, 'max':100.0, 'step':0.000001, 'round':False}),
            'component_tool_path': ('STRING', {'default':_DEFAULT_COMPONENT_BIN, 'multiline':False}),
            'minimum_mem_available_gib': ('INT', {'default':8, 'min':0, 'max':256}),
        }}
    RETURN_TYPES = ('MESHWITHVOXEL','STRING')
    RETURN_NAMES = ('mesh','report')
    FUNCTION = 'clean'
    CATEGORY = 'Orbit3D/Mesh/WTiVo'
    DESCRIPTION = ('Mesh-in / mesh-out conservative component cleaner. '
                   'Disk PLY is internal; pre/post reports and removed geometry saved to files.')

    def clean(self, mesh, stage, max_faces, max_area, component_tool_path, minimum_mem_available_gib):
        t0 = time.monotonic()
        _comfy_interrupt()
        binary = _resolve_component_binary(component_tool_path)
        avail = _mem_available_bytes()
        if avail is not None and avail < int(minimum_mem_available_gib)*_GIB:
            raise MemoryError(f'MemAvailable={avail/_GIB:.2f} GiB below guard {minimum_mem_available_gib} GiB')
        root = _mesh_run_directory('clean_'+stage)
        src = root / 'mesh_input.ply'
        export_stats = _export_mesh_ply_streamed(mesh,src)
        _comfy_interrupt()
        tag = stage + '_' + uuid.uuid4().hex[:8]
        prefix = root / tag
        csv_path = root / f'{tag}.components.csv'
        log = root / f'{tag}.log'
        result = _stream_process([
            str(binary), 'clean', str(src), '--csv', str(csv_path),
            '--max-faces', str(int(max_faces)), '--max-area', str(float(max_area)),
            '--prefix', str(prefix),
        ], log, check_interrupt=_comfy_interrupt)
        kept, removed = Path(str(prefix)+'.kept.ply'), Path(str(prefix)+'.removed.ply')
        if not kept.is_file() or kept.stat().st_size < 100 or not removed.is_file():
            raise RuntimeError(f'Cleaner reported success but outputs missing; log={log}')
        stats = _component_summary(csv_path, int(max_faces), float(max_area))
        _comfy_interrupt()
        output_mesh, precision = _geometry_mesh_from_ply(kept, 'cpu')
        info = dict(stage=stage, source=str(src), kept_ply_path=str(kept),
                    removed_ply_path=str(removed), components_csv_path=str(csv_path),
                    log_path=str(log), threshold_faces=int(max_faces), threshold_area=float(max_area),
                    output_mesh_device=str(output_mesh.device), output_mesh_voxel_attrs='absent',
                    output_xyz_precision=precision,
                    elapsed_total_s=round(time.monotonic()-t0,3),
                    **export_stats, **stats, **result)
        report_path = root / f'{tag}.report.json'
        _json_report(report_path, info)
        info['report_json_path'] = str(report_path)
        report = _report_text(f'Orbit3D Component Cleaner [{stage}]', info)
        print(report, flush=True)
        return output_mesh, report


def _wtivo_extract_summary(log_path: Path):
    text = log_path.read_text(encoding='utf-8', errors='replace')
    result = {}
    regex = {
        'proxy_points': r'\[WTiVo-PointBudget\] CGAL proxy points=([\d,]+)',
        'tetrahedra': r'\btets=(\d+)',
        'graph_surface_faces': r'\[WTiVo-Surface\].*?faces=(\d+)',
        'final_vertices': r'\[FINAL\] v/f=([\d,]+)/[\d,]+',
        'final_faces': r'\[FINAL\] v/f=[\d,]+/([\d,]+)',
        'edge_audit_bad_groups': r'\[WTiVo-Audit\] watertight=(?:True|False) \| bad_edge_groups=(\d+)',
        'cgroup_peak_bytes': r'ORBIT3D_CGROUP_PEAK_BYTES=(\d+)',
        'cgroup_oom_kill': r'ORBIT3D_CGROUP_OOM_KILL=(\d+)',
    }
    for key, pattern in regex.items():
        m = re.search(pattern, text)
        if m:
            result[key] = int(m.group(1).replace(',',''))
    root_m = re.search(r'^ORBIT3D_WTIVO_ROOT=(.+)$', text, re.MULTILINE)
    if root_m:
        result['wtivo_root_resolved'] = root_m.group(1).strip()
    m = re.search(r'\[WTiVo-Audit\] watertight=(True|False)', text)
    if m:
        result['watertight_edge_audit'] = m.group(1) == 'True'
    return result


class Orbit3DWTiVoExternalRunner:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'mesh': ('MESHWITHVOXEL',),
            'input_res': ('INT', {'default':1536, 'min':64, 'max':8192, 'step':64}),
            'final_res': ('INT', {'default':2048, 'min':64, 'max':8192, 'step':64}),
            'proxy_points': ('INT', {'default':12_000_000, 'min':1000, 'max':50_000_000, 'step':1000}),
            'lambda_fill': ('FLOAT', {'default':10.0, 'min':0.0, 'max':1000.0, 'step':0.1}),
            'feature_weight': ('FLOAT', {'default':1.5, 'min':0.0, 'max':100.0, 'step':0.1}),
            'threads': ('INT', {'default':4, 'min':1, 'max':128}),
            'memory_limit_gib': ('INT', {'default':40, 'min':4, 'max':128}),
            'minimum_mem_available_gib': ('INT', {'default':24, 'min':0, 'max':512}),
            'gpu_index': ('INT', {'default':0, 'min':0, 'max':15}),
            'min_gpu_free_gib': ('INT', {'default':8, 'min':0, 'max':96}),
            'wtivo_root': ('STRING', {'default':'', 'multiline':False}),
        }}
    RETURN_TYPES = ('MESHWITHVOXEL','STRING')
    RETURN_NAMES = ('mesh','report')
    FUNCTION = 'run'
    CATEGORY = 'Orbit3D/Mesh/WTiVo'
    DESCRIPTION = ('ComfyUI-controlled transient WTiVo worker: offload ComfyUI models first; '
                   'cgroup memory cap; stop worker on prompt cancellation; per-node report.')

    def run(self, mesh, input_res, final_res, proxy_points, lambda_fill,
            feature_weight, threads, memory_limit_gib, minimum_mem_available_gib,
            gpu_index, min_gpu_free_gib, wtivo_root=''):
        t0 = time.monotonic()
        _comfy_interrupt()
        root = _mesh_run_directory('wtivo')
        src = root / 'mesh_input.ply'
        export_stats = _export_mesh_ply_streamed(mesh, src)
        _comfy_interrupt()
        if not Path(_DEFAULT_LAUNCHER).exists():
            raise RuntimeError(f'Install audited privileged systemd launcher first: {_DEFAULT_LAUNCHER}')
        if shutil.which('sudo') is None:
            raise RuntimeError('sudo missing; configure audited launcher for ComfyUI service user')

        # All GPU ownership changes originate from this ComfyUI node.
        # The transient systemd worker is NOT a standalone resident daemon.
        _comfy_interrupt()
        before_comfy = _gpu_memory(int(gpu_index))
        comfy_offload = _prepare_comfy_memory_for_external_worker()
        available = _mem_available_bytes()
        if available is not None and available < minimum_mem_available_gib*_GIB:
            raise MemoryError(f'Insufficient RAM *after ComfyUI offload* for WTiVo: '
                              f'MemAvailable={available/_GIB:.2f} GiB, guard={minimum_mem_available_gib} GiB')
        used_gpu, free_gpu = _gpu_memory(int(gpu_index))
        if free_gpu is not None and free_gpu < min_gpu_free_gib*_GIB:
            raise MemoryError(f'Insufficient GPU VRAM after ComfyUI offload: '
                              f'{free_gpu/_GIB:.2f} GiB, guard={min_gpu_free_gib} GiB')

        tag = 'wtivo_' + uuid.uuid4().hex[:8]
        unit = 'orbit3d-wtivo-' + uuid.uuid4().hex[:24]
        out = root / (tag+'.ply')
        logfile = root / (tag+'.log')
        cmd = [
            '/usr/bin/sudo', '-n', _DEFAULT_LAUNCHER, 'run', unit,
            str(src), str(out), str(int(input_res)), str(int(final_res)),
            str(int(proxy_points)), str(float(lambda_fill)), str(float(feature_weight)),
            str(int(threads)), str(int(memory_limit_gib)), str(int(gpu_index)),
            str(wtivo_root or '').strip(),
        ]
        _comfy_interrupt()
        try:
            samples = _stream_process(cmd, logfile, monitor_gpu_index=int(gpu_index),
                                      timeout_sec=1900,
                                      check_interrupt=_comfy_interrupt,
                                      abort_callback=lambda: _stop_wtivo_unit(unit))
            final_unit_state = _unit_state(unit)
            if final_unit_state in ('active', 'activating', 'reloading', 'deactivating'):
                _stop_wtivo_unit(unit)
                raise RuntimeError(f'WTiVo worker remained active after launcher exit; forced stop. Unit={unit}')
        except BaseException as exc:
            report_path = root / (tag + '.failure.report.json')
            try:
                _json_report(report_path, dict(status='failure', source=str(src),
                                               unit_name=unit, log_path=str(logfile),
                                               error=str(exc),
                                               unit_state_after_abort=_unit_state(unit),
                                               **comfy_offload))
            except Exception:
                pass
            raise
        if not out.is_file() or out.stat().st_size < 100:
            raise RuntimeError(f'WTiVo exited successfully but no output mesh; log={logfile}')
        parsed = _wtivo_extract_summary(logfile)
        if parsed.get('watertight_edge_audit') is not True or parsed.get('edge_audit_bad_groups') != 0:
            raise RuntimeError(f'WTiVo output missing passing edge audit; log={logfile}')
        _comfy_interrupt()
        result_mesh, xyz_precision = _geometry_mesh_from_ply(out, 'cpu')
        used_gpu_after, free_gpu_after = _gpu_memory(int(gpu_index))
        info = dict(status='success', source=str(src), output_ply_path=str(out),
                    log_path=str(logfile), input_res=int(input_res), final_res=int(final_res),
                    proxy_points_requested=int(proxy_points), lambda_fill=float(lambda_fill),
                    feature_weight=float(feature_weight), threads=int(threads),
                    memory_limit_gib=int(memory_limit_gib), wtivo_root_requested=str(wtivo_root or '').strip() or '(launcher config)', worker_unit=unit,
                    lifecycle_owner='ComfyUI node', worker_resident_after_run=False,
                    model_patcher_registration='not_supported_native_CUDA_worker',
                    gpu_used_before_comfy_offload_bytes=before_comfy[0],
                    gpu_used_after_comfy_offload_bytes=used_gpu,
                    gpu_free_after_comfy_offload_bytes=free_gpu,
                    gpu_used_after_worker_exit_bytes=used_gpu_after,
                    gpu_free_after_worker_exit_bytes=free_gpu_after,
                    unit_state_after_run=final_unit_state,
                    host_mem_available_after_offload_bytes=available,
                    output_mesh_device=str(result_mesh.device), output_mesh_voxel_attrs='absent',
                    output_xyz_precision=xyz_precision,
                    elapsed_total_s=round(time.monotonic()-t0,3),
                    **export_stats, **comfy_offload, **samples, **parsed)
        report_path = root / (tag + '.report.json')
        _json_report(report_path, info)
        info['report_json_path'] = str(report_path)
        report = _report_text('Orbit3D WTiVo External Runner / ComfyUI coordinated', info)
        print(report, flush=True)
        return result_mesh, report


NODE_CLASS_MAPPINGS = {
    'Orbit3DImportMeshPLY': Orbit3DImportMeshPLY,
    'Orbit3DComponentCleanerPLY': Orbit3DComponentCleanerPLY,
    'Orbit3DWTiVoExternalRunner': Orbit3DWTiVoExternalRunner,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'Orbit3DImportMeshPLY': 'Orbit3D - Import Mesh PLY [Geometry Only]',
    'Orbit3DComponentCleanerPLY': 'Orbit3D - Component Cleaner [WTiVo]',
    'Orbit3DWTiVoExternalRunner': 'Orbit3D - WTiVo External Runner',
}
