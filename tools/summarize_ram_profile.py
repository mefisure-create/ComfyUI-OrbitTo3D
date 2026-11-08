#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict, deque
from pathlib import Path

GIB = 1024 ** 3

def gib(v):
    if v in (None, '', 'None'):
        return None
    return float(v) / GIB

def fmt(v):
    return '-' if v is None else f'{v:.2f}'

def latest_session(root: Path) -> Path:
    latest = root / 'LATEST'
    if latest.is_file():
        p = Path(latest.read_text(encoding='utf-8').strip())
        if p.is_dir():
            return p
    sessions = sorted(root.glob('session_*'))
    if not sessions:
        raise SystemExit(f'No profile sessions under {root}')
    return sessions[-1]

def read_samples(session: Path):
    out=[]
    with (session/'samples.csv').open('r',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            def iv(k):
                v=r.get(k)
                return int(v) if v not in (None,'','None') else None
            out.append(dict(
                t=float(r['monotonic']), stage=r.get('stage',''),
                rss=iv('rss'), available=iv('system_available'),
                torch_allocated=iv('torch_allocated'),
                torch_reserved=iv('torch_reserved'),
                nvml_used=iv('nvml_used'),
            ))
    return out

def read_events(session: Path):
    with (session/'events.jsonl').open('r',encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]

def pair_invocations(events):
    pending=defaultdict(deque)
    pairs=[]
    for e in events:
        typ=e.get('event'); stage=e.get('stage')
        if not stage:
            continue
        if typ=='stage_enter':
            pending[stage].append(e)
        elif typ in ('stage_exit','stage_error'):
            if pending[stage]:
                en=pending[stage].popleft()
                pairs.append((en,e))
    pairs.sort(key=lambda p: p[0].get('monotonic',0.0))
    counts=defaultdict(int)
    rows=[]
    for en,ex in pairs:
        stage=en['stage']; counts[stage]+=1
        rows.append((f'{stage}#{counts[stage]}', stage, counts[stage], en, ex))
    return rows

def snapv(e,key):
    return (e.get('snapshot') or {}).get(key)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('session',nargs='?')
    ap.add_argument('--root',default='/home/ai/ComfyUI/output/orbit3d_ram_profiles')
    a=ap.parse_args()
    session=Path(a.session) if a.session else latest_session(Path(a.root))
    samples=read_samples(session); events=read_events(session)
    inv=pair_invocations(events)

    gp=max((s for s in samples if s['rss'] is not None),key=lambda s:s['rss'],default=None)
    print(f'Session: {session}')
    if gp:
        print(f'Global RSS peak: {fmt(gib(gp["rss"]))} GiB  sampled-stage={gp["stage"]}')
    av=min((s['available'] for s in samples if s['available'] is not None),default=None)
    print(f'Min system available: {fmt(gib(av))} GiB\n')

    hdr=['invocation','sec','RSS0','RSSpk','RSS1','dRSS','AvailMin','TAllocPk','TResPk','NVMLPk','OutCPU','OutGPU','status']
    widths=[36,8,7,7,7,7,8,8,8,8,7,7,8]
    print(' '.join(h.ljust(w) for h,w in zip(hdr,widths)))
    print('-'*sum(w+1 for w in widths))
    csv_rows=[]
    for name,stage,num,en,ex in inv:
        t0=float(en.get('monotonic',0)); t1=float(ex.get('monotonic',t0))
        ss=[s for s in samples if t0 <= s['t'] <= t1]
        r0=snapv(en,'rss'); r1=snapv(ex,'rss')
        rpk=max((s['rss'] for s in ss if s['rss'] is not None),default=max(x for x in (r0,r1) if x is not None) if any(x is not None for x in (r0,r1)) else None)
        amin=min((s['available'] for s in ss if s['available'] is not None),default=None)
        tap=max((s['torch_allocated'] for s in ss if s['torch_allocated'] is not None),default=None)
        trp=max((s['torch_reserved'] for s in ss if s['torch_reserved'] is not None),default=None)
        nvp=max((s['nvml_used'] for s in ss if s['nvml_used'] is not None),default=None)
        payload=ex.get('output_payload') or {}
        ocpu=int(payload.get('unique_cpu_bytes',0) or 0); ogpu=int(payload.get('unique_cuda_bytes',0) or 0)
        status='ERROR' if ex.get('event')=='stage_error' else 'OK'
        dur=float(ex.get('duration_s', t1-t0))
        vals=[name,f'{dur:.1f}',fmt(gib(r0)),fmt(gib(rpk)),fmt(gib(r1)),fmt(gib(None if r0 is None or r1 is None else r1-r0)),fmt(gib(amin)),fmt(gib(tap)),fmt(gib(trp)),fmt(gib(nvp)),fmt(gib(ocpu)),fmt(gib(ogpu)),status]
        print(' '.join(str(v).ljust(w)[:w] for v,w in zip(vals,widths)))
        csv_rows.append(dict(invocation=name,stage=stage,invocation_index=num,duration_s=dur,
            rss_start_gib=gib(r0),rss_peak_gib=gib(rpk),rss_end_gib=gib(r1),
            retained_delta_gib=gib(None if r0 is None or r1 is None else r1-r0),
            system_available_min_gib=gib(amin),torch_alloc_peak_gib=gib(tap),torch_reserved_peak_gib=gib(trp),
            nvml_used_peak_gib=gib(nvp),output_cpu_gib=gib(ocpu),output_gpu_gib=gib(ogpu),status=status))

    if csv_rows:
        out=session/'summary_invocations.csv'
        with out.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(csv_rows[0])); w.writeheader(); w.writerows(csv_rows)
        print(f'\nWrote: {out}')

    print('\nLargest output payload leaves by invocation:')
    for name,stage,num,en,ex in inv:
        leaves=(ex.get('output_payload') or {}).get('top_leaves',[])[:8]
        if not leaves:
            continue
        print(f'\n[{name}]')
        for q in leaves:
            print(f"  {q.get('logical_bytes',0)/GIB:7.3f} GiB  {q.get('device','?'):8s}  {q.get('kind','?'):16s}  {q.get('path','?')}")

if __name__=='__main__':
    main()
