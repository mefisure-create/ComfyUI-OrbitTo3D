#!/usr/bin/env python3
"""Compare the pinned visualbruno/ComfyUI-Trellis2 base with a newer upstream ref.

The checker reports upstream changes only for files/symbols retained by Orbit.
It never merges or edits source code.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "upstream_symbols.json"

def run_git(repo: Path, *args: str, check=True) -> str:
    p=subprocess.run(["git","-C",str(repo),*args],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if check and p.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{p.stderr.strip()}")
    return p.stdout

def git_file(repo: Path, ref: str, path: str):
    p=subprocess.run(["git","-C",str(repo),"show",f"{ref}:{path}"],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    return p.stdout if p.returncode == 0 else None

def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def ast_sha(node: ast.AST) -> str:
    return hashlib.sha256(ast.dump(node, annotate_fields=True, include_attributes=False).encode()).hexdigest()

def find_symbol(source: str | None, dotted: str):
    if source is None: return None
    try: tree=ast.parse(source)
    except SyntaxError: return "<syntax-error>"
    parts=dotted.split(".")
    cur=tree.body
    node=None
    for i,part in enumerate(parts):
        node=next((n for n in cur if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.name==part),None)
        if node is None: return None
        if i+1 < len(parts):
            if not isinstance(node,ast.ClassDef): return None
            cur=node.body
    return ast_sha(node)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--upstream-repo',required=True,type=Path,help='local git clone of visualbruno/ComfyUI-Trellis2')
    ap.add_argument('--base',help='override pinned base ref')
    ap.add_argument('--target',default='origin/main',help='new upstream ref to compare')
    ap.add_argument('--fetch',action='store_true',help='git fetch origin before checking')
    ap.add_argument('--fail-on-change',action='store_true',help='exit 2 when upstream review is required')
    ap.add_argument('--json-report',type=Path)
    args=ap.parse_args()
    data=json.loads(MANIFEST.read_text())
    base=args.base or data['upstream']['base_commit']
    repo=args.upstream_repo.expanduser().resolve()
    if not (repo/'.git').exists(): raise SystemExit(f'not a git clone: {repo}')
    if args.fetch: run_git(repo,'fetch','origin')
    # Ensure pinned commit exists locally.
    if subprocess.run(['git','-C',str(repo),'cat-file','-e',f'{base}^{{commit}}']).returncode:
        raise SystemExit(f'base commit is not present in upstream clone: {base}')
    target_sha=run_git(repo,'rev-parse',args.target).strip()
    ancestry=subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',base,target_sha])
    if ancestry.returncode != 0:
        raise SystemExit(f'target {target_sha} is not a descendant of pinned base {base}')
    findings=[]
    # File-level contract: especially important for upstream-verbatim files.
    for item in data['tracked_upstream_files']:
        path=item['path']; mode=item['mode']
        a=git_file(repo,base,path); b=git_file(repo,args.target,path)
        if a is None and b is None: status='ABSENT_BOTH'
        elif a is None: status='NEW_UPSTREAM_FILE'
        elif b is None: status='REMOVED_UPSTREAM_FILE'
        elif sha_text(a)==sha_text(b): status='UNCHANGED'
        else: status='CHANGED'
        if status!='UNCHANGED' and not (status=='ABSENT_BOTH' and mode=='derived_modified'):
            findings.append({'kind':'file','status':status,'path':path,'mode':mode})
    # Symbol-level contract for actively modified upstream-derived files.
    cache={}
    for item in data['tracked_symbols']:
        path=item['path']; symbol=item['symbol']
        if path not in cache:
            cache[path]=(git_file(repo,base,path),git_file(repo,args.target,path))
        old,new=cache[path]
        ah=find_symbol(old,symbol); bh=find_symbol(new,symbol)
        if ah is None and bh is None: status='ABSENT_BOTH'
        elif ah is None: status='NEW_COLLISION'
        elif bh is None: status='REMOVED'
        elif ah==bh: status='UNCHANGED'
        else: status='CHANGED'
        if status not in ('UNCHANGED','ABSENT_BOTH'):
            findings.append({'kind':'symbol','status':status,'path':path,'symbol':symbol})
        # Detect local drift since the manifest was generated.
        lp=ROOT/path
        if lp.exists():
            local=find_symbol(lp.read_text(),symbol)
            if local is None:
                findings.append({'kind':'local','status':'LOCAL_SYMBOL_MISSING','path':path,'symbol':symbol})
            elif local != item['local_ast_sha256']:
                findings.append({'kind':'local','status':'LOCAL_DRIFT','path':path,'symbol':symbol})
        else:
            findings.append({'kind':'local','status':'LOCAL_FILE_MISSING','path':path,'symbol':symbol})
    print(f'Upstream base:   {base}')
    print(f'Upstream target: {target_sha} ({args.target})')
    if findings:
        print(f'REVIEW REQUIRED: {len(findings)} item(s)')
        for x in findings:
            where=x['path'] + (f"::{x['symbol']}" if 'symbol' in x else '')
            print(f"  {x['status']:<22} {where}")
    else:
        print('No tracked upstream changes detected.')
    report={'base':base,'target_ref':args.target,'target_commit':target_sha,'findings':findings}
    if args.json_report:
        args.json_report.write_text(json.dumps(report,indent=2)+'\n')
    if findings and args.fail_on_change: return 2
    return 0

if __name__=='__main__':
    sys.exit(main())
