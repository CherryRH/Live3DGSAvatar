#!/usr/bin/env python
"""安全地编辑自由层：容量受限的分裂 / 剪枝；硬表面 PLY 不变。"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch
REPO_ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(REPO_ROOT/'src'))
import live3dgsavatar  # noqa: F401

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--free',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',type=int,default=0,help='分裂前 N 个自由高斯')
    p.add_argument('--max-gaussians',type=int,default=24000)
    p.add_argument('--prune-min-opacity',type=float,default=0.005)
    p.add_argument('--max-offset',type=float,default=0.08)
    a=p.parse_args()
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply
    av=load_ply(a.model,config=AvatarConfig.from_yaml(a.model.parent/'config.yaml'),device='cpu')
    av.load_free_layer(a.free)
    n0=int(av.free_xyz.shape[0]); print('before',n0)
    if a.split>0:
        k=min(int(a.split), max(0, int(a.max_gaussians)-n0))
        if k>0:
            av.split_free_layer(torch.arange(k)); print('split',k)
    if a.prune_min_opacity>0:
        removed=av.prune_free_layer(min_opacity=a.prune_min_opacity,max_offset=a.max_offset)
        print('prune',removed)
    print('after',int(av.free_xyz.shape[0]))
    a.out.parent.mkdir(parents=True,exist_ok=True); av.save_free_layer(a.out)

if __name__=='__main__': main()
