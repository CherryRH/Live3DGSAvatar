#!/usr/bin/env python
"""把硬表面 + 自由层合并成单个统一高斯集合（初始化/加密前原型）。"""
from __future__ import annotations
import argparse, sys, math
from pathlib import Path
import numpy as np, torch
REPO_ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(REPO_ROOT/'src'))
import live3dgsavatar  # noqa: F401


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--hard',type=Path,required=True)
    p.add_argument('--free',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tex-size',type=int,default=384)
    a=p.parse_args()
    from dataclasses import replace
    from live3dgsavatar.core.avatar import AvatarConfig, GaussianAvatar
    from live3dgsavatar.core.io import load_ply, save_ply
    from live3dgsavatar.core.deform import Binding
    cfg=AvatarConfig.from_yaml(a.hard.parent/'config.yaml')
    hard=load_ply(a.hard,config=cfg,device='cpu')
    d=np.load(a.free)
    f_xyz=torch.from_numpy(d['xyz'].astype('float32'))
    f_rot=torch.from_numpy(d['rotation'].astype('float32'))
    f_scale=torch.from_numpy(d['scaling'].astype('float32'))
    f_op=torch.from_numpy(d['opacity'].astype('float32'))
    f_col=torch.from_numpy(d['color'].astype('float32'))
    fid=torch.from_numpy(d['face_id'].astype('int64'))
    fbary=torch.from_numpy(d['face_bary'].astype('float32'))
    n=int(hard.num_gaussians); m=int(f_xyz.shape[0]); total=n+m
    if a.tex_size*a.tex_size < total:
        raise SystemExit(f'tex_size {a.tex_size} 容量不足：{total}')
    new_cfg=replace(cfg, tex_size=int(a.tex_size))
    binding=Binding(face_id=torch.cat([hard.binding_face_id,fid]),
                    face_bary=torch.cat([hard.binding_face_bary,fbary]),
                    valid_mask=torch.ones(total,dtype=torch.bool))
    av=GaussianAvatar(new_cfg,binding).to('cpu')
    k=hard.num_basis
    with torch.no_grad():
        av.xyz.copy_(torch.cat([hard.xyz,f_xyz]))
        av.opacity.copy_(torch.cat([hard.opacity,f_op]))
        av.scaling.copy_(torch.cat([hard.scaling,f_scale]))
        av.rotation.copy_(torch.cat([hard.rotation,f_rot]))
        av.color.copy_(torch.cat([hard.color,f_col]))
        av.xyz_b.copy_(torch.cat([hard.xyz_b,torch.zeros(k,m,3)],dim=1))
        av.rotation_b.copy_(torch.cat([hard.rotation_b,torch.zeros(k,m,4)],dim=1))
        av.color_b.copy_(torch.cat([hard.color_b,torch.zeros(k,m,1,3)],dim=1))
        av.weight_module.load_state_dict(hard.weight_module.state_dict())
    a.out.parent.mkdir(parents=True,exist_ok=True)
    save_ply(av,a.out)
    print(f'merged {n}+{m}={total} -> {a.out} tex={a.tex_size}')
if __name__=='__main__': main()
