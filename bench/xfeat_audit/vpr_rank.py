#!/usr/bin/env python3
"""Rank the test frames against the tile descriptors built by vpr_recall.py."""
import math, os, sys
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay as R, fclog
sys.path.insert(0, "/data/DRONE/vps_vio")
sys.path.insert(0, "/data/DRONE/third_party/accelerated_features")
from map_cache import MapCache
from map_match_xfeat import _as_3ch, _equalise
from map_query import north_up_query
from modules.xfeat import XFeat

torch.set_num_threads(4); MDEG=111320.0
z=np.load("/data/DRONE/bench/xfeat_audit/vpr_tiles.npz")
cen,G,M=z["cen"],z["gem"],z["meanl2"]
cache=MapCache(os.path.realpath("/data/DRONE/maps/active.tif"))
model=XFeat(top_k=1024)
mlon=MDEG*math.cos(math.radians(float(cen[:,1].mean())))
print("%d tiles loaded (%s m, %s m step)"%(len(cen),z["tile_m"],z["step_m"]))

def gem(d,p=3.0):
    x=np.asarray(d,np.float32)
    x=np.clip(x,1e-6,None) if x.min()>=0 else np.abs(x)+1e-6
    v=np.power(np.mean(np.power(x,p),axis=0),1.0/p); n=np.linalg.norm(v)
    return v/n if n>1e-9 else None
def meanl2(d):
    v=np.asarray(d,np.float32).mean(axis=0); n=np.linalg.norm(v)
    return v/n if n>1e-9 else None

tests=[]
for run,bin_ in (("vps_00478",None),
                 ("vps_00486","/home/abc/Downloads/2026-11-05 08-57-56.bin")):
    if bin_: fclog.BIN[run]=bin_
    try: fclog.attitude(run,verbose=False)
    except Exception as e: print(" skip",run,e); continue
    fr=R.frames(run,alt_min=85,step=3)
    for f in fr:
        f["prior_err"]=R.metres(f["est_lon"],f["est_lat"],f["true_lon"],f["true_lat"])
        f["run"]=run
    for tag,sel,k in (("LOST",[f for f in fr if f["prior_err"]>=50],8),
                      ("GOOD",[f for f in fr if f["prior_err"]<12],8)):
        if not sel: continue
        for i in np.linspace(0,len(sel)-1,min(k,len(sel))).round().astype(int):
            tests.append((tag,sel[i]))
print("%d test frames\n"%len(tests))
res=[]
for tag,f in tests:
    img=cv2.imread(f["path"],cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    q,_=north_up_query(img,altitude_m=f["alt"],focal_px=R.CAM["fx"],
        yaw_deg=f["yaw"]+R.YAW_OFFSET,map_gsd_m=cache.gsd,
        map_gsd_x_m=getattr(cache,"gsd_x",None),shape="full",
        track_xy=(R.CAM["cx"],R.CAM["cy"]))
    d=model.detectAndCompute(_as_3ch(_equalise(q)),top_k=1024)[0]["descriptors"].cpu().numpy()
    qg,qm=gem(d),meanl2(d)
    dist=np.hypot((cen[:,1]-f["true_lat"])*MDEG,(cen[:,0]-f["true_lon"])*mlon)
    ti=int(np.argmin(dist))
    row=dict(tag=tag,run=f["run"],t=f["t"],alt=f["alt"],prior=f["prior_err"])
    for nm,Q,DB in (("gem",qg,G),("meanl2",qm,M)):
        o=np.argsort(-(DB@Q)); row[nm]=int(np.where(o==ti)[0][0])+1
        row[nm+"_top1_m"]=float(dist[o[0]])
    res.append(row)
    print("  %-4s %s t=%5.0f alt %5.1f prior %6.1f | sahi tile ka rank: gem %4d  meanl2 %4d | top1 %6.1f m door"%(
        tag,row["run"][-3:],row["t"],row["alt"],row["prior"],row["gem"],row["meanl2"],row["gem_top1_m"]))
print("\n"+"="*66+"\nrecall@K  (%d tiles)\n"%len(cen)+"="*66)
for tag in ("LOST","GOOD","ALL"):
    s=[r for r in res if tag=="ALL" or r["tag"]==tag]
    if not s: continue
    for nm in ("gem","meanl2"):
        rk=np.array([r[nm] for r in s])
        print("  %-5s n=%2d  %-7s "%(tag,len(s),nm)+"  ".join(
            "@%-2d %3.0f%%"%(k,100*np.mean(rk<=k)) for k in (1,5,20,50,100))
            +"  median rank %4.0f"%np.median(rk))
print("\n  chance: top 20 of %d = %.1f%%"%(len(cen),100*20/len(cen)))
