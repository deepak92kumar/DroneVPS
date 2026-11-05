"""The same frame from today, matched three ways, drawn the same way.

NCC correlates the whole template and reports a peak - it has no points, so
there is nothing to draw but the window it chose. XFeat and LighterGlue match
features, so every pair is a line: green where they agree on one shift, red
where the fit threw them out.
"""
import math, os, sys
import numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay as R
import fclog
sys.path.insert(0, "/data/DRONE/vps_vio")
sys.path.insert(0, "/data/DRONE/third_party/accelerated_features")
from map_cache import MapCache
from map_fix_worker import MapFixWorker
from map_match_xfeat import _as_3ch, _equalise, _translation
from map_match_ncc import locate as ncc_locate
from map_query import north_up_query
from modules.xfeat import XFeat

torch.set_num_threads(4)
MDEG=111320.0
RUN="vps_00486"
cache=MapCache(os.path.realpath("/data/DRONE/maps/active.tif"))
model=XFeat(top_k=1024)
fclog.BIN[RUN]="/home/abc/Downloads/2026-11-05 08-57-56.bin"
fclog.attitude(RUN)

fr=R.frames(RUN, alt_min=90, step=5)
for f in fr:
    f["prior_err"]=R.metres(f["est_lon"],f["est_lat"],f["true_lon"],f["true_lat"])
fr=[f for f in fr if f["prior_err"]<12]
f=fr[len(fr)//2]
img=cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
roll,pitch,_=fclog.at(RUN,f["t"])
dN,dE=MapFixWorker.nadir_offset_m(f["alt"],roll,pitch,f["yaw"])
q,qc=north_up_query(img,altitude_m=f["alt"],focal_px=R.CAM["fx"],
                    yaw_deg=f["yaw"]+R.YAW_OFFSET,map_gsd_m=cache.gsd,
                    map_gsd_x_m=getattr(cache,"gsd_x",None),shape="full",
                    track_xy=(R.CAM["cx"],R.CAM["cy"]))
qe=_as_3ch(_equalise(q)); qc=np.array(qc,float)

def feat(pairs):
    if pairs=="lg":
        d0=model.detectAndCompute(qe,top_k=1024)[0]
        d0.update({"image_size":(qe.shape[1],qe.shape[0])})
        d1=model.detectAndCompute(WE,top_k=1024)[0]
        d1.update({"image_size":(WE.shape[1],WE.shape[0])})
        m0,m1,_=model.match_lighterglue(d0,d1,min_conf=0.1)
        return np.asarray(m0,float),np.asarray(m1,float)
    o=model.match_xfeat(qe,WE,top_k=1024)
    return np.asarray(o[0],float),np.asarray(o[1],float)

panels=[]
# NCC needs its own, wider window; XFeat and LG share theirs.
for tag in ("ncc","xfeat","lg"):
    rad=float(np.clip(0.60*f["alt"],40,150))
    if tag=="ncc":
        rad=max(rad,0.5*1280*f["alt"]/R.CAM["fx"]+45.0)
    win,wg=cache.window(f["est_lon"],f["est_lat"],rad,kind="grey")
    WE=_as_3ch(_equalise(win))
    if tag=="ncc":
        res=ncc_locate(img,win,wg,altitude_m=f["alt"],focal_px=R.CAM["fx"],
                       yaw_deg=f["yaw"]+R.YAW_OFFSET,map_gsd_m=cache.gsd,
                       map_gsd_x_m=getattr(cache,"gsd_x",None),
                       min_score=0.05,min_margin=0.01,
                       principal_xy=(R.CAM["cx"],R.CAM["cy"]),query_shape="full")
        lat=res.lat-dN/MDEG
        lon=res.lon-dE/(MDEG*math.cos(math.radians(res.lat or 30.9)))
        err=R.metres(lon,lat,f["true_lon"],f["true_lat"]) if res.lon else float("nan")
        panels.append(dict(tag="NCC  (correlation, koi point nahi)",win=win,
                           src=None,dst=None,keep=None,ok=res.ok,
                           n=0,n_in=0,err=err,rad=rad,score=res.score))
        continue
    src,dst=feat("lg" if tag=="lg" else "mnn")
    if len(src)>=3:
        off,keep,mad=_translation(src,dst,4.0)
        n_in=int(keep.sum()); mc=qc+off
        lon,lat=wg.to_lonlat(float(mc[0]),float(mc[1]))
        lat-=dN/MDEG; lon-=dE/(MDEG*math.cos(math.radians(lat)))
        err=R.metres(lon,lat,f["true_lon"],f["true_lat"])
    else:
        keep=np.zeros(0,bool); n_in=0; err=float("nan"); mad=4.0
    mi=20 if tag=="lg" else 12
    ratio=n_in/max(len(src),1)
    ok=(len(src)>=12 and n_in>=mi and ratio>=0.07
        and max(0.0,1-mad/4.0)>=0.20)
    panels.append(dict(tag=("XFeat + mutual NN" if tag=="xfeat"
                            else "XFeat + LighterGlue"),
                       win=win,src=src,dst=dst,keep=keep,ok=ok,n=len(src),
                       n_in=n_in,err=err,rad=rad,score=ratio))

fig,ax=plt.subplots(3,1,figsize=(13.5,13.0))
qg=cv2.cvtColor(qe,cv2.COLOR_BGR2GRAY)
for a,p in zip(ax,panels):
    wgi=p["win"]
    Hh=max(qg.shape[0],wgi.shape[0])
    canvas=np.zeros((Hh,qg.shape[1]+wgi.shape[1]),np.uint8)
    canvas[:qg.shape[0],:qg.shape[1]]=qg
    canvas[:wgi.shape[0],qg.shape[1]:]=wgi
    a.imshow(canvas,cmap="gray",interpolation="nearest")
    if p["src"] is not None and len(p["src"]):
        dx=p["dst"].copy(); dx[:,0]+=qg.shape[1]; k=p["keep"]
        for s_,d_ in zip(p["src"][~k][::2],dx[~k][::2]):
            a.plot([s_[0],d_[0]],[s_[1],d_[1]],color="#e53e3e",lw=0.3,alpha=0.4)
        for s_,d_ in zip(p["src"][k],dx[k]):
            a.plot([s_[0],d_[0]],[s_[1],d_[1]],color="#38a169",lw=0.6,alpha=0.95)
    a.axis("off")
    a.set_title("%s   |   map window %.0f m radius (%d px)   |   %s\n"
                "%s   error vs M9N: %s"%(
        p["tag"],p["rad"],wgi.shape[1],
        "FIX bana" if p["ok"] else "MANA kiya",
        ("%d matches, %d sehmat (ratio %.3f)"%(p["n"],p["n_in"],p["score"])
         if p["src"] is not None else "peak score %.3f"%p["score"]),
        "%.2f m"%p["err"] if p["err"]==p["err"] else "-"),fontsize=9.5)
fig.suptitle("ek hi frame (%s, alt %.0f m, 5 Nov) — teen matcher, teeno ka"
             " apna window aur apna tareeka"%(os.path.basename(f["path"]),f["alt"]),
             fontsize=11,y=0.998)
plt.tight_layout(rect=[0,0,1,0.985])
plt.savefig("/data/DRONE/bench/xfeat_audit/three_matchers_486.png",dpi=118)
print("wrote three_matchers_486.png  frame=%s alt=%.0f prior_err=%.1f"%(
    os.path.basename(f["path"]),f["alt"],f["prior_err"]))
for p in panels:
    print("  %-38s ok=%d  %4d matches %3d sehmat  err %s"%(
        p["tag"],p["ok"],p["n"],p["n_in"],
        "%.2f m"%p["err"] if p["err"]==p["err"] else "-"))
