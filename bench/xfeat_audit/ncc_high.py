"""Was NCC's suppressed answer any good above 85 m on the 16:25 flight?

The 16 s hole at 101.6 m was XFeat getting 0-1 inliers while NCC accepted
every attempt and was discarded for being above map_ncc_below_m. Whether that
was the right call is a measurement, not an opinion: score NCC on those very
frames against the M9N.
"""
import math, os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay as R, fclog
sys.path.insert(0, "/data/DRONE/vps_vio")
from map_cache import MapCache
from map_fix_worker import MapFixWorker
from map_match_ncc import locate as ncc_locate
from map_match_xfeat import locate as xf_locate

RUN="vps_00494"; fclog.BIN[RUN]="/home/abc/Downloads/2026-11-05 16-25-23.bin"
fclog.attitude(RUN, verbose=False)
MDEG=111320.0; FOC=R.CAM["fx"]
cache=MapCache(os.path.realpath("/data/DRONE/maps/active.tif"))
fr=R.frames(RUN, alt_min=85, step=2)
for f in fr:
    f["prior_err"]=R.metres(f["est_lon"],f["est_lat"],f["true_lon"],f["true_lat"])
hole=[f for f in fr if 222<=f["t"]<=243]
rest=[f for f in fr if not (222<=f["t"]<=243)]
sel=hole+[rest[i] for i in np.linspace(0,len(rest)-1,min(40,len(rest))).round().astype(int)]
print("%d frames (hole se %d, baaki se %d)"%(len(sel),len(hole),len(sel)-len(hole)))
rows=[]
for f in sel:
    img=cv2.imread(f["path"],cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    roll,pitch,_=fclog.at(RUN,f["t"])
    dN,dE=MapFixWorker.nadir_offset_m(f["alt"],roll,pitch,f["yaw"])
    rec=dict(t=f["t"],alt=f["alt"],prior=f["prior_err"],hole=(222<=f["t"]<=243))
    for nm,fn,rad in (("ncc",ncc_locate,max(0.60*f["alt"],0.5*1280*f["alt"]/FOC+45.0)),
                      ("xfeat",xf_locate,0.60*f["alt"])):
        try:
            win,wg=cache.window(f["est_lon"],f["est_lat"],float(np.clip(rad,40,150)),kind="grey")
            g={"ncc":dict(min_score=0.05,min_margin=0.01),
               "xfeat":dict(min_score=0.07,min_margin=0.20)}[nm]
            res=fn(img,win,wg,altitude_m=f["alt"],focal_px=FOC,
                   yaw_deg=f["yaw"]+R.YAW_OFFSET,map_gsd_m=cache.gsd,
                   map_gsd_x_m=getattr(cache,"gsd_x",None),
                   principal_xy=(R.CAM["cx"],R.CAM["cy"]),query_shape="full",**g)
        except Exception:
            rec[nm+"_ok"]=0; rec[nm+"_err"]=np.nan; continue
        lat=res.lat-dN/MDEG
        lon=res.lon-dE/(MDEG*math.cos(math.radians(res.lat or 30.9)))
        rec[nm+"_ok"]=int(res.ok)
        rec[nm+"_err"]=R.metres(lon,lat,f["true_lon"],f["true_lat"]) if res.lon else np.nan
    rows.append(rec)
def rep(nm,s):
    for m in ("ncc","xfeat"):
        a=[q[m+"_err"] for q in s if q.get(m+"_ok") and np.isfinite(q.get(m+"_err",np.nan))]
        print("  %-18s %-6s accept %3.0f%% | err p50 %6s p90 %7s WORST %7s | >20m %d"%(
            nm,m,100*len(a)/max(len(s),1),
            "%.2f"%np.median(a) if a else "-","%.2f"%np.percentile(a,90) if a else "-",
            "%.2f"%max(a) if a else "-",sum(1 for v in a if v>20)))
print()
rep("16 s ka HOLE",[q for q in rows if q["hole"]])
rep("baaki (>85 m)",[q for q in rows if not q["hole"]])
rep("sab",rows)
