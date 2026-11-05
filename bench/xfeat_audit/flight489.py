"""5 Nov 10:57 flight (vps_00489): why the tracker kept dropping out, and
the map with M9N / VPS / flow-only on it."""
import csv, math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/DRONE/vps_vio")
from map_cache import MapCache
plt.rcParams.update({"figure.dpi":130,"font.size":9,"axes.grid":True,"grid.alpha":.25})
MDEG=111320.0
P='/home/abc/NVME/DRONE/flight_logs/vps/vps_00489/vps_00489.csv'
r=list(csv.DictReader(open(P)))
F=lambda x,k:(float(x[k]) if x.get(k) not in (None,'') else np.nan)
g=lambda k: np.array([F(x,k) for x in r])
t,alt,tr=g('t'),g('alt_m'),g('tracks')
n,e=g('north_m'),g('east_m'); gn,ge=g('gps1_north_m'),g('gps1_east_m')
lat,lon=g('lat'),g('lon'); fl,fa=g('fix_lon'),g('fix_lat')
dn,de,dt_=g('step_dn'),g('step_de'),g('step_dt_s')
dual,dis=g('flow_dual'),g('flow_disagree_px')
dd,fage=g('depth_dropped'),g('frame_age_s')
used=np.array([x.get('fix_used')=='1' for x in r])
src=[x.get('fix_source','') for x in r]
act=[x.get('step_action','') for x in r]
vr=[x.get('vel_reason','') for x in r]
ok=np.isfinite(gn)&np.isfinite(ge)&(np.abs(gn)<400)&(np.abs(ge)<400)&np.isfinite(n)
err=np.where(ok,np.hypot(n-gn,e-ge),np.nan)
fi=np.where(used)[0]; leg=(t>=t[fi[0]])&(t<=t[fi[-1]])

print("=== vps_00489, 10:57, %.0f s, alt max %.1f m ==="%(t[-1]-t[0],np.nanmax(alt)))
print("VPS leg: t=%.0f .. %.0f"%(t[fi[0]],t[fi[-1]]))
print()
low=(tr<30)&np.isfinite(tr); zero=(tr==0)
# episodes = edges
ed=np.where(np.diff(low.astype(int))==1)[0]
print("TRACKER: tracks<30 par %d samples (%.2f%%), tracks==0 par %d"%(
    low.sum(),100*low.mean(),zero.sum()))
print("  alag-alag episodes (jitni baar 'TRACK LOST' bola): %d"%len(ed))
print("  tracks: p50 %.0f  p10 %.0f  min %.0f"%(
    np.nanmedian(tr),np.nanpercentile(tr[np.isfinite(tr)],10),np.nanmin(tr)))
print()
print("  episodes (pehle 12):   t      alt   tracks  dual  disagree  depth_drop  frame_age")
for i in ed[:12]:
    j=min(i+1,len(t)-1)
    print("    %7.1f  %5.1f   %4.0f    %s    %6s     %6s     %.4f"%(
        t[j],alt[j],tr[j],
        "-" if not np.isfinite(dual[j]) else "%d"%dual[j],
        "-" if not np.isfinite(dis[j]) else "%.2f"%dis[j],
        "-" if not np.isfinite(dd[j]) else "%.0f"%dd[j],
        0 if not np.isfinite(fage[j]) else fage[j]))
print()
print("  track loss kis altitude par:")
for lo,hi in ((0,40),(40,60),(60,80),(80,100)):
    m=(alt>=lo)&(alt<hi)&np.isfinite(tr)
    if m.sum()<50: continue
    print("    alt %3d-%3d m n=%5d | tracks p50 %4.0f | <30 par %5.2f%% | dual %3.0f%% | disagree p50 %s"%(
        lo,hi,m.sum(),np.nanmedian(tr[m]),100*np.mean(tr[m]<30),
        100*np.nanmean(dual[m]) if np.isfinite(dual[m]).any() else -1,
        "%.2f"%np.nanmedian(dis[m]) if np.isfinite(dis[m]).any() else "-"))
print()
import collections
print("  step_action ki ginti:",dict(collections.Counter(a for a in act if a).most_common(6)))
print("  vel_reason ki ginti:",dict(collections.Counter(v for v in vr if v).most_common(6)))
print()
print("ERROR vs M9N, phase ke hisab se:")
for lo in range(int(t[fi[0]]//50*50),int(t[fi[-1]])+1,50):
    m=(t>=lo)&(t<lo+50)&np.isfinite(err)
    if m.sum()<10: continue
    print("  t %3d-%3d alt %5.1f | err p50 %6.2f MAX %6.2f | tracks p50 %4.0f | fixes %d"%(
        lo,lo+50,np.nanmedian(alt[(t>=lo)&(t<lo+50)]),np.nanmedian(err[m]),
        np.nanmax(err[m]),np.nanmedian(tr[(t>=lo)&(t<lo+50)]),
        int(used[(t>=lo)&(t<lo+50)].sum())))
e2=err[leg&np.isfinite(err)]
print("poori leg: err p50 %.2f p90 %.2f MAX %.2f m | fixes %d"%(
    np.median(e2),np.percentile(e2,90),e2.max(),used[leg].sum()))
print()
# published fix accuracy
i0=np.where(ok&np.isfinite(lat))[0][0]
mlon=MDEG*math.cos(math.radians(lat[i0])); olat=lat[i0]-n[i0]/MDEG; olon=lon[i0]-e[i0]/mlon
gi=np.where(ok)[0]; rows=[]
for i in np.where(used&np.isfinite(fa))[0]:
    j=gi[np.argmin(np.abs(t[gi]-t[i]))]
    if abs(t[j]-t[i])>1.0: continue
    rows.append((src[i],math.hypot((fa[i]-olat)*MDEG-gn[j],(fl[i]-olon)*mlon-ge[j])))
print("PUBLISH hue fix, M9N ke khilaf:")
for s in ("xfeat","ncc","xfeat_lg"):
    v=[q[1] for q in rows if q[0]==s]
    if v: print("  %-9s n=%3d p50 %5.2f p90 %6.2f WORST %6.2f | >20m %d"%(
        s,len(v),np.median(v),np.percentile(v,90),max(v),sum(1 for x in v if x>20)))
av=[q[1] for q in rows]
print("  %-9s n=%3d p50 %5.2f p90 %6.2f WORST %6.2f | >20m %d"%(
    "SAB",len(av),np.median(av),np.percentile(av,90),max(av),sum(1 for x in av if x>20)))

# ---- flow-only (VIO) reconstruction: step_dn/de integrate, no map
fn=np.full(len(t),np.nan); fe=np.full(len(t),np.nan)
cn=ce=0.0; started=False
for i in range(len(t)):
    if not started and leg[i]:
        cn,ce=n[i],e[i]; started=True
    if started and np.isfinite(dn[i]) and np.isfinite(de[i]):
        cn+=dn[i]; ce+=de[i]
    if started: fn[i],fe[i]=cn,ce
ferr=np.where(ok&np.isfinite(fn),np.hypot(fn-gn,fe-ge),np.nan)
print()
print("VIO (sirf flow, map ke bina) vs VPS (flow+map):")
print("  VIO  err p50 %.2f  MAX %.2f m"%(np.nanmedian(ferr[leg]),np.nanmax(ferr[leg])))
print("  VPS  err p50 %.2f  MAX %.2f m"%(np.median(e2),e2.max()))

# ---- map figure
span=float(max(np.nanmax(np.abs(n[leg])),np.nanmax(np.abs(e[leg])),
    np.nanmax(np.abs(gn[leg&ok])),np.nanmax(np.abs(ge[leg&ok])),
    np.nanmax(np.abs(fn[leg])),np.nanmax(np.abs(fe[leg])))+70)
cN,cE=np.nanmean(n[leg]),np.nanmean(e[leg])
cache=MapCache(os.path.realpath('/data/DRONE/maps/active.tif'))
win,wg=cache.window(olon+cE/mlon,olat+cN/MDEG,span,kind="grey")
H,W=win.shape[:2]
px=lambda N,E:(W/2+(E-cE)/(2*span/W), H/2-(N-cN)/(2*span/H))
fig=plt.figure(figsize=(14.6,6.6))
a0=fig.add_subplot(1,2,1); a1=fig.add_subplot(2,2,2); a2=fig.add_subplot(2,2,4)
a0.imshow(win,cmap="gray",interpolation="nearest")
for arr,col,lbl,lw in ((("gps"),"#2f855a","M9N / GPS (sach)",2.0),
                        (("vps"),"#2b6cb0","VPS (flow + map)",1.4),
                        (("vio"),"#d69e2e","VIO (sirf flow, map nahi)",1.2)):
    if arr=="gps": x,y=px(gn[leg&ok],ge[leg&ok])
    elif arr=="vps": x,y=px(n[leg],e[leg])
    else: x,y=px(fn[leg],fe[leg])
    a0.plot(x,y,"-",color=col,lw=lw,label=lbl)
for s,c in (("xfeat","#805ad5"),("ncc","#c05621"),("xfeat_lg","#b83280")):
    m=leg&used&np.array([q==s for q in src])&np.isfinite(fa)
    if not m.any(): continue
    x,y=px((fa[m]-olat)*MDEG,(fl[m]-olon)*mlon)
    a0.scatter(x,y,s=24,color=c,zorder=6,edgecolor="white",lw=.5,
               label="%s fix (%d)"%(s,m.sum()))
mlow=leg&low
if mlow.any():
    x,y=px(n[mlow],e[mlow])
    a0.scatter(x,y,s=70,facecolor="none",edgecolor="#e53e3e",lw=1.4,zorder=7,
               label="tracks<30 (%d)"%mlow.sum())
a0.set_xlim(0,W); a0.set_ylim(H,0); a0.axis("off")
a0.legend(fontsize=7.5,loc="upper left",framealpha=.9)
a0.set_title("vps_00489 — satellite sheet, %.0f x %.0f m"%(2*span,2*span),fontsize=9.5)
a1.plot(t[leg],err[leg],"-",color="#2b6cb0",lw=1.3,label="VPS error")
a1.plot(t[leg],ferr[leg],"-",color="#d69e2e",lw=1.1,label="VIO error (map nahi)")
a1b=a1.twinx(); a1b.plot(t[leg],alt[leg],"-",color="0.65",lw=.9); a1b.set_ylabel("alt",color="0.5",fontsize=8)
a1.legend(fontsize=8); a1.set_ylabel("error vs M9N (m)"); a1.set_title("map kitna bachata hai",fontsize=9)
a2.plot(t[leg],tr[leg],"-",color="#2f855a",lw=1.0)
a2.axhline(30,color="#e53e3e",ls="--",lw=1.2)
a2.text(t[leg][0],33,"30 = TRACK LOST ki hadd",fontsize=7.5,color="#e53e3e")
a2.set_xlabel("t (s)"); a2.set_ylabel("tracks"); a2.set_title("tracker ki haalat",fontsize=9)
plt.tight_layout(); plt.savefig('/data/DRONE/bench/xfeat_audit/flight489_map.png',dpi=125)
print("\nwrote flight489_map.png")
