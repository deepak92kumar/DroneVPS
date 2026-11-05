"""5 Nov 16:25 flight (vps_00494): mostly GPS, 97 s on VPS at 102 m.

Where was the VPS while the GPS flew, what happened at the handover, and was
the matcher asleep when it mattered.
"""
import csv, math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay as R, fclog
sys.path.insert(0, "/data/DRONE/vps_vio")
from map_cache import MapCache
plt.rcParams.update({"figure.dpi":130,"font.size":9,"axes.grid":True,"grid.alpha":.25})
MDEG=111320.0
RUN="vps_00494"; BIN="/home/abc/Downloads/2026-11-05 16-25-23.bin"
fclog.BIN[RUN]=BIN
att,info=fclog.attitude(RUN)       # raises if yaw and alt disagree
OFF=info["off"]                    # node_t = fc_t + OFF
SW_ON, SW_OFF = 347.1+OFF, 444.1+OFF
print("handover (node clock): VPS t=%.1f .. %.1f  (%.0f s)"%(SW_ON,SW_OFF,SW_OFF-SW_ON))

P=f'/home/abc/NVME/DRONE/flight_logs/vps/{RUN}/{RUN}.csv'
r=list(csv.DictReader(open(P)))
F=lambda x,k:(float(x[k]) if x.get(k) not in (None,'') else np.nan)
g=lambda k: np.array([F(x,k) for x in r])
t,alt=g('t'),g('alt_m'); n,e=g('north_m'),g('east_m')
gn,ge=g('gps1_north_m'),g('gps1_east_m')
lat,lon=g('lat'),g('lon'); fl,fa=g('fix_lon'),g('fix_lat')
sg,trk=g('sigma_m'),g('tracks'); fage=g('frame_age_s')
used=np.array([x.get('fix_used')=='1' for x in r])
src=[x.get('fix_source','') for x in r]
lost=np.array([x.get('lost')=='1' for x in r])
lr=[x.get('lost_reason','') for x in r]
ok=np.isfinite(gn)&np.isfinite(ge)&(np.abs(gn)<400)&(np.abs(ge)<400)&np.isfinite(n)
err=np.where(ok,np.hypot(n-gn,e-ge),np.nan)
vps=(t>=SW_ON)&(t<=SW_OFF)

print("\n=== 1. GPS par udte waqt VPS kahan tha ===")
for lo,hi,lbl in ((0,100,"zameen"),(100,200,"climb"),(200,300,"102 m GPS"),
                  (300,int(SW_ON),"102 m GPS, switch se pehle")):
    m=(t>=lo)&(t<hi)&np.isfinite(err)
    if m.sum()<10: continue
    print("  t %3d-%3d %-26s err p50 %7.2f  MAX %7.2f | sigma %5.2f | fixes %d"%(
        lo,hi,lbl,np.nanmedian(err[m]),np.nanmax(err[m]),
        np.nanmedian(sg[(t>=lo)&(t<hi)]),int(used[(t>=lo)&(t<hi)].sum())))

print("\n=== 2. handover ke aas-paas (t=%.0f) ==="%SW_ON)
for dt in (-20,-10,-5,-2,0,2,5,10,20,40,60,80,97):
    i=int(np.argmin(np.abs(t-(SW_ON+dt))))
    print("  t%+4d (%6.1f)  err %7.2f m  N %8.2f E %8.2f | M9N N %8.2f E %8.2f | sigma %5.2f lost %d"%(
        dt,t[i],err[i],n[i],e[i],gn[i],ge[i],sg[i],int(lost[i])))

print("\n=== 3. matcher so raha tha? ===")
fi=np.where(used)[0]
pre=[t[i] for i in fi if t[i]<SW_ON]; dur=[t[i] for i in fi if SW_ON<=t[i]<=SW_OFF]
print("  switch se pehle %d fix (aakhri t=%.1f, switch se %.1f s pehle)"%(
    len(pre),pre[-1] if pre else -1,SW_ON-(pre[-1] if pre else SW_ON)))
print("  VPS leg me %d fix | pehla t=%.1f (switch ke %.1f s baad)"%(
    len(dur),dur[0] if dur else -1,(dur[0]-SW_ON) if dur else -1))
if len(dur)>1: print("  VPS leg me fix ka antaral p50 %.2f s MAX %.2f s"%(
    np.median(np.diff(dur)),np.max(np.diff(dur))))
import collections
print("  VPS leg me kis matcher ne diya:",dict(collections.Counter(
    src[i] for i in fi if SW_ON<=t[i]<=SW_OFF)))
print("  VPS leg me koshish (fix_score likha):",sum(
    1 for i,x in enumerate(r) if SW_ON<=t[i]<=SW_OFF and x.get('fix_score') not in (None,'')))

print("\n=== 4. lost detector ===")
print("  lost=1 par %d samples; pehla t=%.1f"%(lost.sum(),t[lost][0] if lost.any() else -1))
lrs=set(x for x in lr if x)
for s_ in list(lrs)[:3]: print("   reason:",s_)
print("  tracks p50 %.0f | <30 par %.1f%% | frame_age p50 %.4f | 0.28 se upar %.1f%%"%(
    np.nanmedian(trk),100*np.mean(trk[np.isfinite(trk)]<30),
    np.nanmedian(fage[np.isfinite(fage)]),
    100*np.mean(fage[np.isfinite(fage)]>0.28)))

print("\n=== 5. VPS leg ka saar ===")
m=vps&np.isfinite(err)
print("  err p50 %.2f  p90 %.2f  MAX %.2f m  (97 s me)"%(
    np.nanmedian(err[m]),np.nanpercentile(err[m],90),np.nanmax(err[m])))
print("  drift rate: %.3f m/s"%(np.polyfit(t[m],err[m],1)[0]))

# ---- figure
i0=np.where(ok&np.isfinite(lat))[0][0]
mlon=MDEG*math.cos(math.radians(lat[i0])); olat=lat[i0]-n[i0]/MDEG; olon=lon[i0]-e[i0]/mlon
fly=(t>=100)&(t<=500)
span=float(max(np.nanmax(np.abs(n[fly])),np.nanmax(np.abs(e[fly])),
    np.nanmax(np.abs(gn[fly&ok])),np.nanmax(np.abs(ge[fly&ok])))+80)
cN,cE=np.nanmean(gn[fly&ok]),np.nanmean(ge[fly&ok])
cache=MapCache(os.path.realpath('/data/DRONE/maps/active.tif'))
win,wg=cache.window(olon+cE/mlon,olat+cN/MDEG,span,kind="grey")
H,W=win.shape[:2]
px=lambda N,E:(W/2+(E-cE)/(2*span/W), H/2-(N-cN)/(2*span/H))
fig=plt.figure(figsize=(14.6,6.8))
a0=fig.add_subplot(1,2,1); a1=fig.add_subplot(2,2,2); a2=fig.add_subplot(2,2,4)
a0.imshow(win,cmap="gray",interpolation="nearest")
x,y=px(gn[fly&ok],ge[fly&ok]); a0.plot(x,y,"-",color="#2f855a",lw=2.0,label="M9N / GPS (sach)")
x,y=px(n[fly&~vps[fly]] if False else n[(t>=100)&(t<SW_ON)],e[(t>=100)&(t<SW_ON)])
a0.plot(x,y,"-",color="#718096",lw=1.1,label="VPS jab GPS uda raha tha")
x,y=px(n[vps],e[vps]); a0.plot(x,y,"-",color="#2b6cb0",lw=2.0,label="VPS jab VPS uda raha tha (97 s)")
for s_,c in (("xfeat","#805ad5"),("ncc","#c05621"),("xfeat_lg","#b83280")):
    m2=used&np.array([q==s_ for q in src])&np.isfinite(fa)&fly
    if not m2.any(): continue
    x,y=px((fa[m2]-olat)*MDEG,(fl[m2]-olon)*mlon)
    a0.scatter(x,y,s=20,color=c,zorder=6,edgecolor="white",lw=.4,label="%s (%d)"%(s_,m2.sum()))
i=int(np.argmin(np.abs(t-SW_ON))); x,y=px(n[i],e[i])
a0.scatter([x],[y],s=180,marker="*",color="#f6e05e",edgecolor="black",lw=.8,zorder=9,label="switch ON")
j=int(np.argmin(np.abs(t-SW_OFF))); x,y=px(n[j],e[j])
a0.scatter([x],[y],s=150,marker="X",color="#e53e3e",edgecolor="white",lw=.8,zorder=9,label="switch OFF")
a0.set_xlim(0,W); a0.set_ylim(H,0); a0.axis("off"); a0.legend(fontsize=7.5,loc="upper left",framealpha=.9)
a0.set_title("vps_00494 — %.0f x %.0f m"%(2*span,2*span),fontsize=9.5)
a1.plot(t[fly],err[fly],"-",color="#2b6cb0",lw=1.3)
a1.axvspan(SW_ON,SW_OFF,color="#2b6cb0",alpha=.12)
a1.text(SW_ON+2,a1.get_ylim()[1]*.85,"VPS uda raha tha",fontsize=8,color="#2b6cb0")
a1.axhline(25,color="#b83280",ls="--",lw=1); a1.set_ylabel("error vs M9N (m)")
a1.set_title("error — VPS leg se PEHLE hi badha hua tha",fontsize=9)
a2.plot(t[fly],sg[fly],"-",color="#2f855a",lw=1.2,label="declared sigma")
a2.plot(t[fly],err[fly],"-",color="#2b6cb0",lw=.9,alpha=.6,label="asli error")
a2.axvspan(SW_ON,SW_OFF,color="#2b6cb0",alpha=.12)
a2.set_xlabel("t (s)"); a2.set_ylabel("m"); a2.legend(fontsize=8)
a2.set_title("sigma vs sach",fontsize=9)
plt.tight_layout(); plt.savefig('/data/DRONE/bench/xfeat_audit/flight494_map.png',dpi=125)
print("\nwrote flight494_map.png")
