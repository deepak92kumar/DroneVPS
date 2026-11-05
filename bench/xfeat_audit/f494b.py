"""flight 494 again, with the origin handled PER ROW.

north_m/east_m and gps1_north_m/gps1_east_m are both measured from the node's
CURRENT origin, and the origin is re-anchored at the handover. Plotting them
across that change puts two coordinate frames on one picture, which is what
made one leg look detached. Every row's own origin is recovered from its
lat/lon against its north/east, exactly as replay.frames does.
"""
import csv, math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/DRONE/vps_vio")
from map_cache import MapCache
plt.rcParams.update({"figure.dpi":130,"font.size":9,"axes.grid":True,"grid.alpha":.25})
MDEG=111320.0
P='/home/abc/NVME/DRONE/flight_logs/vps/vps_00494/vps_00494.csv'
SW_ON,SW_OFF=280.1,377.1
r=list(csv.DictReader(open(P)))
F=lambda x,k:(float(x[k]) if x.get(k) not in (None,'') else np.nan)
g=lambda k: np.array([F(x,k) for x in r])
t,alt=g('t'),g('alt_m'); n,e=g('north_m'),g('east_m')
gn,ge=g('gps1_north_m'),g('gps1_east_m'); lat,lon=g('lat'),g('lon')
fl,fa=g('fix_lon'),g('fix_lat'); sg=g('sigma_m')
used=np.array([x.get('fix_used')=='1' for x in r])
src=[x.get('fix_source','') for x in r]

# ---- per-row origin -> absolute lon/lat for BOTH tracks
ml=MDEG*np.cos(np.radians(lat))
olat=lat-n/MDEG; olon=lon-e/ml
vlat,vlon=lat,lon                                  # VPS absolute, direct
g1lat=olat+gn/MDEG; g1lon=olon+ge/ml               # M9N absolute
ok=(np.isfinite(g1lat)&np.isfinite(g1lon)&np.isfinite(vlat)
    &(np.abs(gn)<400)&(np.abs(ge)<400))
# one fixed reference for metres: the M9N's own median
rlat=np.nanmedian(g1lat[ok]); rlon=np.nanmedian(g1lon[ok])
rml=MDEG*math.cos(math.radians(rlat))
to=lambda la,lo:((la-rlat)*MDEG,(lo-rlon)*rml)
VN,VE=to(vlat,vlon); GN,GE=to(g1lat,g1lon)
err=np.where(ok,np.hypot(VN-GN,VE-GE),np.nan)
fly=(alt>5)&np.isfinite(VN)

print("=== origin kab badla (per-row se pakda) ===")
d=np.abs(np.diff(olat))*MDEG+np.abs(np.diff(olon))*rml
for i in np.where(d>5)[0]:
    print("  t=%.2f par origin %.1f m hila"%(t[i+1],d[i]))
print()
print("=== ASLI movement (absolute frame me) ===")
print("  M9N:  north %.1f .. %.1f m   east %.1f .. %.1f m   = %.0f x %.0f m"%(
    np.nanmin(GN[ok&fly]),np.nanmax(GN[ok&fly]),np.nanmin(GE[ok&fly]),
    np.nanmax(GE[ok&fly]),np.nanmax(GN[ok&fly])-np.nanmin(GN[ok&fly]),
    np.nanmax(GE[ok&fly])-np.nanmin(GE[ok&fly])))
print("  switch ON  par: M9N north %.1f east %.1f"%to(g1lat[np.argmin(np.abs(t-SW_ON))],
                                                      g1lon[np.argmin(np.abs(t-SW_ON))]))
print("  switch OFF par: M9N north %.1f east %.1f"%to(g1lat[np.argmin(np.abs(t-SW_OFF))],
                                                      g1lon[np.argmin(np.abs(t-SW_OFF))]))
i0=np.argmin(np.abs(t-SW_ON)); i1=np.argmin(np.abs(t-SW_OFF))
print("  dono ke beech ki doori: %.1f m"%math.hypot(GN[i1]-GN[i0],GE[i1]-GE[i0]))
print()
print("=== error, absolute frame me ===")
for lo,hi,lbl in ((100,200,"climb, GPS"),(200,280,"102 m, GPS"),
                  (280,377,"102 m, VPS uda raha"),(377,470,"102 m, wapas GPS")):
    m=(t>=lo)&(t<hi)&np.isfinite(err)
    if m.sum()<10: continue
    print("  t %3d-%3d %-22s err p50 %6.2f  p90 %6.2f  MAX %6.2f | fix %d"%(
        lo,hi,lbl,np.nanmedian(err[m]),np.nanpercentile(err[m],90),
        np.nanmax(err[m]),int(used[(t>=lo)&(t<hi)].sum())))
print()
print("=== fix ke beech ke sabse bade gap (udte waqt) ===")
fi=np.where(used&fly)[0]
gaps=[(t[fi[k]]-t[fi[k-1]],t[fi[k-1]],t[fi[k]]) for k in range(1,len(fi))]
for gp,a,b in sorted(gaps,reverse=True)[:5]:
    m=(t>=a)&(t<=b)&np.isfinite(err)
    mv=math.hypot(GN[np.argmin(np.abs(t-b))]-GN[np.argmin(np.abs(t-a))],
                  GE[np.argmin(np.abs(t-b))]-GE[np.argmin(np.abs(t-a))])
    print("  %.1f s koi fix nahi (t=%.0f..%.0f) | alt %5.1f | drone %5.1f m chala | err %5.2f -> %5.2f"%(
        gp,a,b,np.nanmedian(alt[(t>=a)&(t<=b)]),mv,
        err[np.argmin(np.abs(t-a))],err[np.argmin(np.abs(t-b))]))
print()
print("=== aage-peeche (VPS leg, 280-377) ===")
m=(t>=SW_ON)&(t<=SW_OFF)&np.isfinite(err)
tt=t[m]; ee=err[m]-np.mean(err[m])
f=np.fft.rfftfreq(len(ee),np.median(np.diff(tt))); A=np.abs(np.fft.rfft(ee))
k=[i for i in np.argsort(A)[::-1][:6] if f[i]>0.005][:3]
print("  error peak-to-peak %.2f m | top period: %s"%(
    ee.max()-ee.min(),", ".join("%.0f s"%(1/f[i]) for i in k)))
mv=np.hypot(np.diff(GN[m]),np.diff(GE[m]))
print("  M9N: drone ne %.1f m ka net safar kiya, path length %.1f m"%(
    math.hypot(GN[m][-1]-GN[m][0],GE[m][-1]-GE[m][0]),np.nansum(mv)))

# ---- figure, cropped to the real movement
pad=35.0
lo_n,hi_n=np.nanmin(GN[ok&fly])-pad,np.nanmax(GN[ok&fly])+pad
lo_e,hi_e=np.nanmin(GE[ok&fly])-pad,np.nanmax(GE[ok&fly])+pad
cN,cE=(lo_n+hi_n)/2,(lo_e+hi_e)/2
span=max(hi_n-lo_n,hi_e-lo_e)/2
cache=MapCache(os.path.realpath('/data/DRONE/maps/active.tif'))
win,wg=cache.window(rlon+cE/rml,rlat+cN/MDEG,float(span),kind="grey")
H,W=win.shape[:2]
px=lambda N,E:(W/2+(E-cE)/(2*span/W), H/2-(N-cN)/(2*span/H))
fig=plt.figure(figsize=(14.8,7.0))
a0=fig.add_subplot(1,2,1); a1=fig.add_subplot(2,2,2); a2=fig.add_subplot(2,2,4)
a0.imshow(win,cmap="gray",interpolation="nearest")
m=ok&fly; x,y=px(GN[m],GE[m]); a0.plot(x,y,"-",color="#2f855a",lw=2.2,label="M9N / GPS (sach)")
mg=fly&(t<SW_ON); x,y=px(VN[mg],VE[mg]); a0.plot(x,y,"-",color="#718096",lw=1.2,label="VPS (GPS uda raha tha)")
mv_=fly&(t>=SW_ON)&(t<=SW_OFF); x,y=px(VN[mv_],VE[mv_]); a0.plot(x,y,"-",color="#2b6cb0",lw=2.2,label="VPS (VPS uda raha tha, 97 s)")
ma=fly&(t>SW_OFF); x,y=px(VN[ma],VE[ma]); a0.plot(x,y,"-",color="#a0aec0",lw=1.2,label="VPS (wapas GPS)")
for s_,c in (("xfeat","#805ad5"),("ncc","#c05621"),("xfeat_lg","#b83280")):
    mm=used&np.array([q==s_ for q in src])&np.isfinite(fa)&fly
    if not mm.any(): continue
    FN,FE=to(fa[mm],fl[mm]); x,y=px(FN,FE)
    a0.scatter(x,y,s=18,color=c,zorder=6,edgecolor="white",lw=.3,label="%s (%d)"%(s_,mm.sum()))
for i,mk,c,lb in ((i0,"*","#f6e05e","switch ON"),(i1,"X","#e53e3e","switch OFF")):
    x,y=px(GN[i],GE[i]); a0.scatter([x],[y],s=200,marker=mk,color=c,edgecolor="black",lw=.8,zorder=9,label=lb)
a0.set_xlim(0,W); a0.set_ylim(H,0); a0.axis("off")
a0.legend(fontsize=7.5,loc="best",framealpha=.9)
a0.set_title("vps_00494 — sirf %.0f x %.0f m (asli movement)"%(2*span,2*span),fontsize=9.5)
a1.plot(t[fly],err[fly],"-",color="#2b6cb0",lw=1.3)
a1.axvspan(SW_ON,SW_OFF,color="#2b6cb0",alpha=.13)
a1.text(SW_ON+2,np.nanmax(err[fly])*.8,"VPS uda raha tha",fontsize=8,color="#2b6cb0")
for gp,a,b in sorted(gaps,reverse=True)[:3]:
    a1.axvspan(a,b,color="#e53e3e",alpha=.18)
a1.set_ylabel("error vs M9N (m)"); a1.set_title("error — laal patti = fix ka gap",fontsize=9)
a2.plot(t[fly],alt[fly],"-",color="0.5",lw=1.0,label="alt")
a2b=a2.twinx()
a2b.plot(t[used&fly],np.ones(int((used&fly).sum())),"|",color="#805ad5",ms=9)
a2b.set_yticks([]); a2b.set_ylabel("fix",color="#805ad5",fontsize=8)
a2.axvspan(SW_ON,SW_OFF,color="#2b6cb0",alpha=.13)
a2.set_xlabel("t (s)"); a2.set_ylabel("alt (m)"); a2.legend(fontsize=8,loc="lower left")
a2.set_title("altitude aur har fix (baingani lakeer)",fontsize=9)
plt.tight_layout(); plt.savefig('/data/DRONE/bench/xfeat_audit/f494b.png',dpi=125)
print("\nwrote f494b.png")
