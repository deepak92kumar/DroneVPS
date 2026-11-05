"""5 Nov flight 486 on the map, and the oscillation that drove it.

Left: the ground the aircraft flew over, the M9N's track, the VPS estimate,
and every published fix coloured by the matcher that produced it.
Right: our position error against the M9N, and the velocity we declared -
the two together are the whole diagnosis.
"""
import csv, math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/DRONE/vps_vio")
from map_cache import MapCache

plt.rcParams.update({"figure.dpi":130,"font.size":9,"axes.grid":True,
                     "grid.alpha":.25,"axes.axisbelow":True})
MDEG=111320.0
C={"m9n":"#2f855a","vps":"#2b6cb0","xfeat":"#805ad5","ncc":"#c05621",
   "xfeat_lg":"#b83280"}
P='/home/abc/NVME/DRONE/flight_logs/vps/vps_00486/vps_00486.csv'
r=list(csv.DictReader(open(P)))
F=lambda x,k:(float(x[k]) if x.get(k) not in (None,'') else np.nan)
g=lambda k: np.array([F(x,k) for x in r])
t,alt=g('t'),g('alt_m'); n,e=g('north_m'),g('east_m')
gn,ge=g('gps1_north_m'),g('gps1_east_m')
lat,lon=g('lat'),g('lon'); vs=g('vel_speed')
fl,fa=g('fix_lon'),g('fix_lat')
src=[x.get('fix_source','') for x in r]
used=np.array([x.get('fix_used')=='1' for x in r])
ok=np.isfinite(gn)&np.isfinite(ge)&(np.abs(gn)<400)&(np.abs(ge)<400)&np.isfinite(n)
err=np.where(ok,np.hypot(n-gn,e-ge),np.nan)
leg=(t>=198)&(t<=514)

lat0=np.nanmedian(lat[leg]); mlon=MDEG*math.cos(math.radians(lat0))
olat=np.nanmedian(lat[leg]-n[leg]/MDEG); olon=np.nanmedian(lon[leg]-e[leg]/mlon)
cache=MapCache(os.path.realpath('/data/DRONE/maps/active.tif'))
span=max(np.nanmax(np.abs(n[leg&ok])),np.nanmax(np.abs(e[leg&ok])),
         np.nanmax(np.abs(gn[leg&ok])),np.nanmax(np.abs(ge[leg&ok])))+60
clon=olon+np.nanmean(e[leg])/mlon; clat=olat+np.nanmean(n[leg])/MDEG
win,wg=cache.window(clon,clat,float(span),kind="grey")
H,W=win.shape[:2]
def px(N,E):
    cN=np.nanmean(n[leg]); cE=np.nanmean(e[leg])
    return W/2+(E-cE)/(2*span/W), H/2-(N-cN)/(2*span/H)

fig=plt.figure(figsize=(14.5,6.4))
ax0=fig.add_subplot(1,2,1); ax1=fig.add_subplot(2,2,2); ax2=fig.add_subplot(2,2,4)
ax0.imshow(win,cmap="gray",interpolation="nearest")
x,y=px(gn[leg&ok],ge[leg&ok]); ax0.plot(x,y,"-",color=C["m9n"],lw=2.0,label="M9N (sach)")
x,y=px(n[leg],e[leg]);         ax0.plot(x,y,"-",color=C["vps"],lw=1.3,label="VPS estimate")
for s in ("xfeat","ncc","xfeat_lg"):
    m=leg&used&np.array([q==s for q in src])&np.isfinite(fa)
    if not m.any(): continue
    fN=(fa[m]-olat)*MDEG; fE=(fl[m]-olon)*mlon
    fx,fy=px(fN,fE)
    ax0.scatter(fx,fy,s=26,color=C[s],zorder=6,edgecolor="white",linewidth=0.5,
                label=f"{s} ne diya fix ({m.sum()})")
ax0.set_xlim(0,W); ax0.set_ylim(H,0); ax0.axis("off")
ax0.legend(fontsize=8,loc="upper left",framealpha=0.9)
ax0.set_title("flight 486, 5 Nov — satellite sheet par: M9N, VPS, aur har matcher ke fix\n"
              "(%.0f x %.0f m dikha hai)"%(2*span,2*span),fontsize=9.5)

ax1.plot(t[leg],err[leg],"-",color=C["vps"],lw=1.3)
ax1b=ax1.twinx(); ax1b.plot(t[leg],alt[leg],"-",color="0.6",lw=1.0)
ax1b.set_ylabel("alt (m)",color="0.5",fontsize=8)
ax1.axhline(25,color="#b83280",ls="--",lw=1)
ax1.text(t[leg][0]+3,26.5,"EK3_GLITCH_RAD 25 m",fontsize=7.5,color="#b83280")
ax1.set_ylabel("error vs M9N (m)"); ax1.set_title(
    "position error — 96 m par 22.85 m peak-to-peak, 23-92 s ke period",fontsize=9)
ax2.plot(t[leg],vs[leg],"-",color="#c05621",lw=1.1,label="jo speed hum FC ko bolte hai")
ax2.axhline(0.845,color="#2f855a",ls="--",lw=1.2,label="deadband 0.845 m/s")
ax2.set_xlabel("t (s)"); ax2.set_ylabel("m/s"); ax2.legend(fontsize=8)
ax2.set_title("phantom velocity — drone khada tha, par 90-100 m par p50 1.76 m/s",fontsize=9)
plt.tight_layout()
plt.savefig('/data/DRONE/bench/xfeat_audit/flight486_map.png',dpi=125)
print("wrote flight486_map.png")
