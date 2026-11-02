#!/usr/bin/env python3
"""What, if anything, separates a good map fix from a bad one on flight 406."""
import csv, sys
import numpy as np

P = '/data/DRONE/flight_logs/analysis/406_replay.csv'
R = list(csv.DictReader(open(P)))


def F(k):
    o = []
    for r in R:
        try:
            o.append(float(r[k]))
        except (TypeError, ValueError):
            o.append(np.nan)
    return np.array(o)


ok = F('ok') == 1
pe, fe = F('prior_err_m'), F('fix_err_m')
sc, mg, inl = F('score'), F('margin'), F('inliers')
dis = F('disagree_m')
print(f"{len(R)} attempts, matcher said ok on {ok.sum()}")
print(f"  inliers > 0 on {np.nansum(inl > 0):.0f};  "
      f"disagree available on {np.sum(~np.isnan(dis))}")
m = ok & ~np.isnan(fe) & ~np.isnan(pe)
print(f"\n=== kya fix prior se behtar hai? (n={m.sum()}) ===")
print(f"  prior error  p50 {np.median(pe[m]):6.1f} m")
print(f"  fix error    p50 {np.median(fe[m]):6.1f} m")
print(f"  corr(prior, fix) = {np.corrcoef(pe[m], fe[m])[0,1]:+.3f}")
better = fe[m] < pe[m]
print(f"  fix ne sudhara: {better.sum()}/{m.sum()} = {100*better.mean():.0f}%")
print(f"  median sudhaar: {np.median(pe[m] - fe[m]):+.1f} m")

print("\n=== prior kitna galat tha, uske hisaab se ===")
for lo, hi in ((0, 15), (15, 30), (30, 45), (45, 60), (60, 200)):
    s = m & (pe >= lo) & (pe < hi)
    if s.sum() < 3:
        print(f"  prior {lo:3d}-{hi:3d} m: n={s.sum()}")
        continue
    print(f"  prior {lo:3d}-{hi:3d} m: n={s.sum():3d}  fix err p50 "
          f"{np.median(fe[s]):6.1f} m  sudhaar {np.median(pe[s]-fe[s]):+6.1f} m  "
          f"behtar {100*np.mean(fe[s] < pe[s]):3.0f}%")

print("\n=== kya score/margin achhe fix ko alag karta hai? ===")
for nm, v in (("score", sc), ("margin", mg), ("inliers", inl)):
    s = m & ~np.isnan(v)
    if s.sum() < 10:
        continue
    print(f"  corr({nm:8s}, fix error) = {np.corrcoef(v[s], fe[s])[0,1]:+.3f}")
good = m & (fe < 15)
print(f"\n  fix 15 m ke andar: {good.sum()}/{m.sum()}")
if good.sum() >= 3:
    print(f"    un par score p50 {np.median(sc[good]):.3f}  "
          f"margin p50 {np.median(mg[good]):.3f}")
    bad = m & (fe >= 15)
    print(f"    baaki par score p50 {np.median(sc[bad]):.3f}  "
          f"margin p50 {np.median(mg[bad]):.3f}")
    print("\n  threshold se separation (precision = us threshold ke upar "
          "kitne fix 15 m ke andar):")
    for nm, v in (("score", sc), ("margin", mg)):
        print(f"    {nm}:")
        for q in (50, 70, 80, 90, 95):
            thr = np.percentile(v[m], q)
            sel = m & (v >= thr)
            if sel.sum() < 3:
                continue
            prec = np.mean(fe[sel] < 15)
            print(f"      >= {thr:.3f} (p{q}): n={sel.sum():3d}  "
                  f"precision {100*prec:3.0f}%  "
                  f"(baseline {100*np.mean(fe[m] < 15):.0f}%)")
