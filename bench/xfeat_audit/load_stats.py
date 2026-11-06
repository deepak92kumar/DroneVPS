"""Summarise a run's load log. Skips the first 30 s - model load is not load."""
import csv, sys, statistics as st
def f(x):
    try: return float(x)
    except: return None
def one(p, label, tail=None):
    rows=[r for r in csv.DictReader(open(p)) if (f(r["t"]) or 0) > 30.0]
    if tail: rows = rows[-tail:]
    if not rows: print("%-22s no rows"%label); return
    g=lambda k:[f(r[k]) for r in rows if f(r[k]) is not None]
    cpu,rq,sl,th=g("cpu_pct"),g("rq_wait_pct"),g("slices_s"),g("threads")
    tops={}
    for r in rows:
        for k in ("top1","top2","top3"):
            v=(r.get(k) or "").strip()
            if ":" in v:
                n,p=v.rsplit(":",1)
                try: tops.setdefault(n,[]).append(float(p))
                except ValueError: pass
    print("%-22s n=%4d  cpu p50 %6.1f%%  p90 %6.1f%%  |  rq_wait p50 %6.1f%%  p90 %6.1f%%  |  slices/s %6.0f  thr %d"
          % (label,len(rows),st.median(cpu),sorted(cpu)[int(len(cpu)*.9)],
             st.median(rq),sorted(rq)[int(len(rq)*.9)],st.median(sl),int(st.median(th))))
    rank=sorted(((st.median(v)*len(v)/len(rows), n, st.median(v), len(v)) for n,v in tops.items()), reverse=True)
    for share,n,med,cnt in rank[:4]:
        print("      %-16s median %5.1f%% when listed, listed %d/%d rows" % (n,med,cnt,len(rows)))
for a in sys.argv[1:]:
    p,_,lab = a.partition("=")
    one(p, lab or p.split("/")[-1])
