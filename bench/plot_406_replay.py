#!/usr/bin/env python3
"""One picture: does a map fix move the answer toward the truth or away."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = list(csv.DictReader(open('/data/DRONE/flight_logs/analysis/406_replay.csv')))


def F(k):
    o = []
    for r in R:
        try:
            o.append(float(r[k]))
        except (TypeError, ValueError):
            o.append(np.nan)
    return np.array(o)


pe, fe, t, sc = F('prior_err_m'), F('fix_err_m'), F('t'), F('score')
m = ~np.isnan(fe) & ~np.isnan(pe) & (F('ok') == 1)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
a = ax[0]
a.scatter(pe[m], fe[m], s=10, alpha=.45, c=sc[m], cmap='viridis')
lim = max(np.nanmax(pe[m]), np.nanmax(fe[m])) * 1.05
a.plot([0, lim], [0, lim], 'k--', lw=1, label='fix ne kuch nahi badla')
a.axhline(15, color='tab:green', lw=1.2, ls=':', label='15 m (kaam ka fix)')
a.set_xlabel('node us waqt kitna galat tha (m)')
a.set_ylabel('fix ke baad error (m)')
a.set_title('Har line ke upar = matcher ne bigaada\n'
            f'{100*np.mean(fe[m] > pe[m]):.0f}% attempts line ke upar hai')
a.legend(fontsize=8)
a.set_xlim(0, lim); a.set_ylim(0, lim)
b = ax[1]
b.plot(t[m], pe[m], '.', ms=3, label='node ka error')
b.plot(t[m], fe[m], '.', ms=3, label='agar fix lagti')
b.axhline(45, color='tab:red', lw=1.2, ls='--',
          label='NCC ka search radius 45 m')
b.set_xlabel('flight time (s)'); b.set_ylabel('error (m)')
b.set_title('45 m ke baad sach search window ke bahar hai')
b.legend(fontsize=8)
fig.suptitle('Flight 406: map matcher ko production code se dobara chalaya '
             f'({m.sum()} attempts)')
fig.tight_layout()
p = '/data/DRONE/flight_logs/analysis/406_matcher.png'
fig.savefig(p, dpi=110)
print("saved " + p)
