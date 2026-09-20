import json,glob,numpy as np
from pathlib import Path
O=Path('results/v2/enh')
names={'REF':'MSC-MIL original (clean-trained)','E1':'E1 MSC-MIL-NA','E2':'E2 MSC-MIL-PF','E3':'E3 Fusion (CatBoost-NA + E1)','E4':'E4 MSC-MIL-Long'}
rows={}
L=[]
for c in names:
    R=[json.load(open(f)) for f in sorted(O.glob(f'{c}_scma*_seed*.json'))]
    d={'c':c}
    for t in (1,4):
        rt=[r for r in R if r['scma']==t]
        for k in ('clean_f1','noisy_mean','S_T'):
            v=[r[k] for r in rt]; d[f'{k}{t}']=(np.mean(v),np.std(v))
        d[f'rec{t}']=min(r['recall_clean'] for r in rt)
    d['S']=np.mean([r['S_T'] for r in R]); d['clean']=np.mean([r['clean_f1'] for r in R]); d['noisy']=np.mean([r['noisy_mean'] for r in R])
    d['params']=R[0]['n_params']; d['minrec']=min(r['recall_clean'] for r in R)
    lat=[r for r in R if 'latency_ms_cpu' in r][0]; d['lg']=lat.get('latency_ms_cuda',float('nan')); d['lc']=lat['latency_ms_cpu']
    d['ep']=np.mean([r.get('best_epoch',np.nan) for r in R]) if c!='E3' else float('nan')
    rows[c]=d
best=max((c for c in rows if rows[c]['minrec']>0),key=lambda c:rows[c]['S'])
# tie-break
top=rows[best]['S']; tied=[c for c in rows if rows[c]['minrec']>0 and top-rows[c]['S']<=0.005]
win=min(tied,key=lambda c:rows[c]['params'])
L.append('| candidate | S | clean F1 (mean) | noisy F1 (mean) | params | disq. (min clean recall) | latency b=1 GPU / CPU (ms) |\n|---|---|---|---|---|---|---|')
for c,d in sorted(rows.items(),key=lambda x:-x[1]['S']):
    L.append(f"| {names[c]} | **{d['S']:.4f}** | {d['clean']:.4f} | {d['noisy']:.4f} | {d['params']:,} | no ({d['minrec']:.3f}) | {d['lg']:.1f} / {d['lc']:.1f} |")
L.append('\nPer type, mean +/- std over seeds 0-2 (population std):\n\n| candidate | SCMA-1 clean | SCMA-1 noisy | SCMA-1 S_T | SCMA-4 clean | SCMA-4 noisy | SCMA-4 S_T |\n|---|---|---|---|---|---|---|')
for c,d in rows.items():
    f=lambda k:'%.4f +/- %.4f'%d[k]
    L.append(f"| {names[c]} | {f('clean_f11')} | {f('noisy_mean1')} | {f('S_T1')} | {f('clean_f14')} | {f('noisy_mean4')} | {f('S_T4')} |")
L.append(f"\nCandidates within 0.005 of the top S: {tied}; winner by protocol: **{win}**")
Path('results/v2/enh_tables.md').write_text('\n'.join(L)); print('\n'.join(L)); print(win)
