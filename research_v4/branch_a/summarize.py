import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
rows=[]
for name in ['a1_weight','a2_wrist','a3_all']:
    folder=ROOT/'experiments'/name
    if not (folder/'complete.json').exists(): continue
    config=json.loads((folder/'config.json').read_text())
    validation=json.loads((folder/'validation.json').read_text())
    comparisons=json.loads((folder/'comparison.json').read_text())
    if (folder/'replacement.json').exists():
        comparisons+=json.loads((folder/'replacement.json').read_text())
    best=max(comparisons,key=lambda x:x['mean']['hrfs_local'])
    baseline=comparisons[0]
    rows.append(dict(name=name,epoch=config['best_epoch'],single_raw=validation['mean']['hrfs_local'],
                     best_alpha=best['alpha'],best_mode=best.get('mode','append'),best_blend=best['mean']['hrfs_local'],
                     delta=best['mean']['hrfs_local']-baseline['mean']['hrfs_local'],
                     sessions_improved=best.get('sessions_improved',0),
                     pre_wrist_delta=best['phases']['pre_ton']['wrist_error']-baseline['phases']['pre_ton']['wrist_error'],
                     mean=best['mean'],phases=best['phases']))
result=dict(platform_baseline_user_report=54.38719,local_v3_baseline=52.06467483592634,development_split=42,
            platform_submissions_used=0,candidates=rows,
            first_round_complete=len(rows)==3,
            development_gate_passed=[r['name'] for r in rows if r['delta']>=.4 and r['sessions_improved']>=12],
            caveat='Selected development results only; not platform scores or independent confirmation.')
(ROOT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
