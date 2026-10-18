import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
rows=[]
for name in ['continue_a3','go4_unsupervised','go4_supervised','clock4']:
    folder=(ROOT if name!='clock4' else ROOT.parent/'branch_a_clock')/'experiments'/name
    if not (folder/'complete.json').exists():continue
    config=json.loads((folder/'config.json').read_text())
    validation=json.loads((folder/'validation.json').read_text())
    comparisons=json.loads((folder/'comparison.json').read_text())+json.loads((folder/'replacement.json').read_text())
    best=max(comparisons,key=lambda r:r['mean']['hrfs_local'])
    row=dict(name=name,epoch=config['best_epoch'],single_raw=validation['mean']['hrfs_local'],
             best_mode=best['mode'],alpha=best['alpha'],best_blend=best['mean']['hrfs_local'],
             delta_vs_v3=best['mean']['hrfs_local']-52.06467483592634,
             delta_vs_round1_A3=best['mean']['hrfs_local']-52.23365738873004,
             sessions_improved=best.get('sessions_improved',0),phases=best['phases'],mean=best['mean'])
    if (folder/'phase_audit.json').exists():row['phase_prediction']=json.loads((folder/'phase_audit.json').read_text())['mean']
    rows.append(row)
result=dict(development_split=42,platform_submissions_used=0,round_complete=len(rows)==4,candidates=rows,
            development_gate_passed=[r['name'] for r in rows if r['delta_vs_v3']>=.4 and r['sessions_improved']>=12],
            caveat='Warm-start development experiments, selected on split42; no independent confirmation or platform evaluation.')
replay=ROOT/'experiments/go4_supervised_replay'
if (replay/'phase_audit_final.json').exists():
    result['supervised_final_epoch_diagnostic']=json.loads((replay/'phase_audit_final.json').read_text())
    result['replay_verification']=json.loads((replay/'replay_check.json').read_text())
(ROOT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
