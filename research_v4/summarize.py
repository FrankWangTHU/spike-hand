import json
from pathlib import Path
root=Path(__file__).resolve().parent
result={'target':58,'score_type':'local HRFS reproduction; not leaderboard','split_seed':42,'experiments':{},'confirmation':None}
for name in ['denoising_complete','manifold','score_finetune','tcn_pair']:
    folder=root/'experiments'/name
    comparisons=json.loads((folder/'comparison.json').read_text())
    best=max(comparisons,key=lambda r:r['mean']['hrfs_local'])
    entry={'best_blend_score':best['mean']['hrfs_local'],'candidate_weight':best['candidate_weight'],'mean':best['mean']}
    if (folder/'validation.json').exists():
        raw=json.loads((folder/'validation.json').read_text())
        entry.update(raw_score=raw['mean']['hrfs_local'],selected_epoch=raw['epoch'])
    result['experiments'][name]=entry
result['reference_score']=comparisons[0]['mean']['hrfs_local']
result['best_development_score']=max(x['best_blend_score'] for x in result['experiments'].values())
result['target_reached']=result['best_development_score']>=58
confirm=root/'experiments/confirmation43/comparison.json'
if confirm.exists():
    result['confirmation']={'protocol':json.loads((root/'confirmation_protocol.json').read_text(encoding='utf-8-sig')),'results':json.loads(confirm.read_text())}
result['limitations']=['Development checkpoints and weights selected on split42.','Alternate split uses same heldin trials; not independent external data.','Full v3 plus TCN-pair combination has not received second-split confirmation.','No official platform evaluation, full-data refit or submission package.']
(root/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='confirmation'},indent=2))
