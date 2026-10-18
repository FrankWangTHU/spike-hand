import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
outer=json.loads((ROOT.parent/'branch_a/experiments/a3_all/config.json').read_text())
from runtime.data import read_trial
rng=np.random.default_rng(1702)
folds=[{} for _ in range(3)]
for session,split in outer['split'].items():
    groups={}
    for file in split['train']:
        item=read_trial(ROOT.parent.parent/'player_data/Data_heldin'/session/'heldin'/file,False)
        groups.setdefault(item['condition'],[]).append(file)
    assignment={}
    for condition,files in sorted(groups.items()):
        rng.shuffle(files); start=int(rng.integers(3))
        for i,file in enumerate(files):assignment[file]=(i+start)%3
    for f in range(3):
        train=[file for file in split['train'] if assignment[file]!=f]
        valid=[file for file in split['train'] if assignment[file]==f]
        assert train and valid and not set(train)&set(valid)
        assert not (set(train)|set(valid))&set(split['validation'])
        folds[f][session]={'train':train,'validation':valid}
(ROOT/'folds.json').write_text(json.dumps(folds,indent=2),encoding='utf-8')
s=(ROOT/'train.py').read_text()
s=s.replace("    p.add_argument('--output', required=True)","    p.add_argument('--output', required=True)\n    p.add_argument('--inner-fold',type=int,required=True,choices=[0,1,2])")
s=s.replace("        tr, va = split_trials(items, 0.2, args.split_seed)","        partition=json.loads((Path(__file__).resolve().parent/'folds.json').read_text())[args.inner_fold][session.name]\n        tr=[i for i,t in enumerate(items) if Path(t['path']).name in partition['train']]\n        va=[i for i,t in enumerate(items) if Path(t['path']).name in partition['validation']]")
s=s.replace("    args = p.parse_args()", "    args = p.parse_args()\n    if args.init_checkpoint or args.refit or not args.last_epoch:\n        p.error('OOF models must train from scratch with fixed epoch selection')")
(ROOT/'train.py').write_text(s,encoding='utf-8')
print('Inner OOF counts',[(sum(len(v['train']) for v in f.values()),sum(len(v['validation']) for v in f.values())) for f in folds])
