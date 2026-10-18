import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
folder=ROOT/'experiments/clock4'
if folder.exists():raise ValueError('Refusing to overwrite clock experiment')
folder.mkdir(parents=True)
cmd=json.loads((ROOT.parent/'branch_a_round2/experiments/go4_unsupervised/command.json').read_text())
cmd[cmd.index('--output')+1]=str(folder)
cmd[cmd.index('--phase-mode')+1]='clock4'
(folder/'command.json').write_text(json.dumps(cmd,indent=2),encoding='utf-8')
print('START clock4',flush=True)
with (folder/'train.log').open('w',encoding='utf-8') as log:
    subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
for flag,name in [([],'comparison.log'),(['--replace-stacked'],'replacement.log')]:
    with (folder/name).open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8','evaluate_branch.py','--experiment',str(folder)]+flag,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
result=json.loads((folder/'validation.json').read_text())['mean']
(folder/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('COMPLETE clock4',result,flush=True)
