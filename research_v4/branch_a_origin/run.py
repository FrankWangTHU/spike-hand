import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
folder=ROOT/'experiments/anchor'
folder.mkdir(parents=True,exist_ok=False)
cmd=[sys.executable,'-X','utf8','train.py','--output',str(folder),'--epochs','60','--hidden','192','--batch-size','8','--input-transform','raw','--shared-input','--adapter-scale','.1','--feature-mode','multiscale','--stacked-dual','--channel-dropout','.25','--lr','.0001','--tf32','--device','cuda','--phase-mode','all','--init-checkpoint','../branch_a/experiments/a3_all/checkpoint.pt']
(folder/'command.json').write_text(json.dumps(cmd,indent=2),encoding='utf-8')
with (folder/'train.log').open('w',encoding='utf-8') as f:
    subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
with (folder/'replacement.log').open('w',encoding='utf-8') as f:
    subprocess.run([sys.executable,'-X','utf8','evaluate_branch.py','--experiment',str(folder),'--replace-stacked'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
(folder/'complete.json').write_text('{"complete":true,"submissions":0}',encoding='utf-8')
print('COMPLETE',flush=True)
