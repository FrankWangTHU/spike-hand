import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for fold in range(3):
    folder=ROOT/'experiments'/f'fold{fold}'
    if (folder/'complete.json').exists():continue
    folder.mkdir(parents=True,exist_ok=False)
    cmd=[sys.executable,'-X','utf8','train.py','--inner-fold',str(fold),'--output',str(folder),'--epochs','150','--schedule-epochs','220','--last-epoch','--hidden','192','--batch-size','8','--input-transform','raw','--shared-input','--adapter-scale','.1','--feature-mode','multiscale','--stacked-dual','--channel-dropout','.25','--lr','.0007','--tf32','--device','cuda','--phase-mode','all']
    (folder/'command.json').write_text(json.dumps(cmd,indent=2),encoding='utf-8')
    print('START',fold,flush=True)
    with (folder/'train.log').open('w',encoding='utf-8') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    (folder/'complete.json').write_text('{"complete":true}',encoding='utf-8')
    print('COMPLETE',fold,flush=True)
