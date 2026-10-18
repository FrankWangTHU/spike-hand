import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',default=['continue_a3','go4_unsupervised','go4_supervised']);args=p.parse_args()
    modes={'continue_a3':('all','0'),'go4_unsupervised':('go4','0'),'go4_supervised':('go4','.05')}
    for name in args.names:
        folder=ROOT/'experiments'/name
        if (folder/'complete.json').exists():continue
        if folder.exists():raise ValueError(f'Refusing to overwrite {folder}')
        folder.mkdir(parents=True)
        mode,weight=modes[name]
        cmd=[sys.executable,'-X','utf8','train.py','--output',str(folder),'--epochs','60','--hidden','192','--batch-size','8',
             '--input-transform','raw','--shared-input','--adapter-scale','.1','--feature-mode','multiscale','--stacked-dual',
             '--channel-dropout','.25','--lr','.0001','--tf32','--device','cuda','--phase-mode',mode,'--phase-aux-weight',weight,
             '--init-checkpoint','../branch_a/experiments/a3_all/checkpoint.pt']
        (folder/'command.json').write_text(json.dumps(cmd,indent=2),encoding='utf-8')
        print('START',name,flush=True); start=time.time()
        with (folder/'train.log').open('w',encoding='utf-8') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        for flag,logname in [([], 'comparison.log'),(['--replace-stacked'],'replacement.log')]:
            with (folder/logname).open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-X','utf8','evaluate_branch.py','--experiment',str(folder)]+flag,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((folder/'validation.json').read_text())['mean']
        (folder/'complete.json').write_text(json.dumps(dict(seconds=time.time()-start,mean=result)),encoding='utf-8')
        print('COMPLETE',name,json.dumps(result),flush=True)

if __name__=='__main__':main()
