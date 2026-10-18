"""Sequential GPU queue. No platform submission."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(); p.add_argument('--names',nargs='+',default=['control','a1_weight','a2_wrist','a3_all']); args=p.parse_args()
    options={'control':[],'a1_weight':['--pre-weight','1.5'],'a2_wrist':['--phase-mode','wrist'],'a3_all':['--phase-mode','all']}
    for name in args.names:
        output=ROOT/'experiments'/name
        if (output/'complete.json').exists(): continue
        if output.exists(): raise RuntimeError(f'Refusing to overwrite incomplete run {output}')
        output.mkdir(parents=True)
        cmd=[sys.executable,'-X','utf8','train.py','--output',str(output),'--epochs','220','--hidden','192','--batch-size','8',
             '--input-transform','raw','--shared-input','--adapter-scale','.1','--feature-mode','multiscale','--stacked-dual',
             '--channel-dropout','.25','--lr','.0007','--tf32','--device','cuda']+options[name]
        (output/'command.json').write_text(json.dumps(cmd,indent=2),encoding='utf-8')
        print('START',name,flush=True); start=time.time()
        with (output/'train.log').open('w',encoding='utf-8') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (output/'comparison.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8','evaluate_branch.py','--experiment',str(output)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((output/'validation.json').read_text())['mean']
        (output/'complete.json').write_text(json.dumps(dict(seconds=time.time()-start,mean=result)),encoding='utf-8')
        print('COMPLETE',name,json.dumps(result),flush=True)

if __name__=='__main__': main()
