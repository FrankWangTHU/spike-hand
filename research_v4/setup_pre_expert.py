from pathlib import Path
import shutil
root=Path(__file__).resolve().parent
src=root/'branch_a_round2'
dst=root/'branch_a_pre_expert'
dst.mkdir(exist_ok=False)
shutil.copytree(src/'runtime',dst/'runtime')
for name in ['metrics.py','training_utils.py','evaluate_branch.py','reference_v3.json','reference_config.json']:
    shutil.copy2(src/name,dst/name)
s=(src/'train.py').read_text(encoding='utf-8')
s=s.replace('def evaluate(model,', "A3_PATH = Path(__file__).resolve().parent.parent/'branch_a/experiments/a3_all/validation_predictions.npz'\nA3_PREDICTIONS = np.load(A3_PATH)\n\ndef blend_weight(times, onset, width=.2):\n    t=(times-times[onset]).astype('float64')/1e9\n    z=np.clip((t+width)/(2*width),0,1)\n    return (1-z*z*(3-2*z)).astype('float32')\n\ndef evaluate(model,")
s=s.replace('    predictions = {}','    predictions = {}\n    expert_predictions = {}')
s=s.replace("        score = score_trial(item['y'], p, item['times'], names[s])", "        key=f'{s}_{Path(item[\"path\"]).stem}'\n        expert_predictions[key]=p.copy()\n        weight=blend_weight(item['times'],item['onset'])[:,None]\n        p=weight*p+(1-weight)*A3_PREDICTIONS[key]\n        score = score_trial(item['y'], p, item['times'], names[s])")
s=s.replace("        np.savez_compressed(output / 'validation_predictions.npz', **predictions)","        np.savez_compressed(output / 'validation_predictions.npz', **predictions)\n        np.savez_compressed(output / 'expert_predictions.npz', **expert_predictions)")
s=s.replace('            loss = position + 0.2 * velocity + 0.05 * bone', '''            # Independent expert: full input context, losses only before Ton+0.2s.
            # Smoothly taper supervision over the same overlap as inference.
            z=((meta[:,:,0]*4+.2)/.4).clamp(0,1)
            weight=(1-z.square()*(3-2*z))*mask
            def weighted_mean(value, weights):
                return (value*weights).sum()/weights.sum().clamp_min(1)
            position=weighted_mean(.4*distances[:,:,0]/20+.6*distances[:,:,1:].mean(-1)/15,weight)
            pair_weight=torch.minimum(weight[:,1:],weight[:,:-1])
            velocity=weighted_mean(delta.square().mean(-1),pair_weight)
            bone=weighted_mean(((lp-ly).abs()/ly.clamp_min(1)).mean(-1),weight)
            loss = position + 0.2 * velocity + 0.05 * bone''')
s=s.replace("    if validation:\n        model, stats", "    save_checkpoint(output/'final_checkpoint.pt',model,stats,config,names)\n    if validation:\n        model, stats")
(dst/'train.py').write_text(s,encoding='utf-8')
print(dst)
