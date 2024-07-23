#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=IRENE-loop-no-echo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-16:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err

uname -a
nvidia-smi

ulimit -n 4096

for seed in {46..53}; do
    poetry run didactic-runner 'hydra.run.dir=/home/stympopper/didacticWORKSHOP/irene_ternary-no-echo${seed}' +experiment=cardinal/xtab-finetune +trainer.max_epochs=100 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade,sanity] +seed=$seed task.irene_baseline=True
done