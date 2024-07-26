#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=TabularBaseline-no-tte-xtab
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-16:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err

uname -a
nvidia-smi

ulimit -n 4096

for seed in {42..51}; do
    # poetry run didactic-runner 'hydra.run.dir=/home/stympopper/didacticWORKSHOP/irene_ternary-no-echo${seed}' +experiment=cardinal/xtab-finetune +trainer.max_epochs=100 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade,sanity] +seed=$seed task.irene_baseline=True
    # poetry run didactic-runner 'hydra.run.dir=/home/stympopper/didacticWORKSHOP/cross-xtab-prenorm${seed}' +experiment=cardinal/xtab-finetune +trainer.max_epochs=100 'task.predict_losses={sanity:{_target_:torch.nn.BCELoss}}' exclude_tabular_attrs=[ht_severity,ht_grade,sanity] +seed=$seed task.cross_attention=True task.first_prenormalization=True ckpt=/home/stympopper/didacticJerem/ckpts/xtab.ckpt
    # poetry run didactic-runner 'hydra.run.dir=/home/stympopper/didacticWORKSHOP/late-concat-xtab${seed}' +experiment=cardinal/xtab-finetune +trainer.max_epochs=100 'task.predict_losses={sanity:{_target_:torch.nn.BCELoss}}' exclude_tabular_attrs=[ht_severity,ht_grade,sanity] +seed=$seed task.cross_attention=False task.first_prenormalization=False task.late_concat=True ckpt=/home/stympopper/didacticJerem/ckpts/xtab.ckpt
    # poetry run didactic-runner 'hydra.run.dir=/home/stympopper/didacticWORKSHOP/sum-fusion-xtab${seed}' +experiment=cardinal/xtab-finetune +trainer.max_epochs=100 'task.predict_losses={sanity:{_target_:torch.nn.BCELoss}}' exclude_tabular_attrs=[ht_severity,ht_grade,sanity] +seed=$seed task.cross_attention=False task.first_prenormalization=False task.sum_fusion=True ckpt=/home/stympopper/didacticJerem/ckpts/xtab.ckpt
    # poetry run didactic-runner 'hydra.run.dir=/home/stympopper/didacticWORKSHOP/product-fusion-xtab${seed}' +experiment=cardinal/xtab-finetune +trainer.max_epochs=100 'task.predict_losses={sanity:{_target_:torch.nn.BCELoss}}' exclude_tabular_attrs=[ht_severity,ht_grade,sanity] +seed=$seed task.cross_attention=False task.first_prenormalization=False task.product_fusion=True ckpt=/home/stympopper/didacticJerem/ckpts/xtab.ckpt
    # poetry run didactic-runner 'hydra.run.dir=/home/stympopper/didacticWORKSHOP/early-fusion${seed}' +experiment=cardinal/xtab-finetune +trainer.max_epochs=100 'task.predict_losses={sanity:{_target_:torch.nn.BCELoss}}' exclude_tabular_attrs=[ht_severity,ht_grade,sanity] +seed=$seed task.cross_attention=False task.first_prenormalization=False ckpt=/home/stympopper/didacticJerem/ckpts/xtab.ckpt
    # poetry run didactic-runner 'hydra.run.dir=/home/stympopper/didacticWORKSHOP/tabular_baseline-xtabREAL${seed}' +experiment=cardinal/xtab-finetune +trainer.max_epochs=100 'task.predict_losses={sanity:{_target_:torch.nn.BCELoss}}' exclude_tabular_attrs=[ht_severity,ht_grade,sanity] +seed=$seed task.cross_attention=False ckpt=/home/stympopper/didacticJerem/ckpts/xtab.ckpt
    # poetry run didactic-runner 'hydra.run.dir=/home/stympopper/didacticWORKSHOP/tabular_baseline-no-tte-xtab${seed}' +experiment=cardinal/xtab-finetune +trainer.max_epochs=100 'task.predict_losses={sanity:{_target_:torch.nn.BCELoss}}' exclude_tabular_attrs=[ht_severity,ht_grade,sanity] +seed=$seed task.cross_attention=False ckpt=/home/stympopper/didacticJerem/ckpts/xtab.ckpt
    poetry run didactic-runner 'hydra.run.dir=/home/stympopper/didacticWORKSHOP/tabular_baseline-no-tte-xtab${seed}' +experiment=cardinal/xtab-finetune +trainer.max_epochs=100 'task.predict_losses={sanity:{_target_:torch.nn.BCELoss}}' exclude_tabular_attrs=[ht_severity,ht_grade,sanity] +seed=$seed task.cross_attention=True
done