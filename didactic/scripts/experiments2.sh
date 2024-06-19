#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=didactic_loop
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-16:00:00
#SBATCH --output=bash/out/%x-%j.out
#SBATCH --error=bash/out/%x-%j.err

exclude=(ht_grade , ht_severity , sanity , hf , stroke , tobacco , ccb , tz_diuretic , bmi , age , cad , ef , alpha_blocker , pad , etiology , A2C/lv_length , A2C/lv_area , A4C/lv_length , A4C/lv_area , sex , central_acting , beta_blocker , edv , dyslipidemia , A2C/myo_thickness_left , gfr , lateral_e_prime , a4c_ed_sc_max , reduced_e_prime , a4c_ed_lc_max , A2C/myo_thickness_right , spironolactone , nt_probnp , ph_vmax_tr , bradycardic , ht_cm , A2C/gls , diabetes , A4C/ls_right , ivs_d , esv , A4C/ls_left , diastolic_dysfunction_param_sum , A2C/ls_right , a2c_ed_ac_min , d_dysfunction_e_e_prime_ratio , A4C/myo_thickness_left , hr_tte , A4C/gls , sbp_24 , lvh , ddd , a2c_ed_ic_min , la_volume , lvid_d , mv_dt , A4C/myo_thickness_right , pw_d , ace_inhibitor , a2c_ed_ic_max , dbp_tte , A2C/ls_left , septal_e_prime , a4c_ed_sc_min , arb , e_e_prime_ratio , creat , e_velocity , pp_tte , sbp_tte , pp_24 , nt_probnp_group , dilated_la , a4c_ed_lc_min , diastolic_dysfunction , lvm_ind , dbp_24 , a_velocity , a2c_ed_ac_max)
exclude_raw=(ht_grade , ht_severity , sanity , bmi , dilated_la , ph_vmax_tr , A4C/ls_right , lvh , A4C/gls , A2C/gls , A4C/ls_left , A2C/ls_right , gfr , mv_dt , A2C/ls_left , nt_probnp_group , a2c_ed_ic_max , nt_probnp , sbp_24 , septal_e_prime , central_acting , a2c_ed_ic_min , dbp_tte , reduced_e_prime , A2C/lv_area , age , dbp_24 , pp_tte , A2C/lv_length , a4c_ed_sc_min , A4C/lv_length , a2c_ed_ac_min , a4c_ed_sc_max , A4C/lv_area , diastolic_dysfunction , ivs_d , beta_blocker , e_velocity , lvid_d , a4c_ed_lc_min , arb , ht_cm , A2C/myo_thickness_left , tobacco , d_dysfunction_e_e_prime_ratio , lateral_e_prime , pp_24 , a2c_ed_ac_max , A2C/myo_thickness_right , e_e_prime_ratio , ace_inhibitor , a_velocity , lvm_ind , alpha_blocker , sbp_tte , a4c_ed_lc_max , ef , A4C/myo_thickness_right , pw_d , A4C/myo_thickness_left , diastolic_dysfunction_param_sum , sex , la_volume , creat , dyslipidemia , edv , diabetes , esv , ddd , spironolactone , etiology , bradycardic , cad , hr_tte , tz_diuretic , ccb , pad , stroke , hf)
for i in ${!exclude[@]}; do
  if [[ $((2*i)) -le $((${#exclude[@]}+1)) ]] && [[ $((2*i+1)) -ge 5 ]]; then
    exclude_tabular_attrs=[${exclude[@]:0:$((2*i))}]
    job_path=didactic_attention_res/scripts/${i}
    poetry run didactic-runner $( echo hydra.run.dir=$HOME/$job_path) +experiment=cardinal/xtab-finetune +trainer.max_epochs=100 'task.predict_losses={sanity:{_target_:torch.nn.BCELoss}}' exclude_tabular_attrs=$( echo $exclude_tabular_attrs | sed 's/ //g') +seed=42
  fi;
done