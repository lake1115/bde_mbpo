# 修改seed
python -m mbrl.examples.main \
    algorithm=bde_mbpo \
    overrides=bde_mbpo_hopper \
    dynamics_model=gaussian_bde_mlp_ensemble \
    overrides.rollout_length=0 \
    experiment=bde_hopper_test \
    overrides.N_s=30 \
    seed=1 \
    overrides.log_model_weights_freq=25

python -m mbrl.examples.main \
    algorithm=mbpo \
    overrides=mbpo_hopper \
    dynamics_model=gaussian_mlp_ensemble \
    experiment=mbpo_hopper_test \
    seed=1

# N_s 设置为1即可, 此处不涉及到BDE, 和MBPO的差距在rollout时它对model pool做平均, 而MBPO是随机选一个
python -m mbrl.examples.main \
    algorithm=bde_mbpo \
    overrides=bde_mbpo_hopper \
    dynamics_model=gaussian_mean_mlp_ensemble \
    seed=1 \
    experiment=mean_hopper_test \
    overrides.N_s=1 \
    overrides.rollout_length=0 # \
    # rollout_length=2
