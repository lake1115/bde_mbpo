# README

## GaussianMeanMLP

- 在使用`GaussianMeanMLP`时需要使用
  - `- dynamics_model: gaussian_mean_mlp_ensemble`

- 可以使用如下命令进行修改

```bash
python -m mbrl.examples.main \
  algorithm=bde_mbpo \
  overrides=mbpo_cartpole \
  dynamics_model=gaussian_mean_mlp_ensemble \ 
  ... \
  dynamics_model.activation_fn_cfg._target_=torch.nn.ReLU
```

其中文件夹命名规则为, 以下为例:
- rl: rollout_length参数

## GaussianBdeMLP

- 在使用`GaussianBdeMLP`时需要使用
  - `- dynamics_model: gaussian_bde_mlp_ensemble`

- 可以使用如下命令进行修改

```bash
python -m mbrl.examples.main \
  algorithm=bde_mbpo \
  overrides=mbpo_cartpole \
  dynamics_model=gaussian_bde_mlp_ensemble \ 
  ... \
  dynamics_model.activation_fn_cfg._target_=torch.nn.ReLU
```

bash中的`...`表示其他参数, 例如`rl=5`, 一些示例的bash代码可以参考 `test.sh` 文件

## visualization

目前针对不同的rollout length进行可视化, `visualization_rollout_length.py`
