# PLAS: Latent Action Space for Offline Reinforcement Learning

This is the repository for our paper "PLAS: Latent Action Space for Offline Reinforcement Learning" in CoRL 2020.
Please visit our [website](https://sites.google.com/view/latent-policy) for more information.

This repository is built on top of [BCQ](https://github.com/sfujim/BCQ). The logger is from [BEAR](https://github.com/aviralkumar2907/BEAR).

## Requirements
- Python 3.7.4
- [PyTorch](https://github.com/pytorch/pytorch) (v1.2.0)
- [mujoco_py](https://github.com/openai/mujoco-py) (v2.0)
- [gym](https://github.com/openai/mujoco-py) (v0.13)
- [d4rl](https://github.com/rail-berkeley/d4rl) (commit: 87d13f1)

You may install the above packages following the instructions in their repositories, or run the following command:
```
pip3 install -r requirements.txt
```
Note that the latest d4rl repository has some problem loading the mujoco dataset. We recommend the users to install [this](https://github.com/rail-berkeley/d4rl/tree/87d13f172aa253004caa32b24df0ce449328f3b3) commit version.

## Instructions

To train the Latent Policy for the d4rl datasets:
```
python main.py --env_name walker2d-medium-expert-v0 --algo_name Latent --max_latent_action 2
```

To train the Latent Policy with the perturbation layer:
```
python main.py --env_name walker2d-medium-expert-v0 --algo_name LatentPerturbation --max_latent_action 2 --phi 0.05
```

By default, the algorithm trains a VAE before the policy to model the behavior policy of the dataset. You may also load a pre-trained vae and then train policy.
```
python main.py --env_name walker2d-medium-expert-v0 --algo_name Latent --vae_mode v6
```
This command will load the vae models under the "models/vae_v6" folder according to the name of the dataset and the random seed automatically.

The results will be saved under the "results" folder. You may use [viskit](https://github.com/vitchyr/viskit) to visualize the curves.

## Citation
```
@inproceedings{PLAS_corl2020,
 title={PLAS: Latent Action Space for Offline Reinforcement Learning},
 author={Zhou, Wenxuan and Bajracharya, Sujay and Held, David},
 booktitle={Conference on Robot Learning},
 year={2020}
}
```
