
## Dependencies
- *Python* - 3.10  
- [*PyTorch* - 2.4.0 with CUDA 12.1](https://pytorch.org/get-started/previous-versions/)

## Installation
1. Create the conda environment:
    ```bash
    conda env create -f environment.yml
    ```
2. Modify xml file (in xml/genesis/quick_scene.xml):
    <hfield name="terrain" file="Absolute path to agent_eval_gym.png" size="10.5 10.5 .4 0.1" />
    <include file="Absolute path to quick_biped_wheel.xml"/>

## Quick Start
1. Replay the RL policy on Biped wheel robot:
    ```bash
    conda activate rlmujoco
    cd ~/quick_bipedal_urdf
    python quick_gs2mj.py


