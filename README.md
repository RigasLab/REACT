# Reinforcement Learning for Environmental Adaptation and Control of Turbulence (REACT)

[![arXiv](https://img.shields.io/badge/arXiv-2509.11002-b31b1b.svg)](https://arxiv.org/abs/2509.11002)

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Instructions for use](#instructions-for-use)
- [Issues]()
- [Citation](#citation)

# Overview

**Reinforcement Learning for Environmental Adaptation and Control of Turbulence (REACT)** is a data-driven flow-control platform that integrates a real-time sensing–actuation loop with on-line reinforcement learning. It learns control policies on the fly and supports a range of flow-control objectives.

# Repo Contents

- [LabVIEW real-time loop](./Labview_rtcode): `LabVIEW` code for real-time communication and data acquisition on a PXI system.  
  Runs on a Windows host connected via TCP to a PXI chassis booted in real-time mode (without a general-purpose operating system).
- [RL training and environment](./RLtraining): Python implementation including
  - RL algorithms to train the control policy, and
  - a custom Gym environment of the flow system that constructs observations from real-time sensing and transmits action signals to the real-time loop via UDP.
- [Data_postprocessing](./postprocessing_code_matlab): MATLAB post-processing scripts to reproduce the paper’s figures from raw pressure/force time-series and PIV velocity-field data.

# System Requirements

## Hardware Requirements

This experiment requires:
- a real-time **NI PXI** device with **LabVIEW** installed, and  
- a standard **PC with a GPU** capable of supporting reinforcement learning controller training.

For optimal performance, we recommend a system with specifications similar to those used in our experiments:

- **NI PXI** PXIe-8135
- **GPU:** NVIDIA GeForce RTX 4090  
- **CPU:** Intel® Core™ i9-14900K

Non-standard Hardware Requirements

- **ESP-DTC pressure scanner:** Chell **microDAQ-64DTC**, configured to boot in **UDP** mode.
- **Force balance:** **ATI Gamma-IP68** 6-axis force and torque sensor (with compatible signal conditioner).
- **Actuator:** Any motor/valve/jet driver that accepts **analog output** (e.g., 0–5 V) from the PXI system.


## Software Requirements

### Operating Systems
- **PXI (target):** Real-time OS (NI RT), PXI chassis booted in **real-time mode**
- **Host PC (for PXI control):** **Windows**
- **RL + custom environment:** **Linux**

**Versions used in our experiments**
- Linux: Ubuntu 22.04.3 LTS
- Windows: Windows 10

### Software Versions
- LabVIEW: 2021
- Python: 3.10.13


# Installation Guide

### LabVIEW Real-Time on NI PXI

1) **Install on Host PC**
- Install **LabVIEW** and **VI Package Manager**.
- From VI Package Manager, install/modify the **LabVIEW Real-Time** module (see NI [support](https://knowledge.ni.com/KnowledgeArticleDetails?id=kA03q000000x1r4CAA&l=en-GB)).

2) **Boot PXIe-8135 in Real-Time Mode**
- Enter BIOS (press **Delete** at boot) → **LabVIEW RT** → set **Boot Configuration** to *LabVIEW RT*.  
  (USB method also available—see NI [instruction](https://knowledge.ni.com/KnowledgeArticleDetails?id=kA03q000000YHpZCAW&l=en-GB).)

3) **Configure with NI MAX**
- Connect PXI and host PC to the same network.
- In **NI MAX**: format the PXI target → choose **Reliance File System**.
- **Install Software**: select **LabVIEW Real-Time 21.0.0** and **Network Streams 21.0** (others default).

4) **Connect from LabVIEW**
- Create a new **LabVIEW Project** on the host.
- **Add Target** → **New Targets and Devices** → select your **RT PXI**.
- Right-click the PXI target → **Connect**.

### Preparing for RL algorithm and custom gym environment

The recommended versions of the packages are:
```
pip install numpy==1.26.4 mamba-ssm==2.2.2 gym==0.21.0 drl-platform==0.1.5

```
The torch version that current experiment using is:

```
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121
```

The package for RL training should take approximately up to 1000 seconds to install on a recommended computer. 

If you are having an issue that you believe to be tied to software versioning issues, please raise a [Issue]()

# Demo
### RL toy problem
A minimal **Gym Pendulum** toy task is included to check the RL pipeline used in the experiment.  
- **Script:** [`RLtraining/easytest.py`](./RLtraining/easytest.py)  
- **Use:** quick training/evaluation sanity checks before running the full experimental setup  
- **Env:** `Pendulum-v1` (Gym 0.21)

**Run:**

```
# from the repo root
python RLtraining/easytest.py          # default run
```

The problem takes 100 episodes to converge, which is around 30 seconds of training for the listed machine. After convergence, you should see an average return around −100, with the best episodes approaching 0 (the highest possible return). Exact numbers can vary with seeds, hardware, and minor version differences.


# Instructions for use

### Launch training

1. Connect LabVIEW project
   - Open LabVIEW and load [`REACT_real_time.lvproj`](./Labview_rtcode/REACT_real_time.lvproj).
   - Connect the PXI (remote target) and the Windows host in the project.

2. Configure networking
   - Ensure the UDP port matches both the LabVIEW target main VI and the custom Gym environment.

3. Start data acquisition (Windows host)
   - Run the LabVIEW Host Main (if your setup requires host-side logging/DAQ).

4. Start online training (Linux GPU host)
   ```
   # from the repo root
   python RLtraining/onlinetraining.py
   ```
   Models and replay buffers are auto-saved at a user-defined interval (configurable in the script).
   Episode length: 4096 steps at 100 Hz (≈ 40.96 s per episode).
   
6. Start offline training (Linux GPU host)
   - After running step 4 under two different environmental conditions (e.g., wind speeds), open RLtraining/load_replaybuffer_offlinetrain.py
   - Set the replay-buffer paths to the two datasets you recorded.
   - Ensure offline_training = True in the setup.
   - Then run
     ```
     python RLtraining/load_replaybuffer_offlinetrain.py
     ```


### Results reproduction

MATLAB post-processing code is provided to reproduce the quantitative analysis in the manuscript.  
- **Code:** [`postprocessing_code_matlab`](./postprocessing_code_matlab)  
- **Data:** download the raw experimental dataset from [Zenodo (record 15801190)](https://zenodo.org/records/15801190)

**How to use**
1. Download and extract the Zenodo dataset to a local folder.
2. Open MATLAB and navigate to `postprocessing_code_matlab/`.
3. If required, update the data path variable at the top of the scripts (e.g., `DATA_DIR`/`data_path`).
4. Run the scripts to regenerate the figures reported in the manuscript.

**Hardware & runtime notes**
- Processing the **PIV velocity fields** requires **at least 8 GB RAM** (16–32 GB recommended).
- The **POD analysis** of the PIV fields may take **>600 s** depending on hardware.
- Reference machine used to produce the manuscript figures: **Intel Core i7 desktop, 32 GB RAM**.

## Contributors

The project code is actively developped and maintained by 

- [@Georgios-rigas](g.rigas@imperial.ac.uk) — RL Training & LabView
- [@Chengwei-xia](chengwei.xia20@imperial.ac.uk) — RL Training & LabView
- [@Junjie-zhang](jacky.zhang20@imperial.ac.uk) — RL Training & LabView



# Citation

In case of use of any of the codes, please cite [arXiv](https://arxiv.org/abs/2509.11002)
