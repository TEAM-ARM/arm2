<h1 align="center"> ARM2: Adaptive Reasoning Model with Vision Understanding and Executable Code</h1>

## Overview

ARM2 is an adaptive reasoning model with vision understanding and executable code capabilities. This repository contains the codebase for Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [VeRL](https://github.com/volcengine/verl).

## Table of Contents

- [Data & Model](#data--model)
- [Environment Setup](#environment-setup)
- [Stage 1: Supervised Fine-Tuning (SFT)](#stage-1-supervised-fine-tuning-sft)
- [Stage 2: Reinforcement Learning (RL)](#stage-2-reinforcement-learning-rl)
- [Contact](#contact)
- [Citation](#citation)

## Data & Model

### Model Download

You can download our model from [🤗HuggingFace](https://huggingface.co/arm-team/ARM2-7B).

### Dataset Download

For SFT, please download the images from [🤗HuggingFace](https://huggingface.co/datasets/TIGER-Lab/VisualWebInstruct).

**Note**: After downloading, you should adjust the file paths of images in `LLaMA-Factory/data/visualwebinstruct_sft.json`.

## Environment Setup

This project requires two separate conda environments for SFT and RL stages.

### Environment Files

For easy reproduction, we provide exported environment files:
- `environment.yml`: Complete conda environment export (recommended)
- `requirements.txt`: All pip dependencies
- `setup_verl_env.sh`: Automated setup script

### SFT Environment Setup

```bash
# Create conda environment
conda create -n llamafactory python=3.11
conda activate llamafactory

# Install LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
pip3 install flash-attn --no-build-isolation
```

### RL Environment Setup

#### Option 1: Quick Setup (Recommended - Using Exported Environment)

We provide exported environment files for easy reproduction:

```bash
# Method A: Using conda environment file (recommended)
conda env create -f environment.yml
conda activate verl
cd verl
pip install -e .

# Method B: Using automated setup script
bash setup_verl_env.sh

# Method C: Manual installation from requirements.txt
conda create -n verl python=3.11
conda activate verl
cd verl
pip install -e .
pip install flash-attn==2.7.4.post1 --no-build-isolation  # Install flash-attn separately
pip install -r ../requirements.txt
```

**Note**: `flash-attn` may need to be installed separately with `--no-build-isolation` flag if installation fails.

#### Option 2: Manual Setup

```bash
# Create conda environment
conda create -n verl python=3.11
conda activate verl

# Install VeRL and dependencies
cd verl
pip3 install -e .
pip3 install flash-attn --no-build-isolation
pip3 install fastapi uvicorn openai vllm==0.8.3 numpy<2.0.0
pip install "opentelemetry-api>=1.34.0" "opentelemetry-sdk>=1.34.0" "opentelemetry-exporter-otlp>=1.34.0"
```

**Note**: The exported environment includes:
- Python 3.11.0
- PyTorch 2.6.0
- vLLM 0.8.3
- Ray 2.43.0
- Transformers 4.57.3
- Flash Attention 2.7.4.post1
- All other dependencies (see `requirements.txt` for full list)

## Stage 1: Supervised Fine-Tuning (SFT)

### Activate Environment

```bash
conda activate llamafactory
cd LLaMA-Factory
```

### Training

```bash
llamafactory-cli train examples/train_lora/qwen2_5vl_lora_sft.yaml
```

## Stage 2: Reinforcement Learning (RL)

### Activate Environment

```bash
conda activate verl
cd verl
```

### Data Processing

You can find examples in `verl/verl/data`.

### Training

**Important**: Before running the script, please adjust the paths of policy models and datasets to your own paths.

**System Requirements**:
- Ensure sufficient system resources (process limits, memory, etc.)

```bash
bash verl/verl/scripts/run.sh
```

**Troubleshooting**:
- If you see import errors for `Qwen2_5_VLFlashAttention2`, this is expected in newer transformers versions (4.57+) and can be safely ignored
- For Ray-related issues, check Ray logs in `/tmp/ray/session_*/logs/`

## Contact

If you have any problems, please contact [Jian Xie](mailto:jianx0321@gmail.com).

## Citation

If our paper or related resources prove valuable to your research, we kindly ask for a citation.

<a href="https://github.com/TEAM-ARM/ARM"><img src="https://img.shields.io/github/stars/TEAM-ARM/ARM?style=social&label=ARM" alt="GitHub Stars"></a>

```bibtex
@article{xie2025arm2,
  title={ARM2: Adaptive Reasoning Model with Vision Understanding and Executable Code},
  author={Jian Xie and Zhendong Chu and Aoxiao Zhong and Kai Zhang and Mingzhe Han and Xing Fan and Jialie Shen and Qingsong Wen},
  journal={arXiv preprint arXiv:2510.08163},
  year={2025}
}
```