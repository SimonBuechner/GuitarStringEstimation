# Guitar String Estimation on GuitarSet via YMT3+

## Description
A project developed at IEM Graz during WS24/25. It uses YMT3 as a base for an algorithm for guitar string estimation on acoustic guitars.

This repository implements a system for automatic recognition and assignment of played guitar strings in audio recordings based on the [YourMT3](https://github.com/mimbres/YourMT3) model for F0-tracking. It is currently WIP and further instructions and restructuring will follow.

## Installation

1. Set up Git LFS:
```bash
git lfs install
```

2. Clone the repository:
```bash
git clone https://github.com/SimonBuechner/GuitarStringEstimation.git
cd GuitarStringEstimation
```

3. Install Python 3.11 in a new environment (recommended with conda or venv):
```bash
conda create -n guitar_string_estimation python=3.11
conda activate guitar_string_estimation
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Install the dataset:
```bash
python install_dataset.py
```
Follow the instructions in the CLI. For this project, only the GuitarSet dataset is needed, no checkpoints.


## Dataset

This project uses the [GuitarSet Dataset](https://guitarset.weebly.com/), which contains audio recordings and annotations of acoustic guitar performances.

## Based on

- [YourMT3](https://github.com/mimbres/YourMT3) - A modified Music Transcription Transformer model
- [GuitarSet](https://guitarset.weebly.com/) - Dataset for guitar recordings with annotations

## Project Structure

