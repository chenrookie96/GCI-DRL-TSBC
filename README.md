# GCI-DRL-TSBC

**Global Coordination Integrated Deep Reinforcement Learning for Transit Scheduling with Bidirectional Coordination**

A deep reinforcement learning approach for optimizing bus scheduling in bidirectional transit systems, ensuring balanced departure frequencies and minimizing passenger waiting times while eliminating stranded passengers.

## Overview

This project implements a DRL-based transit scheduling system that:
- Coordinates bidirectional bus operations (upward and downward directions)
- Ensures equal departure counts between directions
- Minimizes average passenger waiting time
- Eliminates stranded passengers through intelligent scheduling
- Supports multiple omega (ω) parameters for different optimization objectives

## Key Features

- **Bidirectional Coordination**: Simultaneous optimization of both directions with hard constraints on departure balance
- **10-Dimensional State Space**: Comprehensive state representation including passenger demand, traffic conditions, and inter-direction coordination
- **4-Action Space**: Independent departure decisions for each direction
- **Deep Q-Network (DQN)**: 12-layer neural network with 500 neurons per layer
- **Flexible Reward Function**: Configurable omega parameter to balance departure frequency and waiting time

## Project Structure

```
GCI-DRL-TSBC/
├── drl_tsbc_brain.py          # DQN implementation
├── drl_tsbc_environment.py    # Bidirectional bus system environment
├── train_drl_tsbc.py          # Training script
├── inference_drl_tsbc.py      # Inference script
├── data_loader.py             # Data loading utilities
├── config_drl_tsbc.py         # Configuration parameters
├── test_data/                 # Test data directory
│   └── 208/                   # Line 208 data
├── saved_models/              # Trained models
│   ├── 208_omega500.pth       # ω=1/500 model
│   ├── 208_omega1000.pth      # ω=1/1000 model
│   ├── 208_omega2000.pth      # ω=1/2000 model
│   └── 208_omega3000.pth      # ω=1/3000 model
└── training_checkpoints/      # Episode-wise checkpoints (not tracked)
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- pandas
- numpy
- CUDA (optional, for GPU acceleration)

## Installation

```bash
git clone https://github.com/yourusername/GCI-DRL-TSBC.git
cd GCI-DRL-TSBC
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train_drl_tsbc.py
```

Training parameters can be modified in `train_drl_tsbc.py`:
- `omega_factor`: Set to 500, 1000, 2000, or 3000
- `max_episode`: Number of training episodes (default: 100)
- `train_counter`: Training frequency (default: 5)

### Inference

```bash
python inference_drl_tsbc.py
```

Make sure the `omega_factor` in `inference_drl_tsbc.py` matches the trained model.

## Results

Performance on Line 208 with different omega values:

| ω | Departures (Total) | Avg Waiting Time (min) | Stranded Passengers |
|---|-------------------|------------------------|---------------------|
| 1/500 | 152 (76+76) | 2.96 | 0 |
| 1/1000 | 138 (69+69) | 3.96 | 0 |
| 1/2000 | 122 (61+61) | 5.02 | 0 |
| 1/3000 | 114 (57+57) | 5.89 | 0 |

All models achieve:
- ✅ Zero stranded passengers
- ✅ Perfect departure balance (equal counts for both directions)
- ✅ Efficient scheduling adapted to passenger demand

## Key Innovations

1. **Bidirectional Coordination**: Unlike single-direction approaches, this system coordinates both directions simultaneously with hard constraints on departure balance.

2. **Enhanced State Space**: 10-dimensional state includes inter-direction information for better coordination.

3. **Flexible Optimization**: Omega parameter allows tuning between service frequency and waiting time based on operational requirements.

4. **Episode-wise Checkpointing**: Saves models after each training episode for optimal model selection.

## License

MIT License

## Citation

If you use this code in your research, please cite:

```
@software{gci_drl_tsbc,
  title={GCI-DRL-TSBC: Global Coordination Integrated Deep Reinforcement Learning for Transit Scheduling},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/GCI-DRL-TSBC}
}
```

## Acknowledgments

This project builds upon deep reinforcement learning techniques for transit optimization and extends them to bidirectional coordination scenarios.
