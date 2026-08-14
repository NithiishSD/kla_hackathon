# TeamName - KLA PS01

## Setup

```bash
pip install -r requirements.txt
```

## Evaluation

```bash
python evaluate.py <path_to_test_images_dir> <path_to_output_dir>
```

## Training

```bash
python train.py <path_to_data_root>
```

`data_root/` must contain:
```
data_root/
├── train/
│   ├── gt/*.npy
│   └── NoisyLR/*.npy
└── test/*.npy
```

