# TeamName - KLA PS01

## Setup
pip install -r requirements.txt

## Inference
python inference.py <path_to_test_images> <path_to_output_dir>

## Training (reproduce from scratch)
python train.py <path_to_data_root>

## Final model
weights/final_model.pt -- SemiRestoreNet_V2, dim=64, num_blocks=2,
scale_factor=2, fine-tuned with IntensityProfileLoss (weight=3.0),
edge_weight=0.5, lr=1e-5. See training log / report for full ablation history.