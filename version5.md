Three pieces: (1) bake calibration stats into checkpoints going forward, (2) a one-time patch script so your existing baseline checkpoint doesn't need a retrain just to gain this, (3) the actual infer.py matching the required contract.


scripts done

ython bake_calib_into_checkpoints.py ./data ./checkpoints_baseline/best_model.pt 
python infer.py <test_images_dir> <output_dir>

pip install lpips   # one-time, on your laptop (downloads AlexNet weights, needs normal internet)
python train_hard_oversample.py ./data ./checkpoints_baseline/best_model.pt
python evaluate.py ./data ./checkpoints_hard_oversample/best_model.pt


complete the validation and packaging process for submission