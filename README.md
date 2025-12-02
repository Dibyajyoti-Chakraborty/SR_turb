# Swin Transformer Super-Resolution for Turbulence

Swin Transformer-based super-resolution model for 2D turbulence velocity fields (u, v components).

## Training

Train the model using the configuration file:

```bash
python train.py --config config.yaml
```

Optional command-line overrides:
```bash
python train.py --config config.yaml --batch_size 32 --num_epochs 100 --learning_rate 0.0001
```

## Configuration

All training parameters are specified in `config.yaml`. Key sections:

### Data Configuration
- `data_root`: Path to dataset directory (contains `train/` and `test/` subfolders)
- `upscale_factor`: Super-resolution upscaling factor (e.g., 8)
- `normalize_data`: Whether to normalize input data

### Model Configuration
- `img_size`: Input low-resolution image size
- `in_chans` / `out_chans`: Number of input/output channels (2 for u, v velocity components)
- `embed_dim`: Transformer embedding dimension
- `depths`: Number of transformer blocks per stage
- `num_heads`: Number of attention heads per stage
- `window_size`: Window size for windowed attention
- `upscale`: Upscaling factor

### Training Configuration
- `batch_size`: Training batch size
- `num_epochs`: Number of training epochs
- `learning_rate`: Initial learning rate
- `scheduler_type`: Learning rate scheduler (`cosine`, `step`, `plateau`, `exponential`)
- `warmup_epochs`: Number of warmup epochs
- `use_amp`: Enable mixed precision training
- `clip_grad_norm`: Gradient clipping threshold

### Logging Configuration
- `exp_name`: Experiment name
- `output_dir`: Output directory for checkpoints and logs
- `resume`: Whether to resume from checkpoint
- `resume_checkpoint`: Path to checkpoint file (or auto-find `best_model.pth`)

## Project Structure

- `train.py`: Main training script
- `swin_sr.py`: Swin Transformer model implementation
- `dataset.py`: Turbulence dataset loader
- `config.yaml`: Configuration file
- `viz.ipynb`: Demo visualization notebook showing super-resolution results after 1 epoch
- `experiments/`: Training outputs (checkpoints, logs)

