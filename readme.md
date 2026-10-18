# RegionSegProject

Research scripts for identifying color-developed sample regions in surface-enhanced Raman spectroscopy (SERS) images. The workflow converts polygon annotations into binary masks, trains a DeepLabV3-ResNet50 segmentation model, and visualizes predicted regions.

This repository provides the image-segmentation training reference for automated Raman experiments. It does not control a microscope or implement an autonomous experiment workflow. Datasets and trained checkpoints are not included.

## Repository contents

| File | Purpose |
| --- | --- |
| `boundary2mask.py` | Convert VIA-style polygon annotations into binary PNG masks. |
| `DeepLabV3.py` | Train a binary segmentation model and save the best checkpoint and TensorBoard logs. |
| `run_train.sh` | Example Slurm job for GPU training on KAUST Ibex. |
| `visualize_predictions.py` | Run inference and save masks and overlays with a foreground centroid. |
| `visualize_predictions copy.py` | Earlier visualization variant that also darkens the background. |
| `test.py` | Small image color-conversion utility; not an automated test suite. |

Historical notes referred to date-suffixed scripts that are not present in this checkout. These instructions use the actual filenames in the repository.

## Environment

Use a Python environment with compatible versions of PyTorch and torchvision. The scripts also import Pillow, NumPy, OpenCV, Matplotlib, tqdm, and TensorBoard.

```bash
python -m pip install torch torchvision pillow numpy opencv-python matplotlib tqdm tensorboard
```

For GPU training, install a PyTorch build compatible with your CUDA environment. Dependencies are not pinned, and the scripts use the legacy torchvision `pretrained` argument, so verify compatibility with your chosen versions. Initial model construction may download pretrained weights.

## 1. Prepare images and masks

The training script expects this layout relative to the working directory:

```text
RegionSegProject/
  DeepLabV3.py
  run_train.sh
  dataset/
    images/
      sample_001.png
      sample_002.png
    masks/
      sample_001_mask.png
      sample_002_mask.png
  model_save/                 # Created during training
```

Each input PNG must have a matching grayscale mask named `<image_stem>_mask.png`, with foreground 255 and background 0. Keep the images directory limited to matching PNG inputs: the training dataset enumerates every file without filtering by extension.

To generate masks, edit these variables in `boundary2mask.py`:

- `boundary_filename`: annotation JSON path.
- `images_filename`: source image directory.
- `output`: mask output directory.

```bash
python boundary2mask.py
```

The converter expects a top-level mapping of image records, each containing a `regions` dictionary. Each polygon region must contain `shape_attributes` with `name: "polygon"`, `all_points_x`, and `all_points_y`. All polygons for an image are combined into a single binary mask.

The current converter handles polygons only; circles and other JSON export layouts require adaptation. Records with no regions are skipped, so create zero-valued masks separately for negative images. Verify generated filenames against the required pairing above, especially when annotation keys contain suffixes after `.png`. Prefer ASCII paths for the converter's OpenCV file operations; use raw strings or forward slashes when editing Windows paths in Python.

## 2. Train the model

Run from the repository root:

```bash
python DeepLabV3.py
```

The defaults are defined directly in the script:

| Setting | Value |
| --- | --- |
| Architecture | Pretrained DeepLabV3 with a ResNet50 backbone |
| Output | One foreground logit channel |
| Input size | 640 pixels high x 1024 pixels wide |
| Dataset split | Random 80% training / 20% held-out |
| Batch size | 8 |
| Optimizer | Adam, learning rate `1e-4` |
| Loss | `BCEWithLogitsLoss` |
| Epochs | 200 |
| Device | CUDA when available, otherwise CPU |

The held-out split is called `test` in the code, but it selects the best checkpoint and therefore serves as validation. Use a separate test set for independent evaluation. The split is not seeded; ensure both training and validation splits contain data.

Outputs are written to `model_save/<YYYYMMDD_HHMMSS>/`: `best_model.pth` contains the checkpoint with the lowest validation loss, and `logs/` contains TensorBoard training/validation losses and the model graph.

```bash
tensorboard --logdir model_save --port 7008
```

### Slurm / KAUST Ibex

`run_train.sh` requests one GPU, eight CPUs, and five hours. It loads `cuda/12.2`, activates the `torch24` Conda environment, and runs from `/ibex/user/$USER/data/RegionSegProject`.

Adapt the module, environment, working directory, and resource settings to your cluster account before submitting:

```bash
sbatch run_train.sh
squeue -u "$USER"
```

The example writes scheduler output and errors to `output.txt` and `error.txt`.

## 3. Visualize predictions

Edit the variables at the end of `visualize_predictions.py`:

- `model_dir`: timestamped directory containing `best_model.pth`.
- `data_dir`: input PNG directory.
- `save_dir`: prediction output directory.

```bash
python visualize_predictions.py
```

Inference resizes images to the training dimensions, applies sigmoid, and thresholds predictions at 0.5. Outputs include:

- `masks/<image_stem>_pred.png`: a mask visualization at inference resolution.
- `combined/<image_stem>_combined.png`: an overlay at the original resolution, with foreground tinted red and its centroid marked.

The centroid covers all foreground pixels together, not each disconnected region separately. An empty mask is displayed with coordinates `(0, 0)`. Mask PNGs are produced by Matplotlib for visualization; an automated downstream pipeline should use explicit binary arrays and preserve image dimensions and coordinate conventions.

## Integration and related repositories

- [WITecControlProject](https://github.com/JunLin615/WITecControlProject): existing microscope-control, AI-recognition, and frontend code that can run as separate services on separate computers, originally designed for human operation.
- [WITecAgent](https://github.com/JunLin615/WITecAgent): development repository for software designed for AI agents to operate WITec Raman experiments.
- [Raman_base](https://github.com/JunLin615/Raman_base): shared Raman data-processing utilities. Submit reusable Raman-processing contributions there as pull requests.

When adapting this model for microscope automation, define the image-to-stage coordinate mapping and treatment of multiple regions explicitly. Acquisition and experiment orchestration belong in the control project.

## Contributing

Open a pull request for improvements to annotation conversion, training, evaluation, or inference. Describe the change, how it was checked, and any dataset or checkpoint requirements. Keep local datasets, checkpoints, and machine-specific configuration out of contributions. The maintainer will review and merge contributions periodically.
