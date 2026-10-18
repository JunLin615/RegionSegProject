# RegionSegProject

Research scripts for identifying color-developed sample regions in surface-enhanced Raman spectroscopy (SERS) images. The workflow converts region annotations into binary masks, trains a DeepLabV3-ResNet50 segmentation model, and visualizes predicted regions.

This repository provides the image-segmentation training reference for automated Raman experiments. It does not control a microscope or implement an autonomous experiment workflow. Datasets and trained checkpoints are not included.

## Repository contents

| File | Purpose |
| --- | --- |
| `boundary2mask.py` | Convert VIA-style polygon, circle, and ellipse annotations into binary PNG masks. |
| `DeepLabV3.py` | Train a binary segmentation model and save the best checkpoint and TensorBoard logs. |
| `run_train.sh` | Example Slurm job for GPU training on KAUST Ibex. |
| `visualize_predictions.py` | Run inference and save masks and overlays with per-region centroids. |
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

To generate masks, pass the annotation file, source image directory, and mask output directory:

```bash
python boundary2mask.py --json annotations.json --img_dir dataset/images --out_dir dataset/masks
```

The converter accepts a VIA project JSON containing `_via_img_metadata`, or annotation records supplied directly as a dictionary or list. Each record's `regions` can be a dictionary or list. It uses the record's `filename`, falling back to the dictionary key, and expects PNG source images.

Supported shapes in `shape_attributes` are:

- `polygon`: `all_points_x` and `all_points_y`.
- `circle`: `cx`, `cy`, and `r`.
- `ellipse`: `cx`, `cy`, `rx`, `ry`, and `theta` (radians).

All supported regions in an image are combined into one binary mask. The output is named `<image_stem>_mask.png`. Image reading and writing support Unicode paths. Records with no regions and unreadable images are skipped, so create zero-valued masks separately for negative images and verify every training image has a corresponding mask.

## 2. Train the model

Run from the repository root:

```bash
python DeepLabV3.py --data_dir dataset --model_dir model_save
```

`--data_dir` and `--model_dir` are required. The data directory must contain `images/` and `masks/`; the model directory is created if needed. Other defaults are defined directly in the script:

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

Outputs are written to `<model_dir>/<YYYYMMDD_HHMMSS>/` (for example, `model_save/<YYYYMMDD_HHMMSS>/`): `best_model.pth` contains the checkpoint with the lowest validation loss, and `logs/` contains TensorBoard training/validation losses and the model graph.

```bash
tensorboard --logdir model_save --port 7008
```

### Slurm / KAUST Ibex

`run_train.sh` requests one A100 GPU, eight CPUs, 64 GB of memory, and twenty hours. It loads `cuda/12.2`, activates the `torch24` Conda environment, and runs from `/ibex/user/$USER/data/RegionSegProject`.

Set `DATA_DIR` and `MODEL_DIR` in the script (currently pointing to `dataset/yellow_260118`), and adapt the module, environment, working directory, and resources before submitting. The script also overrides `CUDA_VISIBLE_DEVICES` by querying A100 devices; verify that this respects your scheduler allocation on the target cluster.

Submit and monitor the job:

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
- `combined/<image_stem>_combined.png`: an overlay at the original resolution, with foreground tinted red and qualifying region centroids marked.

The visualization finds external contours after resizing the mask to the original image dimensions. It marks each contour centroid when the contour area is at least 50 square pixels and its moment is nonzero. This threshold affects centroid annotation only; smaller regions remain in the saved mask and overlay. Holes are ignored when finding contours, and the overlay also darkens the background.

Mask PNGs are produced by Matplotlib for visualization; an automated downstream pipeline should use explicit binary arrays and preserve image dimensions and coordinate conventions. Missing model files, missing image directories, and empty PNG directories cause the script to print a message and return.

## Integration and related repositories

- [WITecControlProject](https://github.com/JunLin615/WITecControlProject): existing microscope-control, AI-recognition, and frontend code that can run as separate services on separate computers, originally designed for human operation.
- [WITecAgent](https://github.com/JunLin615/WITecAgent): development repository for software designed for AI agents to operate WITec Raman experiments.
- [Raman_base](https://github.com/JunLin615/Raman_base): shared Raman data-processing utilities. Submit reusable Raman-processing contributions there as pull requests.

When adapting this model for microscope automation, define the image-to-stage coordinate mapping and treatment of multiple regions explicitly. Acquisition and experiment orchestration belong in the control project.

## Contributing

Open a pull request for improvements to annotation conversion, training, evaluation, or inference. Describe the change, how it was checked, and any dataset or checkpoint requirements. Keep local datasets, checkpoints, and machine-specific configuration out of contributions. The maintainer will review and merge contributions periodically.
