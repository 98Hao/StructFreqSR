# StructFreqSR

Official implementation of **StructFreqSR**, a lightweight causal video super-resolution (VSR) framework for real-time rendered videos. StructFreqSR combines a **Structure-Guided Frequency Modulation Module (SGFMM)** with an **Alignment-Aware Selective Memory Fusion Module (ASMFM)** to enhance spatial details and fuse temporal information. The network uses only low-resolution (LR) video frames and does not require additional G-buffer inputs.

## Project Structure

```text
StructFreqSR/
├── best.pth                       # Pretrained model weights
├── convert_images_to_h5_fixed.py  # Convert RGB images to HDF5 format
├── video_datasets.py              # Dataset loading
├── video_io.py                    # Video/image I/O utilities
├── video_metrics.py               # Evaluation metrics
├── video_test.py                  # Testing script
├── video_train.py                 # Training script
└── model/
    └── StructFreqSR.py           # StructFreqSR network
```

## Dataset

We use the rendering dataset released by **RDG**, *Efficient Video Super-Resolution for Real-time Rendering with Decoupled G-buffer Guidance*.

- **RDG project and dataset:** https://github.com/sunny2109/RDG

The experiments use 4× super-resolution pairs with LR frames at **480 × 270** and HR frames at **1920 × 1080**. Although the RDG dataset provides auxiliary G-buffer information, StructFreqSR only requires the LR frames as model inputs.

Download and prepare the dataset before running training or testing.

## Environment

Set up a Python environment with the dependencies required by the project scripts, including a compatible PyTorch installation. Dependency versions are not specified here; check the imports in the released scripts for any additional packages.

## Data Preparation (Optional)

To convert RGB images into HDF5 (`.h5`) format for faster data loading, open `convert_images_to_h5_fixed.py` and set `image_data_root` to your dataset path. Then run:

```bash
python convert_images_to_h5_fixed.py
```

Ensure that the paths and data format used by the training and testing scripts match your prepared dataset.

## Training

Open `video_train.py` and set `data_root` to your dataset path. Start training from the repository root:

```bash
python video_train.py
```

## Testing

The pretrained model weights are provided as **`best.pth`** in the repository root.

Open `video_test.py`, set `data_root` to your test dataset path, and ensure that the checkpoint path refers to `best.pth`. Run:

```bash
python video_test.py
```

## Acknowledgments

We thank the authors of [RDG](https://github.com/sunny2109/RDG) for making their work and dataset available. Our work uses the RDG dataset and draws on their research on video super-resolution for real-time rendering.
