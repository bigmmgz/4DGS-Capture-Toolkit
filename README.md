# 4DGS Capture Toolkit

A Blender add-on for generating multi-view datasets and per-frame COLMAP data for 3D Gaussian Splatting (3DGS) and dynamic (4DGS) reconstruction.

This toolkit provides a structured pipeline for synthesizing consistent multi-view observations of static and animated scenes, enabling efficient dataset creation for Gaussian-based neural rendering methods.

---

## Overview

Recent advances in 3D Gaussian Splatting and its dynamic extensions require high-quality multi-view datasets with accurate camera parameters. Generating such datasets from animated 3D assets is not straightforward and often requires multiple tools and manual processing steps.

This add-on integrates the full pipeline inside Blender by providing:

* A predefined studio-style multi-camera rig
* Automated multi-view rendering
* COLMAP-compatible data export
* Frame-wise dataset generation for animated sequences

<img width="1631" height="328" alt="gstool_pipeline" src="https://github.com/user-attachments/assets/46f5739f-489f-4628-9a78-0fd0d5797264" />


## Features

### Studio Capture Dome

A hemispherical multi-camera rig designed to approximate real-world capture setups. Cameras are distributed to provide uniform coverage of the subject.

### Multi-view Rendering

Automatically renders images from all cameras in the array. Supports both static scenes and animated sequences.

### COLMAP Export

Generates standard COLMAP input files:

* `cameras.txt`
* `images.txt`
* `points3D.txt`

Camera intrinsics and extrinsics are computed directly from Blender.

### 4DGS Dataset Generation

Produces structured frame-wise datasets:

```text id="f1k9zs"
Frame0001/
Frame0002/
...
```

Each frame contains:

* multi-view images
* camera parameters
* point cloud data

### Resume Rendering

Allows interrupted rendering processes to continue by skipping already processed frames.

---

## Project Structure

```text id="q2t6ak"
Multi-View-Capture-Toolkit-for-3D-4D-Gaussian-Splatting/
├── __init__.py
├── assets/
│   └── predefined_objects.blend
├── README.md
```

---

## Installation

1. Download this repository as a ZIP file from GitHub, or clone it locally.

2. In Blender, open:

   Edit → Preferences → Add-ons → Install

3. Select the ZIP file of this repository.

4. Enable the add-on in the add-on list.

5. Open the 3D Viewport sidebar (N-panel) and locate the **4DGS Tool** tab.

---

## Usage

### 1. Add Camera Rig

* Click **Add Preset Mesh**
* Select **Studio Capture Dome**

### 2. Create Camera Array

* Select the dome mesh
* Click **Create Cameras**

### 3. Render Multi-view Images

* Set the output directory
* Click **Render Cameras**

### 4. Export COLMAP Data

* Generate:

  * `cameras.txt`
  * `images.txt`
  * `points3D.txt`

### 5. Generate 4DGS Dataset

* Assign the animated object
* Set frame range
* Click:

```text id="8o5jtp"
Render 4DGS Sequence
```

Output structure:

```text id="wq8a9m"
/output/
  Frame0001/
    *.png
    cameras.txt
    images.txt
    points3D.txt
```

---

## Pipeline Summary

The dataset generation process follows:

1. Create a hemispherical camera rig
2. Place cameras based on mesh geometry
3. Render synchronized multi-view images
4. Export camera parameters and sparse geometry
5. Repeat for each frame in an animation sequence

This ensures spatial and temporal consistency required for Gaussian-based reconstruction methods.

---

## Intended Use

This toolkit is designed for:

* 3D Gaussian Splatting (3DGS)
* Dynamic Gaussian Splatting (4DGS)
* Multi-view character and human capture
* Synthetic dataset generation for neural rendering research

---

## TODO

- Extend the toolkit to support NeRF (Instant-NGP) dataset generation pipeline  
- Integrate direct training support via Postshot 

---

## Citation

If you use this tool in your research, please cite:

```text id="0l4v9n"
@article{sun2025far,
  title={From Far and Near: Perceptual Evaluation of Crowd Representations Across Levels of Detail},
  author={Sun, Xiaohan and O'Sullivan, Carol},
  journal={arXiv preprint arXiv:2510.20558},
  year={2025}
}
```

---
