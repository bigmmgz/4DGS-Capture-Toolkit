# Multi-View-Capture-Toolkit-for-3D-4D-Gaussian-Splatting-Blender-Add-on-
A Blender add-on for generating multi-view datasets and per-frame COLMAP data for 3D Gaussian Splatting (3DGS) and dynamic (4DGS) reconstruction.

This tool provides a streamlined pipeline for:

* Multi-camera rig creation (studio-style capture)
* Batch rendering of multi-view images
* Automatic export of COLMAP-compatible data
* Frame-by-frame dataset generation for animated sequences

---

## Features

* **Studio Capture Dome**

  * Predefined hemispherical camera rig for character and human capture
  * Consistent multi-view coverage

* **Multi-view Rendering**

  * Render all cameras automatically
  * Supports per-frame rendering for animations

* **COLMAP Export**

  * Generates:

    * `cameras.txt`
    * `images.txt`
    * `points3D.txt`
  * Compatible with 3DGS pipelines

* **4DGS Dataset Generation**

  * Outputs structured frame folders:

    ```
    Frame0001/
    Frame0002/
    ...
    ```
  * Each frame contains multi-view images and COLMAP data

* **Resume Rendering**

  * Skips already rendered frames
  * Suitable for long sequences

---

## Project Structure

```text
4DGS-Data-Tool/
├── 4dgs_data_tool.py
├── assets/
│   └── predefined_objects.blend
├── README.md
└── LICENSE
```

---

## Installation

1. Clone or download the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/4DGS-Data-Tool.git
   ```

2. Open Blender

3. Navigate to:

   ```
   Edit → Preferences → Add-ons → Install
   ```

4. Select:

   ```
   4dgs_data_tool.py
   ```

5. Enable the add-on

---

## Usage

### 1. Add Camera Rig

* Open the N-panel → Cam Array
* Click **Add Preset Mesh**
* Select **Studio Capture Dome**

### 2. Create Cameras

* Select the dome mesh
* Click **Create Cameras**

### 3. Render Multi-view Images

* Set the output directory
* Click **Render Cameras**

### 4. Export COLMAP Data

* Click:

  * **Generate cameras.txt + images.txt**
  * **Generate points3D.txt**

### 5. Generate 4DGS Dataset (Animation)

* Assign the animated object
* Set frame range
* Click:

  ```
  Render 4DGS Sequence
  ```

Output structure:

```text
/output/
  Frame0001/
    images...
    cameras.txt
    images.txt
    points3D.txt
```

---

## Intended Use

This tool is designed for:

* 3D Gaussian Splatting (3DGS)
* Dynamic Gaussian Splatting (4DGS)
* Multi-view character and human capture datasets

---

## Notes

* Camera placement is based on mesh faces to ensure uniform coverage
* Best results are obtained with:

  * clean topology
  * centered objects
* Consistent camera configuration across frames supports stable temporal reconstruction

---

## TODO

* Integration with NeRF / Instant-NGP pipelines
* Direct export to training-ready formats for additional reconstruction frameworks
* Automated dataset validation tools

---

## Citation

If you use this tool in your research, please cite:

```text
@misc{xiaohan2025_4dgs_tool,
  author = {Xiaohan},
  title = {4DGS Data Tool},
  year = {2025},
  howpublished = {GitHub}
}
```

---

## License

MIT License (or your chosen license)

---

## Acknowledgements

* Blender Python API
* COLMAP
* 3D Gaussian Splatting (Kerbl et al. 2023)

---

## Contact

For questions or collaboration, please use GitHub Issues.
