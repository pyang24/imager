# Imager

A Tkinter-based desktop viewer and converter for detector image data in HDF5 and TIFF formats.

This tool is designed for fast browsing and playback of large detector datasets, with support for:
- Single-image files (TIFF or HDF5)
- Multi-image HDF5 stacks
- Pixel masking/interpolation for known bad pixels
- TIFF export
- Metadata extraction from `*_master.h5` files into `.metafile` text files

## Features

- Fast directory browser (lazy-loading tree, no heavy content scan)
- Natural file sorting (numeric-aware)
- Single-image mode and multi-image mode
- Movie playback with adjustable frame stepping
- Linear and log display scale
- Manual intensity controls (`vmin`/`vmax` + max pixel override)
- Click-to-zoom with configurable zoom window size
- Save filtered or full datasets as TIFF
- Extract and save master-file metadata (`.metafile`)

## Requirements

- Python 3.9+
- A desktop environment (Tkinter GUI)

Python packages:
- `numpy`
- `matplotlib`
- `h5py`
- `hdf5plugin`
- `tifffile`

## Installation

From the project folder, install dependencies:

```bash
pip install numpy matplotlib h5py hdf5plugin tifffile
```

## Run

```bash
python imager.py
```

## Basic Workflow

1. Launch the app.
2. Click **HDF5/TIFF File Directory** and choose a folder.
3. Optionally enter one or more filename keywords separated by semicolons (for example: `sam2_001; sam2_002`).
4. Choose mode:
   - Unchecked: single-image file mode
   - Checked: a single H5 file contains multiple images
5. Click **Load Filtered Files** (or use all files in the selected folder).
6. Select files/images in the list or move the slider.
7. Use **Play/Resume** and **Pause** for movie playback.
8. Use **Linear Scale** or **Log Scale**, plus intensity sliders.
9. Export results with:
   - **Save Filtered as Tiff & Metafile**
   - **Save All as Tiff & Metafile**

## File Handling Notes

- HDF5 image data is searched in these dataset paths (first match used):
  - `/entry/data/data`
  - `entry/data/data`
  - `/data`
  - `data`
- For HDF5 filtering/loading, data files are expected to include `_data_` in the filename.
- Master files (typically `*_master.h5`) are excluded from display lists but used for metadata export.

## Detector Masking Notes

The script includes detector-specific bad-pixel handling and interpolation logic. It also preserves detector module gap regions as `NaN` where applicable.

If detector bad-pixel maps change, update the pixel coordinate lists near the top of `imager.py`.

## Troubleshooting

- If no files appear:
  - Confirm the selected folder contains supported files (`.h5`, `.hdf5`, `.tif`, `.tiff`).
  - In HDF5 mode, confirm filenames follow expected data naming patterns (`_data_`).
- If images fail to load:
  - Verify expected HDF5 dataset paths exist.
- If metadata files are not generated:
  - Confirm corresponding `*_master.h5` files exist in the same folder.

## Acknowledgment

Developed by NSLS-II, Brookhaven National Laboratory.
