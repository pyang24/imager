"""

This version uses custom built directory browsing without scanning content, saving a ton of time..
which is much faster than the previous version that can be very slow on network drives or when there are thousands of files.

This is a combined version for both 500K and 1M pixel detectors, with automatic shape detection.

This also converts master.h5 metadata extraction to a separate .metafile text file when saving 000001.h5 files to tiff.

"""

import os
import h5py
import hdf5plugin
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from tkinter import (
    Tk, Toplevel, Listbox, Button, filedialog, SINGLE, END, Scrollbar, RIGHT, Y, BOTH,
    Frame, Scale, Label, HORIZONTAL, StringVar, Entry, Checkbutton, BooleanVar
)
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import (LogNorm, Normalize)
import gc
import re
import tkinter.ttk as ttk
import threading

# Track scheduled callbacks to prevent orphaned commands
scheduled_callbacks = []
current_folder = ''
current_image_data = None
current_colorbar = None
is_playing = False
playback_speed_ms = 50   # milliseconds between frames (adjust as needed) - reduced for smoother playback
current_index = 0
manual_vmax_enabled = False
slider_programmatic = False
zoom_mode_enabled = False
zoom_center = None  # (x, y) or None
zoom_size_x = 50
zoom_size_y = 50
# --- Data containers ---
files_data = {}
output_folder = ""
collapsed_files = {}
current_filename = None
windows_manually_closed = {
    'image_plot': False,
}
# Performance optimization: cache file info to avoid repeated loading
file_info_cache = {}  # filepath -> {'image_count': int, 'last_modified': float}
# Performance cache for multiimage mode
file_image_counts = {}  # Cache: filepath -> number of images
file_load_cache = {}    # Cache: (filepath, image_index) -> image data
multi_mode_cache = {}   # Cache for multi-mode calculations (total images, sorted files)

# Detector-specific excluded pixel lists
excluded_pixels_c = [    # Add any new dead pixels (x, y) for 500K detector here
        (0, 0), (0, 513), (142, 204), (255, 513), (256, 513), 
        (451, 223), (496, 288), (497, 288), (497, 289), (498, 287), (498, 288), (498, 289),
        (499, 287), (499, 288), (499, 289), (500, 287), (500, 288), (500, 289),
        (501, 288), (501, 289), (502, 288), (517, 266), (755, 125), (757, 493), (758, 493), (770, 513), (771, 513),
        (772, 513), (773, 513), (774, 513), (1029, 0), (1029, 513)
        ]
excluded_pixels_d = [    # Add any new dead pixels for 1M detector here
        (382, 975), (754, 58)
        ]

def get_image_shape(data):
    """Return the 2D image shape (rows, cols).
    - If `data` is 2D, return its shape.
    - If `data` is 3D (stack), return the shape of the first frame.
    - Otherwise, return None.
    """
    try:
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                return data.shape
            if data.ndim == 3 and data.shape[0] > 0:
                return data[0].shape
    except Exception:
        pass
    return None

def safe_after(delay, callback):
    """Safely schedule a tkinter after callback with error handling"""
    try:
        if root and root.winfo_exists():
            callback_id = root.after(delay, callback)
            scheduled_callbacks.append(callback_id)
            return callback_id
    except Exception:
        pass  # Ignore errors if root is destroyed
    return None

def cleanup_callbacks():
    """Cancel all scheduled callbacks"""
    global scheduled_callbacks
    for callback_id in scheduled_callbacks:
        try:
            root.after_cancel(callback_id)
        except Exception:
            pass  # Ignore errors if already cancelled or root destroyed
    scheduled_callbacks.clear()

def process_gui_events():
    """Process pending GUI events without blocking - enhanced responsiveness like hdf27integration_log.py"""
    try:
        root.update_idletasks()
        root.update()  # Process all pending events, not just idle tasks
    except Exception:
        pass  # Ignore errors if root is destroyed

def cleanup_memory_caches():
    """Clean up memory caches to prevent memory leaks"""
    global file_image_counts, file_load_cache, file_info_cache
    file_image_counts.clear()
    file_load_cache.clear()
    file_info_cache.clear()

def natural_sort_key(filename):
    """
    Generate sorting key for natural (numerical) ordering of filenames.
    Converts numeric parts to integers for proper sorting.
    Handles filenames that start with numbers correctly.
    e.g., file_000001.h5 comes before file_000010.h5
    """
    parts = re.split(r'(\d+)', filename)
    # Filter out empty strings that can occur when filename starts with a number
    parts = [part for part in parts if part]
    # Use tuples to ensure proper comparison: (0, int) for numbers, (1, str) for text
    result = []
    for text in parts:
        if text.isdigit():
            result.append((0, int(text)))  # Numbers sort before text
        else:
            result.append((1, text.lower()))  # Text sorts after numbers
    return result

def get_clean_filename(display_name):
    """
    Extract clean filename from display name that might have multiimage prefixes.
    Handles: '[+] filename', '[-] filename', '    filename [0]'
    """
    if display_name.startswith('[+] ') or display_name.startswith('[-] '):
        return display_name[4:]
    elif display_name.startswith('    '):  # Expanded image entries
        clean = display_name.strip()
        if '[' in clean and ']' in clean:
            return clean.split(' [')[0]
        return clean
    return display_name

def ensure_slider_range_correct():
    """
    Ensure slider range matches actual total images in multi-mode.
    This prevents 'Image index out of range' errors.
    """
    if not multiimage_mode.get() or not files_data:
        return
    
    # If we don't have cached counts, do the smart counting now
    if not file_image_counts:
        first_fname = sorted(files_data.keys(), key=natural_sort_key)[0]
        first_filepath = files_data[first_fname]
        images_per_file = get_file_image_count_cached(first_filepath)
        
        # Set all files to have the same count
        for fname, filepath in files_data.items():
            file_image_counts[filepath] = images_per_file
        
        expected_total = images_per_file * len(files_data)
        print(f"Calculated: {images_per_file} images per file, {expected_total} total images")
    else:
        # Calculate expected total using cached values
        images_per_file = next(iter(file_image_counts.values()))
        expected_total = images_per_file * len(files_data)
    
    # Check if slider range matches expected total
    current_max = image_slider.cget('to')
    if current_max != expected_total - 1:
        print(f"Correcting slider range from {current_max + 1} to {expected_total}")
        image_slider.config(to=max(0, expected_total - 1))

def get_file_image_count_cached(filepath):
    """
    Get the number of images in a file using caching for performance.
    Only counts images, doesn't load them.
    """
    global file_info_cache
    
    # Check if file exists
    if not os.path.exists(filepath):
        return 0
    
    # Get file modification time
    mtime = os.path.getmtime(filepath)
    
    # Check cache
    if filepath in file_info_cache:
        cached_info = file_info_cache[filepath]
        if cached_info['last_modified'] == mtime:
            return cached_info['image_count']
    
    # Count images and update cache
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".h5", ".hdf5"]:
        count = count_images_in_h5_file(filepath)
    else:
        count = 1 if ext in [".tif", ".tiff"] else 0
    
    file_info_cache[filepath] = {
        'image_count': count,
        'last_modified': mtime
    }
    
    return count

def count_images_in_h5_file(filepath):
    """
    Efficiently count images in HDF5 file without loading them into memory.
    Returns the number of images.
    """
    try:
        with h5py.File(filepath, 'r') as hdf:
            possible_paths = [
                '/entry/data/data', 'entry/data/data', '/data', 'data'
            ]
            for path in possible_paths:
                if path in hdf:
                    data = hdf[path]
                    if data.ndim == 3:
                        return data.shape[0]  # Number of images in stack
                    elif data.ndim == 2:
                        return 1  # Single image
                    break
        return 0
    except Exception as e:
        print(f"Error counting images in {filepath}: {e}")
        return 0

def apply_mask(data):
    # Choose excluded pixel list by detector image shape
    # Note: shape is (rows=y, cols=x)
    shape = get_image_shape(data) or data.shape
    if shape == (514, 1030):
        excluded = excluded_pixels_c
    elif shape == (1065, 1030):
        # Per user instruction, use excluded_pixels_c for 1064x1029 as well
        excluded = excluded_pixels_d
    else:
        # Fallback: no predefined excluded pixels
        excluded = []
    
    # Create a copy of the data to work with
    data_masked = data.copy().astype(float)
    
    # Define detector module gap region correctly for 1029 x 1064 detector
    gap_region = np.zeros(data.shape, dtype=bool)
    if data.shape[0] > 550:  # Height check (1064 > 550 ✓)
        # Gap region: all x pixels (0 to 1028) at y = 514-550 
        gap_region[514:551, :] = True  # y from 514 to 550 (inclusive), all x

    # Set gap region to NaN explicitly
    data_masked[gap_region] = np.nan
    
    # Create mask for ONLY the pixels that need interpolation (excluding gap)
    pixels_to_interpolate = np.zeros(data.shape, dtype=bool)
    
    # 1. Add specified dead pixels ONLY
    for x, y in excluded:
        if 0 <= y < data.shape[0] and 0 <= x < data.shape[1]:
            pixels_to_interpolate[y, x] = True
    
    # 2. Add NaN pixels ONLY (but not in gap region)
    original_nan_mask = np.isnan(data) & ~gap_region
    pixels_to_interpolate |= original_nan_mask
    
    # Interpolate bad pixels row by row
    window_size = 15  # Number of adjacent pixels to use for fitting
    
    for row in range(data.shape[0]):
        # Skip gap rows entirely
        if gap_region[row, :].all():
            continue
            
        # Find bad pixels in this row (excluding gap)
        bad_pixels_in_row = np.where(pixels_to_interpolate[row, :])[0]
        
        if len(bad_pixels_in_row) == 0:
            continue  # No bad pixels in this row
        
        for bad_idx in bad_pixels_in_row:
            # Double check this pixel is not in gap
            if gap_region[row, bad_idx]:
                continue
                
            # Find valid pixels around the bad pixel for interpolation
            left_start = max(0, bad_idx - window_size)
            right_end = min(data.shape[1], bad_idx + window_size + 1)
            
            # Get indices of nearby pixels
            nearby_indices = np.arange(left_start, right_end)
            
            # Filter to only valid pixels (not bad, not in gap, not the current pixel)
            valid_mask = (
                ~pixels_to_interpolate[row, nearby_indices] &  # Not a bad pixel
                ~gap_region[row, nearby_indices] &    # Not in gap
                ~np.isnan(data_masked[row, nearby_indices]) &  # Not NaN
                (nearby_indices != bad_idx)           # Not the current bad pixel
            )
            
            valid_nearby = nearby_indices[valid_mask]
            
            if len(valid_nearby) < 2:
                # Not enough valid pixels for interpolation, use fallback
                # Find the nearest single valid pixel
                left_search = bad_idx - 1
                right_search = bad_idx + 1
                fallback_value = 0.1
                
                # Search left
                while left_search >= 0:
                    if (not pixels_to_interpolate[row, left_search] and 
                        not gap_region[row, left_search] and 
                        not np.isnan(data_masked[row, left_search])):
                        fallback_value = data_masked[row, left_search]
                        break
                    left_search -= 1
                
                # Search right if no left neighbor found
                if fallback_value == 0.1:
                    while right_search < data.shape[1]:
                        if (not pixels_to_interpolate[row, right_search] and 
                            not gap_region[row, right_search] and 
                            not np.isnan(data_masked[row, right_search])):
                            fallback_value = data_masked[row, right_search]
                            break
                        right_search += 1
                
                data_masked[row, bad_idx] = max(0.1, fallback_value)
                
            elif len(valid_nearby) < 4:
                # Use linear interpolation for few points
                x_coords = valid_nearby
                y_values = data_masked[row, valid_nearby]
                
                # Simple linear interpolation
                if len(valid_nearby) == 2:
                    x1, x2 = x_coords[0], x_coords[1]
                    y1, y2 = y_values[0], y_values[1]
                    # Linear interpolation formula
                    interpolated_val = y1 + (y2 - y1) * (bad_idx - x1) / (x2 - x1)
                else:
                    # Use numpy's linear interpolation for 3 points
                    interpolated_val = np.interp(bad_idx, x_coords, y_values)
                
                # Ensure reasonable bounds
                min_val = max(0.1, np.min(y_values) * 0.5)
                max_val = np.max(y_values) * 2.0
                interpolated_val = np.clip(interpolated_val, min_val, max_val)
                
                data_masked[row, bad_idx] = interpolated_val
                
            else:
                # Use polynomial fitting for many points
                try:
                    x_coords = valid_nearby
                    y_values = data_masked[row, valid_nearby]
                    
                    # Use polynomial order based on number of points (max 3)
                    poly_order = min(3, len(valid_nearby) - 1)
                    
                    # Fit polynomial
                    coeffs = np.polyfit(x_coords, y_values, poly_order)
                    
                    # Evaluate polynomial at bad pixel position
                    interpolated_val = np.polyval(coeffs, bad_idx)
                    
                    # Ensure reasonable bounds
                    min_val = max(0.1, np.min(y_values) * 0.1)
                    max_val = np.max(y_values) * 2.0
                    interpolated_val = np.clip(interpolated_val, min_val, max_val)
                    
                    data_masked[row, bad_idx] = interpolated_val
                    
                except (np.linalg.LinAlgError, np.RankWarning, ValueError) as e:
                    # Fallback to mean of valid nearby pixels
                    y_values = data_masked[row, valid_nearby]
                    mean_val = np.mean(y_values[~np.isnan(y_values)])
                    data_masked[row, bad_idx] = max(0.1, mean_val)
    
    # Final cleanup: ensure gap region stays NaN
    data_masked[gap_region] = np.nan
    
    # Replace any remaining problematic values outside gap
    remaining_problems = (
        (np.isnan(data_masked) | (data_masked == 0) | (data_masked >= 4e9)) & 
        ~gap_region
    )
    data_masked[remaining_problems] = 0.1
        
    return data_masked

def apply_mask_with_shape_check(data):
    """Wrapper to validate shape before masking.
    Logs a short note if shape is unknown or non-2D, then applies mask safely.
    """
    shape = get_image_shape(data)
    if shape is None:
        try:
            print("Warning: Unknown image shape; applying generic mask.")
        except Exception:
            pass
    return apply_mask(data)

def load_single_image_from_h5_stack(filepath, image_index):
    """
    Load a single image from an HDF5 stack by index without loading the entire stack.
    Returns None if the image cannot be loaded.
    """
    try:
        with h5py.File(filepath, 'r') as hdf:
            possible_paths = [
                '/entry/data/data', 'entry/data/data', '/data', 'data'
            ]
            for path in possible_paths:
                if path in hdf:
                    data = hdf[path]
                    if data.ndim >= 2:
                        if data.ndim == 3 and image_index < data.shape[0]:
                            # Load only the specific slice we need
                            return data[image_index]
                        elif data.ndim == 2 and image_index == 0:
                            return data[()]
        return None
    except Exception as e:
        print(f"Failed to load image {image_index} from {filepath}: {e}")
        return None

def read_h5_stack(filepath):
    """
    Read all images from an HDF5 file that stacks multiple images (3D array).
    Returns a list of 2D numpy arrays (one per image).
    """
    try:
        with h5py.File(filepath, 'r') as hdf:
            possible_paths = [
                '/entry/data/data', 'entry/data/data', '/data', 'data'
            ]
            data = None
            for path in possible_paths:
                if path in hdf:
                    data = hdf[path]
                    break
            if data is not None and data.ndim >= 2:
                arr = data[()]
                if arr.ndim == 3:
                    return [arr[i] for i in range(arr.shape[0])]
                elif arr.ndim == 2:
                    return [arr]
    except Exception as e:
        print(f"Failed to read HDF5 stack from {filepath}: {e}")
    return []

def read_image_file_multi(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".h5", ".hdf5"]:
        return read_h5_stack(filepath)
    return []

def has_image_data(filepath):
    """Check if an HDF5 file contains actual image data."""
    try:
        with h5py.File(filepath, 'r') as hdf:
            possible_paths = [
                '/entry/data/data', 'entry/data/data', '/data', 'data'
            ]
            for path in possible_paths:
                if path in hdf:
                    data = hdf[path]
                    # Check if it's actually image data (2D or 3D array)
                    if data.ndim >= 2:
                        return True
        return False
    except Exception:
        return False

def get_master_file_path(data_filename, folder):
    """
    Find the corresponding master file for a data file.
    Converts data file names like 'sample_data_000001.h5' 
    to master file names like 'sample_master.h5'
    """
    # Extract the base name by removing _data_XXXXXX.h5 suffix
    if '_data_' in data_filename:
        base_name = data_filename.split('_data_')[0]
        master_filename = base_name + '_master.h5'
        master_path = os.path.join(folder, master_filename)
        if os.path.exists(master_path):
            return master_path
    return None

def extract_metadata_from_master(master_filepath):
    """
    Extract all metadata from a master HDF5 file.
    Returns a dictionary with all metadata found in the file,
    including image IDs from entry/data branches.
    """
    metadata = {}
    image_ids = []
    
    try:
        with h5py.File(master_filepath, 'r') as hdf:
            # First, extract image IDs from entry/data branches
            try:
                if 'entry/data' in hdf:
                    data_group = hdf['entry/data']
                    # Look for data_000001, data_000002, etc.
                    for key in sorted(data_group.keys()):
                        if key.startswith('data_'):
                            image_ids.append(key)
                elif '/entry/data' in hdf:
                    data_group = hdf['/entry/data']
                    for key in sorted(data_group.keys()):
                        if key.startswith('data_'):
                            image_ids.append(key)
            except Exception as e:
                print(f"Note: Could not extract image IDs from {os.path.basename(master_filepath)}: {e}")
            
            # Store image IDs as metadata
            if image_ids:
                metadata['_IMAGE_IDS'] = image_ids
                metadata['_IMAGE_COUNT'] = len(image_ids)
            
            def visit_func(name, obj):
                """Recursively visit all items in the HDF5 file."""
                if isinstance(obj, h5py.Dataset):
                    try:
                        # Get the data value
                        value = obj[()]
                        # Decode bytes to string if necessary
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        # Handle numpy arrays
                        elif isinstance(value, np.ndarray):
                            if value.size == 1:
                                value = value.item()
                                if isinstance(value, bytes):
                                    value = value.decode('utf-8')
                            else:
                                # For arrays, convert to list
                                value = value.tolist()
                        metadata[name] = value
                    except Exception as e:
                        metadata[name] = f"<Error reading: {e}>"
                elif isinstance(obj, h5py.Group):
                    # Store group attributes if any
                    if obj.attrs:
                        for attr_name, attr_value in obj.attrs.items():
                            full_name = f"{name}/@{attr_name}"
                            try:
                                if isinstance(attr_value, bytes):
                                    attr_value = attr_value.decode('utf-8')
                                metadata[full_name] = attr_value
                            except Exception:
                                metadata[full_name] = str(attr_value)
            
            hdf.visititems(visit_func)
            
            # Also get root attributes
            for attr_name, attr_value in hdf.attrs.items():
                try:
                    if isinstance(attr_value, bytes):
                        attr_value = attr_value.decode('utf-8')
                    metadata[f"/@{attr_name}"] = attr_value
                except Exception:
                    metadata[f"/@{attr_name}"] = str(attr_value)
                    
    except Exception as e:
        print(f"Error extracting metadata from {os.path.basename(master_filepath)}: {e}")
        return None
    
    return metadata

def save_metadata_to_txt(metadata, output_path):
    """
    Save metadata dictionary to a text file in a readable format.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("HDF5 Master File Metadata\n")
            f.write("=" * 80 + "\n\n")
            
            # Display image IDs first if available
            if '_IMAGE_IDS' in metadata:
                f.write("IMAGE IDs IN MASTER FILE:\n")
                f.write("-" * 80 + "\n")
                image_ids = metadata['_IMAGE_IDS']
                if len(image_ids) <= 20:
                    for img_id in image_ids:
                        f.write(f"  {img_id}\n")
                else:
                    f.write(f"  Total: {len(image_ids)} images\n")
                    f.write(f"  First 10: {', '.join(image_ids[:10])}\n")
                    f.write(f"  Last 10: {', '.join(image_ids[-10:])}\n")
                f.write("\n" + "=" * 80 + "\n\n")
            
            # Sort keys for consistent output, but skip the special _IMAGE_IDS and _IMAGE_COUNT keys
            sorted_keys = sorted([k for k in metadata.keys() if not k.startswith('_')])
            
            for key in sorted_keys:
                value = metadata[key]
                # Format the output
                if isinstance(value, (list, tuple)) and len(value) > 10:
                    f.write(f"{key}:\n")
                    f.write(f"  [Array with {len(value)} elements]\n")
                    f.write(f"  First 10: {value[:10]}\n\n")
                else:
                    f.write(f"{key}:\n")
                    f.write(f"  {value}\n\n")
        
        return True
    except Exception as e:
        print(f"Error saving metadata to {output_path}: {e}")
        return False

def refresh_file_list_multi():
    global current_folder
    if not current_folder:
        print("No folder loaded yet.")
        return

    keyword_input = file_filter_var.get().strip()

    if not keyword_input:
        matched_files = [
            fname for fname in os.listdir(current_folder)
            if (fname.lower().endswith((".h5", ".hdf5")) and
                "master" not in fname.lower() and  # Exclude master files
                "_data_" in fname.lower())  # Only include data files
        ]
        # Sort files naturally
        matched_files = sorted(matched_files, key=natural_sort_key)
    else:
        keywords = [kw.strip() for kw in keyword_input.split(';') if kw.strip()]
        matched_files = [
            fname for fname in os.listdir(current_folder)
            if (fname.lower().endswith((".h5", ".hdf5")) and 
                any(kw in fname for kw in keywords) and
                "master" not in fname.lower() and  # Exclude master files
                "_data_" in fname.lower())  # Only include data files
        ]
        # Sort files naturally
        matched_files = sorted(matched_files, key=natural_sort_key)

    new_files = 0
    for fname in matched_files:
        if fname not in files_data:
            filepath = os.path.join(current_folder, fname)
            files_data[fname] = filepath
            new_files += 1

    # Smart count update: if we have any files, count one and apply to all
    if files_data and new_files > 0:
        first_fname = sorted(files_data.keys(), key=natural_sort_key)[0]
        first_filepath = files_data[first_fname]
        images_per_file = get_file_image_count_cached(first_filepath)
        
        # Update counts for all files
        for fname, filepath in files_data.items():
            file_image_counts[filepath] = images_per_file
        
        # Update slider range
        total_images = images_per_file * len(files_data)
        image_slider.config(to=max(0, total_images - 1))
        print(f"Updated: {images_per_file} images per file, {total_images} total images")

    print(f"Refresh complete. {new_files} new file(s) added.")
    update_file_list_display_multi()

def update_file_list_display_multi():
    file_list.delete(0, END)
    for fname in sorted(files_data.keys(), key=natural_sort_key):
        filepath = files_data[fname]  # This is now a file path
        
        # Initialize collapsed state if not set
        if fname not in collapsed_files:
            collapsed_files[fname] = True
        
        if collapsed_files[fname]:
            # File is collapsed - just show the [+] without counting images yet
            file_list.insert(END, f"[+] {fname}")
        else:
            # File is expanded - use cached count or default
            image_count = file_image_counts.get(filepath, 1)  # Default to 1 if not cached
            
            file_list.insert(END, f"[-] {fname}")
            for i in range(image_count):
                file_list.insert(END, f"    {fname} [{i}]")
        
    
    # Don't calculate total images here - it makes tree loading slow
    # Slider range will be updated when needed (e.g., during movie playback)

def on_file_list_click(event):
    idx = file_list.nearest(event.y)
    entry = file_list.get(idx)
    # Multi-image mode
    if multiimage_mode.get():
        if entry.startswith("[+] "):
            fname = entry[4:]
            collapsed_files[fname] = False
            update_file_list_display_multi()
            file_list.see(idx)  # <-- Keep the view at the clicked cluster
            return "break"  # Prevent selection change
        elif entry.startswith("[-] "):
            fname = entry[4:]
            collapsed_files[fname] = True
            update_file_list_display_multi()
            file_list.see(idx)  # <-- Keep the view at the clicked cluster
            return "break"  # Prevent selection change
    # Single-image mode: nothing to do for headers
    return

def load_all_files_multi():
    """Load all multiimage HDF5 files in the current directory without filtering."""
    global current_folder
    if not current_folder or not os.path.isdir(current_folder):
        print("Invalid directory. Please select a valid directory.")
        try:
            selected_dir_var.set("Invalid directory. Please select a valid directory.")
        except Exception:
            pass
        return

    file_list.delete(0, END)
    files_data.clear()
    multi_mode_cache.clear()  # Clear cache for fresh calculations
    collapsed_files.clear()   # Clear collapsed state

    # Load all HDF5 files (exclude master files)
    matched_files = [
        fname for fname in os.listdir(current_folder)
        if (fname.lower().endswith((".h5", ".hdf5")) and
            "master" not in fname.lower() and  # Exclude master files
            "_data_" in fname.lower())  # Only include data files
    ]
    
    # Sort files naturally to ensure proper chronological order
    matched_files = sorted(matched_files, key=natural_sort_key)

    # Clear caches for new file set
    file_image_counts.clear()
    file_load_cache.clear()
    
    loaded_files = 0
    for fname in matched_files:
        filepath = os.path.join(current_folder, fname)
        if fname.lower().endswith((".h5", ".hdf5")):
            files_data[fname] = filepath
            loaded_files += 1

    print(f"Loaded {loaded_files} file(s) for processing.")
    update_file_list_display_multi()
    
    # Smart total image calculation: count one file and multiply by file count
    if loaded_files > 0:
        # Get first file to determine images per file
        first_fname = sorted(files_data.keys(), key=natural_sort_key)[0]
        first_filepath = files_data[first_fname]
        images_per_file = get_file_image_count_cached(first_filepath)
        
        # Calculate total images efficiently
        total_images = images_per_file * loaded_files
        
        # Set all files to have the same count (since they should be identical)
        for fname, filepath in files_data.items():
            file_image_counts[filepath] = images_per_file
        
        # Set up slider with correct range (avoid loading any image here)
        image_slider.config(to=max(0, total_images - 1))
        
        print(f"Images per file: {images_per_file}, Total images: {total_images}")
    else:
        image_slider.config(to=0)
        image_slider.set(0)
        image_slider.config(label="No images loaded")

def load_filtered_files_multi():
    global current_folder
    if not current_folder or not os.path.isdir(current_folder):
        print("Invalid directory. Please select a valid directory.")
        try:
            selected_dir_var.set("Invalid directory. Please select a valid directory.")
        except Exception:
            pass
        return

    file_list.delete(0, END)
    files_data.clear()
    multi_mode_cache.clear()  # Clear cache for fresh calculations
    collapsed_files.clear()   # Clear collapsed state

    keyword_input = file_filter_var.get().strip()
    if not keyword_input:
        print("Please enter one or more keywords separated by semicolons.")
        return

    keywords = [kw.strip() for kw in keyword_input.split(';') if kw.strip()]
    if not keywords:
        print("No valid keywords provided.")
        return

    matched_files = [
        fname for fname in os.listdir(current_folder)
        if (fname.lower().endswith((".h5", ".hdf5")) and 
            any(kw in fname for kw in keywords) and
            "master" not in fname.lower())  # Exclude master files
    ]
    
    # Sort files naturally to ensure proper chronological order
    matched_files = sorted(matched_files, key=natural_sort_key)

    # Clear caches for new file set
    file_image_counts.clear()
    file_load_cache.clear()
    
    loaded_files = 0
    for fname in matched_files:
        filepath = os.path.join(current_folder, fname)
        if fname.lower().endswith((".h5", ".hdf5")):
            files_data[fname] = filepath
            loaded_files += 1

    print(f"Loaded {loaded_files} file(s) for processing.")
    update_file_list_display_multi()
    
    # Smart total image calculation: count one file and multiply by file count
    if loaded_files > 0:
        # Get first file to determine images per file
        first_fname = sorted(files_data.keys(), key=natural_sort_key)[0]
        first_filepath = files_data[first_fname]
        images_per_file = get_file_image_count_cached(first_filepath)
        
        # Calculate total images efficiently
        total_images = images_per_file * loaded_files
        
        # Set all files to have the same count (since they should be identical)
        for fname, filepath in files_data.items():
            file_image_counts[filepath] = images_per_file
        
        # Set up slider with correct range (avoid loading any image here)
        image_slider.config(to=max(0, total_images - 1))
        
        print(f"Images per file: {images_per_file}, Total images: {total_images}")
    else:
        image_slider.config(to=0)
        image_slider.set(0)
        image_slider.config(label="No images loaded")
    
def get_global_index_from_listbox(idx):
    """Map the file_list index to the global image index for multi-image mode.
    Correctly handles both collapsed and expanded files.
    """
    if not files_data:
        return 0
    
    sorted_files = sorted(files_data.keys(), key=natural_sort_key)
    global_count = 0    # Tracks cumulative global image index
    listbox_idx = 0     # Tracks current listbox line position
    
    for fname in sorted_files:
        filepath = files_data[fname]
        
        # Get image count for this file
        image_count = file_image_counts.get(filepath, 0)
        if image_count == 0:
            image_count = get_file_image_count_cached(filepath)
            file_image_counts[filepath] = image_count
        
        # Initialize collapsed state if not set
        if fname not in collapsed_files:
            collapsed_files[fname] = True
        
        if collapsed_files[fname]:
            # File is collapsed - only header line shown
            if listbox_idx == idx:
                # Clicked on collapsed file header - return first image of this file
                return global_count
            listbox_idx += 1
        else:
            # File is expanded - header + individual image lines shown
            if listbox_idx == idx:
                # Clicked on expanded file header - return first image of this file
                return global_count
            listbox_idx += 1  # Skip header line
            
            # Check individual image lines
            for i in range(image_count):
                if listbox_idx == idx:
                    # Clicked on specific image line - return that image's global index
                    return global_count + i
                listbox_idx += 1
        
        global_count += image_count
    
    return 0  # Fallback

def get_image_by_global_index_multi(idx):
    """
    Get image by global index using efficient single-image loading.
    Uses pre-calculated image counts from load time.
    """
    if not files_data:
        return None, None, None, None
    
    sorted_files = sorted(files_data.keys(), key=natural_sort_key)
    count = 0
    
    for h5_idx, fname in enumerate(sorted_files, start=1):
        filepath = files_data[fname]
        
        # Use cached image count (should be available from load time)
        # If not cached, count it now (this should not happen after proper loading)
        image_count = file_image_counts.get(filepath, 0)
        if image_count == 0:
            image_count = get_file_image_count_cached(filepath)
            file_image_counts[filepath] = image_count
        
        # Skip files with no images
        if image_count == 0:
            continue
            
        if idx < count + image_count:
            local_idx = idx - count
            # Try to use cached single image loading for better memory efficiency
            cache_key = (filepath, local_idx)
            if cache_key in file_load_cache:
                img = file_load_cache[cache_key]
            else:
                # Use single image loading instead of loading entire stack
                img = load_single_image_from_h5_stack(filepath, local_idx)
                if img is not None:
                    # Cache up to 20 recently loaded images to balance memory and performance
                    if len(file_load_cache) > 20:
                        # Remove oldest cache entries (remove 5 at a time for efficiency)
                        keys_to_remove = list(file_load_cache.keys())[:5]
                        for key in keys_to_remove:
                            del file_load_cache[key]
                    file_load_cache[cache_key] = img
            
            if img is not None:
                return fname, h5_idx, local_idx, img
            else:
                print(f"Failed to load image {local_idx} from {fname}")
                return None, None, None, None
        count += image_count
    
    # If we get here during movie playback, wrap around to the beginning
    if is_playing and sorted_files:
        return get_image_by_global_index_multi(0)
    
    return None, None, None, None

def update_display_from_index_multi(idx):
    global current_index
    fname, h5_idx, local_idx, img = get_image_by_global_index_multi(idx)
    if img is not None:
        display_image(img, f"{fname}[{local_idx}]")
        current_index = idx
        image_slider.set(idx)
        image_slider.config(label=f"File: {fname}[{local_idx}]")
        
        # Clear all caches after displaying to prevent RAM buildup during manual sliding
        file_load_cache.clear()
        file_info_cache.clear()
        multi_mode_cache.clear()
        del img
        import gc
        gc.collect()
    else:
        print("Image index out of range.")

def manual_select_multi(event=None):
    global is_playing, current_index, slider_programmatic
    pause_movie()  # Use proper pause function
    selection = file_list.curselection()
    if not selection:
        return
    
    idx = selection[0]
    global_idx = get_global_index_from_listbox(idx)
    # Removed gc.collect() for better performance
    update_display_from_index_multi(global_idx)
    current_index = global_idx
    slider_programmatic = True
    image_slider.set(global_idx)
    slider_programmatic = False

def on_image_slider_multi(val):
    
    # Ensure slider range is correct to prevent out of range errors
    ensure_slider_range_correct()
    
    idx = int(float(val))
    
    # Clear caches before loading new image to prevent accumulation
    file_load_cache.clear()
    file_info_cache.clear()
    multi_mode_cache.clear()
    gc.collect()  # Release previously loaded image
    
    update_display_from_index_multi(idx)
    listbox_idx = get_listbox_index_from_global_index(idx)
    
    # Ensure proper file list highlighting and scrolling
    if 0 <= listbox_idx < file_list.size():
        file_list.selection_clear(0, END)
        file_list.selection_set(listbox_idx)
        file_list.activate(listbox_idx)
        # Force scroll to make current selection visible
        file_list.see(listbox_idx)
        # Also try scrolling a bit before and after for better visibility
        if listbox_idx > 0:
            file_list.see(listbox_idx - 1)
        file_list.see(listbox_idx)
        if listbox_idx < file_list.size() - 1:
            file_list.see(listbox_idx + 1)

def get_listbox_index_from_global_index(global_idx):
    """Map a global image index to the corresponding listbox index.
    Correctly handles both collapsed and expanded files.
    """
    if not files_data:
        return 0
    
    sorted_files = sorted(files_data.keys(), key=natural_sort_key)
    global_count = 0  # Tracks global image index
    listbox_idx = 0   # Tracks actual listbox line position
    
    for fname in sorted_files:
        filepath = files_data[fname]
        
        # Get image count for this file
        image_count = file_image_counts.get(filepath, 0)
        if image_count == 0:
            image_count = get_file_image_count_cached(filepath)
            file_image_counts[filepath] = image_count
        
        # Check if target global_idx is in this file's range
        if global_idx < global_count + image_count:
            # The target image is in this file
            if fname not in collapsed_files:
                collapsed_files[fname] = True
            
            if collapsed_files[fname]:
                # File is collapsed - return the header line index
                return listbox_idx
            else:
                # File is expanded - calculate which image line
                local_img_idx = global_idx - global_count
                return listbox_idx + 1 + local_img_idx  # +1 for header line
        
        # Move to next file
        global_count += image_count
        
        # Update listbox position based on file display state
        if fname not in collapsed_files:
            collapsed_files[fname] = True
        
        if collapsed_files[fname]:
            listbox_idx += 1  # Just header line
        else:
            listbox_idx += 1 + image_count  # Header + all image lines
    
    return max(0, listbox_idx - 1)  # Fallback

def play_movie_multi():
    global is_playing, current_index
    is_playing = True
    print("Movie playback started.")
    
    # Start from the slider's current position
    idx = int(float(image_slider.get()))
    current_index = idx
    loop_through_images_multi(current_index)

def loop_through_images_multi(idx):
    global is_playing, current_index, slider_programmatic
    if not is_playing:
        return
        
    if not files_data:
        return
    
    # Calculate total images for proper bounds checking
    if file_image_counts:
        images_per_file = next(iter(file_image_counts.values()))
        total_images = images_per_file * len(files_data)
        
        # Handle wraparound: if idx exceeds total, loop back to 0
        if idx >= total_images:
            idx = 0
            print(f"Movie completed full cycle of {total_images} images, looping back to start...")
    
    current_index = idx
    fname, h5_idx, local_idx, data = get_image_by_global_index_multi(idx)
    if data is not None:
        # Cache vmin/vmax values for better performance during movie playback
        cached_vmin = vmin_slider.get()
        cached_vmax = vmax_slider.get()
        display_image(data, f"{fname}[{local_idx}]", vmin=cached_vmin, vmax=cached_vmax)
        slider_programmatic = True
        # Always update the slider and label during movie playback
        image_slider.set(idx)
        image_slider.config(label=f"File: {fname}[{local_idx}]")
        
        # Update file list selection to show current position
        listbox_idx = get_listbox_index_from_global_index(idx)
        file_list.selection_clear(0, END)
        file_list.selection_set(listbox_idx)
        file_list.activate(listbox_idx)
        file_list.see(listbox_idx)
        
        # Release memory after displaying and clear all caches to prevent RAM buildup
        del data
        
        # Aggressively clear all caches during movie playback to prevent RAM accumulation
        file_load_cache.clear()
        file_info_cache.clear()
        multi_mode_cache.clear()
        
        import gc
        gc.collect()
        
        # Enhanced GUI responsiveness like hdf27integration_log.py
        process_gui_events()  # Process all pending GUI events to keep windows movable/resizable
        import time
        time.sleep(0.01)  # Slightly longer yield for better responsiveness (10ms like hdf27integration_log.py)
        
        # Continue to next image (wraparound handled above)
        root.after(playback_speed_ms, lambda: loop_through_images_multi(idx + 1))
    else:
        print(f"Failed to load image at index {idx}, trying to loop back to start...")
        # If image loading fails, try wrapping back to beginning
        root.after(playback_speed_ms, lambda: loop_through_images_multi(0))

def save_all_files_in_directory_multi():
    """Save all images from all HDF5 files in the directory as TIFF files,
    and save metadata from corresponding master files as TXT files (one per master)."""
    if not current_folder:
        print("No directory selected.")
        return

    out_dir = filedialog.askdirectory(title="Select Output Directory for TIFFs and Metadata")
    if not out_dir:
        print("Directory selection cancelled.")
        return

    total_images_saved = 0
    total_files_processed = 0
    total_metadata_saved = 0
    processed_master_files = set()  # Track which master files have been processed

    for fname in os.listdir(current_folder):
        if fname.lower().endswith((".h5", ".hdf5")) and "master" not in fname.lower():
            filepath = os.path.join(current_folder, fname)
            with h5py.File(filepath, 'r') as hdf:
                # Try common paths for image data
                possible_paths = [
                    '/entry/data/data', 'entry/data/data', '/data', 'data'
                ]
                data = None
                for path in possible_paths:
                    if path in hdf:
                        data = hdf[path]
                        break
                if data is not None:
                    arr = data[()]
                    base, _ = os.path.splitext(fname)
                    if arr.ndim == 3:
                        for i in range(arr.shape[0]):
                            out_name = f"{base}_{i}.tiff"
                            out_path = os.path.join(out_dir, out_name)
                            tifffile.imwrite(out_path, arr[i].astype(np.float32))
                            print(f"Saved {out_path}")
                            total_images_saved += 1
                    elif arr.ndim == 2:
                        out_name = f"{base}.tiff"
                        out_path = os.path.join(out_dir, out_name)
                        tifffile.imwrite(out_path, arr.astype(np.float32))
                        print(f"Saved {out_path}")
                        total_images_saved += 1
                    total_files_processed += 1
                    
                    # Save metadata from master file (only once per master file)
                    master_path = get_master_file_path(fname, current_folder)
                    if master_path and master_path not in processed_master_files:
                        metadata = extract_metadata_from_master(master_path)
                        if metadata:
                            # Use master file base name for metadata file
                            master_base = os.path.splitext(os.path.basename(master_path))[0]
                            metadata_out_name = f"{master_base}.metafile"
                            metadata_out_path = os.path.join(out_dir, metadata_out_name)
                            if save_metadata_to_txt(metadata, metadata_out_path):
                                print(f"Saved: {metadata_out_path}")
                                total_metadata_saved += 1
                                processed_master_files.add(master_path)

    print(f"Saved {total_images_saved} image(s) from {total_files_processed} h5 file(s) to {out_dir}")
    print(f"Saved {total_metadata_saved} metadata file(s)")


def save_filtered_files_as_tiff_multi():
    """Save all filtered images as individual TIFF files in a selected output directory,
    and save metadata from corresponding master files as TXT files (one per master)."""
    out_dir = filedialog.askdirectory(title="Select Output Directory for Filtered TIFFs and Metadata")
    if not out_dir:
        print("Directory selection cancelled.")
        return
    
    count = 0
    metadata_count = 0
    processed_master_files = set()  # Track which master files have been processed
    
    # Save all images from all files in files_data (not just visible in file_list)
    for base, filepath in files_data.items():
        images = read_image_file_multi(filepath)
        for img_idx, img in enumerate(images):
            out_name = f"{os.path.splitext(base)[0]}_{img_idx}.tiff"
            out_path = os.path.join(out_dir, out_name)
            tifffile.imwrite(out_path, img.astype(np.float32))
            print(f"Saved {out_path}")
            count += 1
        
        # Save metadata from master file (only once per master file)
        master_path = get_master_file_path(base, current_folder)
        if master_path and master_path not in processed_master_files:
            metadata = extract_metadata_from_master(master_path)
            if metadata:
                # Use master file base name for metadata file
                master_base = os.path.splitext(os.path.basename(master_path))[0]
                metadata_out_name = f"{master_base}.metafile"
                metadata_out_path = os.path.join(out_dir, metadata_out_name)
                if save_metadata_to_txt(metadata, metadata_out_path):
                    print(f"Saved: {metadata_out_path}")
                    metadata_count += 1
                    processed_master_files.add(master_path)
        
        # Release memory after processing each file
        del images
        import gc
        gc.collect()
    
    print(f"All {count} loaded images saved as TIFF in {out_dir}")
    print(f"Saved {metadata_count} metadata file(s)")


# --- v1p10 logic (single-image HDF5 and TIFF) ---

def read_image_file_single(filepath):
    """Read either HDF5 or TIFF/TIF file and return the image data."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".h5", ".hdf5"]:
        try:
            with h5py.File(filepath, 'r') as hdf:
                # Try common paths for image data
                possible_paths = [
                    '/entry/data/data', 'entry/data/data', '/data', 'data'
                ]
                for path in possible_paths:
                    if path in hdf:
                        data = hdf[path]
                        # Validate that it's actual image data (2D or 3D array)
                        if data.ndim >= 2:
                            data_array = np.squeeze(np.array(data))
                            # For single image mode, return only the first frame if 3D
                            if data_array.ndim == 3:
                                print(f"3D data detected with shape {data_array.shape}, taking first frame for single mode")
                                return data_array[0]  # Take first frame
                            elif data_array.ndim == 2:
                                return data_array
                            else:
                                print(f"Unexpected data dimensions: {data_array.ndim}")
                                return None
                # No valid image data found
                print(f"No valid image data found in {filepath}")
                return None
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")
    elif ext in [".tif", ".tiff"]:
        try:
            data = tifffile.imread(filepath)
            return data
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")
    return None

def refresh_file_list_single():
    global current_folder
    if not current_folder:
        print("No folder loaded yet.")
        return

    keyword_input = file_filter_var.get().strip()

    if not keyword_input:
        matched_files = [
            fname for fname in os.listdir(current_folder)
            if (fname.lower().endswith((".h5", ".tif", ".tiff")) and
                ("master" not in fname.lower() if fname.lower().endswith((".h5", ".hdf5")) else True) and  # Exclude master from h5 files
                ("_data_" in fname.lower() if fname.lower().endswith((".h5", ".hdf5")) else True))  # Only include data files from h5
        ]
        # Sort files naturally
        matched_files = sorted(matched_files, key=natural_sort_key)
    else:
        keywords = [kw.strip() for kw in keyword_input.split(';') if kw.strip()]
        matched_files = [
            fname for fname in os.listdir(current_folder)
            if (fname.lower().endswith((".h5", ".tif", ".tiff")) and 
                any(kw in fname for kw in keywords) and
                ("master" not in fname.lower() if fname.lower().endswith((".h5", ".hdf5")) else True) and  # Exclude master from h5 files
                ("_data_" in fname.lower() if fname.lower().endswith((".h5", ".hdf5")) else True))  # Only include data files from h5
        ]
        # Sort files naturally
        matched_files = sorted(matched_files, key=natural_sort_key)
    
    # Sort files naturally
    matched_files = sorted(matched_files, key=natural_sort_key)

    new_files = 0
    for fname in matched_files:
        if fname not in files_data:
            filepath = os.path.join(current_folder, fname)
            # Skip expensive validation for faster refresh
            files_data[fname] = filepath
            file_list.insert(END, fname)
            new_files += 1

    print(f"Refresh complete. {new_files} new file(s) added.")
    image_slider.config(to=max(0, len(files_data)-1))

def load_all_files_single():
    """Load all HDF5 and TIFF files in the current directory without filtering."""
    global current_folder
    if not current_folder or not os.path.isdir(current_folder):
        print("Invalid directory. Please select a valid directory.")
        try:
            selected_dir_var.set("Invalid directory. Please select a valid directory.")
        except Exception:
            pass
        return

    files_data.clear()
    multi_mode_cache.clear()  # Clear cache for fresh calculations
    file_image_counts.clear()  # Clear image count cache
    collapsed_files.clear()   # Clear collapsed state
    file_list.delete(0, END)

    # Load all supported files
    matched_files = [
        fname for fname in os.listdir(current_folder)
        if (fname.lower().endswith((".h5", ".hdf5", ".tif", ".tiff")) and
            ("master" not in fname.lower() if fname.lower().endswith((".h5", ".hdf5")) else True) and  # Exclude master from h5 files
            ("_data_" in fname.lower() if fname.lower().endswith((".h5", ".hdf5")) else True))  # Only include data files from h5
    ]
    
    # Sort files naturally to ensure proper numerical order
    matched_files = sorted(matched_files, key=natural_sort_key)

    if not matched_files:
        print(f"No supported files found in {current_folder}.")
        return

    for fname in matched_files:
        filepath = os.path.join(current_folder, fname)
        # Skip the expensive has_image_data check for faster loading
        # We'll validate files when actually accessing them
        files_data[fname] = filepath
        file_list.insert(END, fname)  # Insert clean filename without prefixes

    # Set slider range and initialize position
    image_slider.config(to=max(0, len(files_data)-1))
    
    # Initialize current_index and set initial position (avoid auto-loading image)
    global current_index
    current_index = 0
    if len(files_data) > 0:
        image_slider.set(0)
        first_file = file_list.get(0)
        image_slider.config(label=f"File: {first_file}")
    
    print(f"Loaded {len(files_data)} file(s) for processing.")

def load_filtered_files_single():
    global current_folder
    if not current_folder or not os.path.isdir(current_folder):
        print("Invalid directory. Please select a valid directory.")
        try:
            selected_dir_var.set("Invalid directory. Please select a valid directory.")
        except Exception:
            pass
        return

    keyword_input = file_filter_var.get().strip()
    if not keyword_input:
        print("Please enter one or more keywords separated by semicolons.")
        return

    keywords = [kw.strip() for kw in keyword_input.split(';') if kw.strip()]
    if not keywords:
        print("No valid keywords provided.")
        return

    files_data.clear()
    multi_mode_cache.clear()  # Clear cache for fresh calculations
    file_image_counts.clear()  # Clear image count cache
    collapsed_files.clear()   # Clear collapsed state
    file_list.delete(0, END)

    matched_files = [
        fname for fname in os.listdir(current_folder)
        if (fname.lower().endswith((".h5", ".tif", ".tiff")) and 
            any(kw in fname for kw in keywords) and
            ("master" not in fname.lower() if fname.lower().endswith((".h5", ".hdf5")) else True))  # Only exclude master from h5 files
    ]
    
    # Sort files naturally to ensure proper numerical order
    matched_files = sorted(matched_files, key=natural_sort_key)

    if not matched_files:
        print(f"No files matching keywords {keywords} found in {current_folder}.")
        return

    for fname in matched_files:
        filepath = os.path.join(current_folder, fname)
        # Skip the expensive has_image_data check for faster loading
        # We'll validate files when actually accessing them
        files_data[fname] = filepath
        file_list.insert(END, fname)  # Insert clean filename without prefixes

    # Set slider range and initialize position
    image_slider.config(to=max(0, len(files_data)-1))
    
    # Initialize current_index and set initial position (avoid auto-loading image)
    global current_index
    current_index = 0
    if len(files_data) > 0:
        image_slider.set(0)
        first_file = file_list.get(0)
        image_slider.config(label=f"File: {first_file}")
    
    print(f"Loaded {len(files_data)} file(s) for processing.")

def update_display_from_index_single(idx):
    if idx < 0 or idx >= file_list.size():
        print(f"Index {idx} out of range")
        return
    sel = file_list.get(idx)
    
    # Use helper function to clean filename
    clean_filename = get_clean_filename(sel)
    
    if clean_filename not in files_data:
        print(f"File {clean_filename} not found in files_data")
        return
    filepath = files_data[clean_filename]  # This is now a file path
    data = read_image_file_single(filepath)
    if data is not None:
        masked = apply_mask_with_shape_check(data)
        vmax_data = np.nanmax(masked)
        vmin_data = np.nanmin(masked)
        
        # Handle NaN or inf values
        if np.isnan(vmax_data) or np.isinf(vmax_data):
            vmax_data = 1
        if np.isnan(vmin_data) or np.isinf(vmin_data):
            vmin_data = 0
        
        # Ensure vmax > vmin (handle case with no pixel variation)
        if vmax_data <= vmin_data:
            vmax_data = vmin_data + 1

        # Update sliders
        vmin_slider.config(from_=0, to=max(1, vmax_data))
        vmax_slider.config(from_=0, to=max(1, vmax_data))
         
        vmin_slider.set(vmin_data if vmin_data else 0)
        vmax_slider.set(vmax_data if vmax_data else 1)

        display_image(data, clean_filename, vmin=vmin_slider.get(), vmax=vmax_slider.get())

        image_slider.set(idx)
        image_slider.config(label=f"File: {sel}")
        
        # Clear all caches after displaying to prevent RAM buildup during manual sliding
        file_load_cache.clear()
        file_info_cache.clear()
        multi_mode_cache.clear()
        
        # Release memory after displaying
        del data, masked
        import gc
        gc.collect()
    else:
        print(f"Failed to load image data from {filepath}")

def manual_select_single(event=None):
    global is_playing, current_index
    pause_movie()  # Use proper pause function

    selection = file_list.curselection()
    if not selection:
        return
    idx = selection[0]
    current_index = idx  # Update current_index when user selects
    # Removed gc.collect() for better performance
    update_display_from_index_single(idx)

def on_image_slider_single(val):
    global current_index, slider_programmatic
    if not slider_programmatic:
        pause_movie()
    idx = int(float(val))
    if 0 <= idx < file_list.size():  # Use file_list.size() instead of len(files_data)
        current_index = idx
        fname = file_list.get(idx)
        
        # Clean filename if it has multiimage prefixes
        clean_fname = fname
        if fname.startswith('[+] ') or fname.startswith('[-] '):
            clean_fname = fname[4:]
        elif fname.startswith('    '):  # Expanded image entries
            clean_fname = fname.strip()
            if '[' in clean_fname and ']' in clean_fname:
                clean_fname = clean_fname.split(' [')[0]
        
        image_slider.config(label=f"File: {clean_fname}")
        file_list.selection_clear(0, END)
        file_list.selection_set(idx)
        file_list.activate(idx)
        file_list.see(idx)
        
        # Clear caches before loading new image to prevent accumulation
        file_load_cache.clear()
        file_info_cache.clear()
        multi_mode_cache.clear()
        gc.collect()  # Release previously loaded image
        
        update_display_from_index_single(idx)
    if slider_programmatic:
        slider_programmatic = False

def play_movie_single():
    global is_playing, current_index
    if not files_data:
        print("No images loaded to play.")
        return
    if is_playing:
        print("Already playing.")
        return
    is_playing = True
    print("Movie playback started.")
    loop_through_images_single(current_index)

def loop_through_images_single(idx):
    global is_playing, current_index, slider_programmatic, multi_mode_cache
    if not is_playing or not files_data:
        return
    
    # Cache file count to avoid repeated calls to file_list.size()
    if 'single_file_count' not in multi_mode_cache:
        multi_mode_cache['single_file_count'] = file_list.size()
    
    num_files = multi_mode_cache['single_file_count']
    if num_files == 0:
        return
    idx = idx % num_files  # Loop back to start
    current_index = idx
    fname = file_list.get(idx)
    
    # Clean filename if it has multiimage prefixes
    clean_fname = fname
    if fname.startswith('[+] ') or fname.startswith('[-] '):
        clean_fname = fname[4:]
    elif fname.startswith('    '):  # Expanded image entries
        clean_fname = fname.strip()
        if '[' in clean_fname and ']' in clean_fname:
            clean_fname = clean_fname.split(' [')[0]
    
    if clean_fname not in files_data:
        print(f"File {clean_fname} not found in files_data")
        root.after(playback_speed_ms, lambda: loop_through_images_single(idx + 1))
        return
    
    filepath = files_data[clean_fname]  # This is a file path
    
    # Load fresh data each time to prevent memory accumulation
    data = read_image_file_single(filepath)
    
    if data is not None:
        # Cache vmin/vmax values for better performance during movie playback
        cached_vmin = vmin_slider.get()
        cached_vmax = vmax_slider.get()
        display_image(data, fname, vmin=cached_vmin, vmax=cached_vmax)
        slider_programmatic = True
        image_slider.set(idx)
        image_slider.config(label=f"File: {fname}")
        
        # Update file list selection to show current position
        file_list.selection_clear(0, END)
        file_list.selection_set(idx)
        file_list.selection_set(idx)
        file_list.activate(idx)
        file_list.see(idx)
        
        # Release memory after displaying and clear all caches to prevent RAM buildup
        del data
        
        # Aggressively clear all caches during movie playback to prevent RAM accumulation
        file_load_cache.clear()
        file_info_cache.clear()
        multi_mode_cache.clear()
        
        import gc
        gc.collect()
        
        # Enhanced GUI responsiveness like hdf27integration_log.py
        process_gui_events()  # Process all pending GUI events to keep windows movable/resizable
        import time
        time.sleep(0.01)  # Slightly longer yield for better responsiveness (10ms like hdf27integration_log.py)
        
        root.after(playback_speed_ms, lambda: loop_through_images_single(idx + 1))
    else:
        print(f"Failed to load image data from {filepath}")
        # Skip to next image if current one fails to load
        root.after(playback_speed_ms, lambda: loop_through_images_single(idx + 1))

def save_image(data, out_dir, fname):
    # Apply mask to the image with shape validation
    masked = apply_mask_with_shape_check(data)

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Generate output file path
    fname_out = os.path.splitext(fname)[0] + '.tiff'
    path_out = os.path.join(out_dir, fname_out)

    # Replace NaNs with zero, but do not clip
    tiff_data = np.nan_to_num(masked, nan=0.0, posinf=0.0, neginf=0.0)
    tiff_data = tiff_data.astype(np.uint32)

    # Write to TIFF
    try:
        tifffile.imwrite(path_out, tiff_data)
        print(f"Saved: {path_out}")
    except Exception as e:
        print(f"Failed to save {path_out}: {e}")

def save_all_files_in_directory_single():
    """Save all HDF5 files in the directory as TIFF files,
    and save metadata from corresponding master files as TXT files (one per master)."""
    if not current_folder:
        print("No directory selected.")
        return

    output_folder = filedialog.askdirectory(title="Select Output Directory to Save TIFF Images and Metadata")
    if not output_folder:
        print("Directory selection cancelled.")
        return
    
    all_files = [
        fname for fname in os.listdir(current_folder)
        if fname.endswith(".h5") and "master" not in fname.lower()
    ]

    count = 0
    metadata_count = 0
    processed_master_files = set()  # Track which master files have been processed
    
    for fname in all_files:
        filepath = os.path.join(current_folder, fname)
        try:
            with h5py.File(filepath, 'r') as hdf:
                if '/entry/data/data' in hdf:
                    data = np.squeeze(np.array(hdf['/entry/data/data']))
                    save_image(data, output_folder, fname)
                    count += 1
                    
                    # Save metadata from master file (only once per master file)
                    master_path = get_master_file_path(fname, current_folder)
                    if master_path and master_path not in processed_master_files:
                        metadata = extract_metadata_from_master(master_path)
                        if metadata:
                            # Use master file base name for metadata file
                            master_base = os.path.splitext(os.path.basename(master_path))[0]
                            metadata_out_name = f"{master_base}.metafile"
                            metadata_out_path = os.path.join(output_folder, metadata_out_name)
                            if save_metadata_to_txt(metadata, metadata_out_path):
                                print(f"Saved: {metadata_out_path}")
                                metadata_count += 1
                                processed_master_files.add(master_path)
        except Exception as e:
            print(f"Failed to load {fname}: {e}")

    print(f"All {count} file(s) saved as TIFF in {output_folder}.")
    print(f"Saved {metadata_count} metadata file(s)")


def save_filtered_files_as_tiff_single():
    """Save all filtered files as TIFF images,
    and save metadata from corresponding master files as TXT files (one per master)."""
    if not files_data:
        print("No files loaded.")
        return
    output_folder = filedialog.askdirectory(title="Select Output Directory to Save TIFF Images and Metadata")
    if not output_folder:
        print("Directory selection cancelled.")
        return

    count = 0
    metadata_count = 0
    processed_master_files = set()  # Track which master files have been processed
    
    for idx in range(file_list.size()):
        fname = file_list.get(idx)
        filepath = files_data[fname]  # This is a file path
        data = read_image_file_single(filepath)
        if data is not None:
            save_image(data, output_folder, fname)
            count += 1
            
            # Save metadata from master file (only once per master file)
            master_path = get_master_file_path(fname, current_folder)
            if master_path and master_path not in processed_master_files:
                metadata = extract_metadata_from_master(master_path)
                if metadata:
                    # Use master file base name for metadata file
                    master_base = os.path.splitext(os.path.basename(master_path))[0]
                    metadata_out_name = f"{master_base}.metafile"
                    metadata_out_path = os.path.join(output_folder, metadata_out_name)
                    if save_metadata_to_txt(metadata, metadata_out_path):
                        print(f"Saved: {metadata_out_path}")
                        metadata_count += 1
                        processed_master_files.add(master_path)
            
            # Release memory after processing each file
            del data
            import gc
            gc.collect()
        else:
            print(f"Failed to load {fname}")
    print(f"All {count} loaded file(s) saved in {output_folder}.")
    print(f"Saved {metadata_count} metadata file(s)")


# --- Shared display_image function (uses scale_mode) ---

def display_image(data, title, vmin=None, vmax=None):
    global current_image_data, current_colorbar, ax, manual_vmax_enabled, plot_window, current_filename
    if 'plot_window' not in globals() or not plot_window.winfo_exists():
        show_plot_window()
    current_image_data = apply_mask(data)
    current_filename = title  # Update the current filename

    # Remove old colorbar more thoroughly to prevent overlapping
    if current_colorbar:
        try:
            current_colorbar.remove()
        except Exception as e:
            print(f"Failed to remove colorbar: {e}")
        current_colorbar = None
    
    # Clear the axis and any remaining artists
    ax.clear()
    # Clear any remaining colorbars from the figure
    for ax_obj in fig.get_axes():
        if ax_obj != ax:  # Don't clear the main axis we just cleared
            try:
                ax_obj.remove()
            except Exception:
                pass

    if vmin is None:
        vmin = 0

    # Always use manual vmax if enabled, regardless of data
    if manual_vmax_enabled:
        try:
            manual_vmax = int(max_pixel_var.get())
            vmax = manual_vmax
        except (ValueError, TypeError):
            print("Manual vmax is invalid, reverting to auto-scaling.")
            vmax = np.nanmax(current_image_data)
            manual_vmax_enabled = False
    elif vmax is None:
        vmax = np.nanmax(current_image_data)

    # Handle case when all pixels have the same value (avoid vmin == vmax)
    if vmax <= vmin or np.isnan(vmax) or np.isinf(vmax):
        vmax = vmin + 1
        print(f"Warning: No pixel intensity variation detected. Setting vmax to {vmax}")

    # Update sliders: ensure the slider's range includes the manual vmax
    data_vmax = np.nanmax(current_image_data)
    if np.isnan(data_vmax) or np.isinf(data_vmax):
        data_vmax = 1
    slider_max = max(int(vmax), int(data_vmax), 1)
    vmin_slider.config(from_=0, to=slider_max)
    vmax_slider.config(from_=0, to=slider_max)

    vmin_slider.set(vmin)
    vmax_slider.set(int(vmax))

    # Plot the image
    if scale_mode.get() == "log":
        # For log scale, ensure vmin > 0 and vmax > vmin
        log_vmin = max(vmin, 0.1)
        log_vmax = max(log_vmin * 10, vmax)
        im = ax.imshow(current_image_data, cmap='viridis', origin='upper',
                       norm=LogNorm(vmin=log_vmin, vmax=log_vmax))
    else:
        im = ax.imshow(current_image_data, cmap='viridis', origin='upper',
                       vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel("X pixels")
    ax.set_ylabel("Y pixels")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    current_colorbar = fig.colorbar(im, cax=cax)
    current_colorbar.set_label("Intensity (Counts)")
    
    # Ensure proper layout to prevent colorbar overlap
    fig.tight_layout()
    canvas.draw_idle()
    apply_zoom()

# --- Shared GUI and event logic ---

def set_max_pixel():
    global manual_vmax_enabled
    try:
        val = int(max_pixel_var.get())
        vmax_slider.config(to=val)
        vmax_slider.set(val)
        manual_vmax_enabled = True  # Activate manual vmax
        # on_update_plot()
    except ValueError:
        print("Invalid input for max pixel intensity.")

def reset_max_pixel():
    global manual_vmax_enabled
    manual_vmax_enabled = False
    max_pixel_var.set("")  # Clear the entry field

    # Reset the vmax slider to automatic value based on current image data
    if current_image_data is not None:
        vmax = np.nanmax(current_image_data)
        # Handle NaN or inf values
        if np.isnan(vmax) or np.isinf(vmax):
            vmax = 1
        vmax_slider.config(to=max(1, vmax))
        vmax_slider.set(max(1, vmax))
    # on_update_plot()

def pause_movie():
    global is_playing
    if is_playing:
        is_playing = False
        print("Movie playback paused.")

def set_scale_mode(mode):
    scale_mode.set(mode)
    print(f"Scale mode set to {mode}")
    on_update_plot()  # Update current plot immediately

def on_update_plot(event=None):
    if current_image_data is None:
        return

    vmin = vmin_slider.get()
    vmax = vmax_slider.get()
    
    # Handle NaN or inf values
    if np.isnan(vmin) or np.isinf(vmin):
        vmin = 0
    if np.isnan(vmax) or np.isinf(vmax):
        vmax = 1

    # For log scale, ensure vmin > 0 and vmax > vmin
    if scale_mode.get() == "log":
        vmin = max(vmin, 0.1)
        if vmax <= vmin:
            vmax = vmin * 10

    # For linear scale, ensure vmax > vmin
    if vmax <= vmin:
        vmax = vmin + 1

    # Update color limits without redrawing the whole image
    if ax.images:
        if scale_mode.get() == "log":
            ax.images[0].set_norm(LogNorm(vmin=vmin, vmax=vmax))
        else:
            ax.images[0].set_norm(Normalize(vmin=vmin, vmax=vmax))
        if current_colorbar:
            current_colorbar.update_normal(ax.images[0])
    canvas.draw_idle()

def enable_zoom_mode():
    global zoom_mode_enabled, zoom_label
    zoom_mode_enabled = True
    print("Zoom mode enabled. Click on the image to zoom in.")
    # Change cursor to crosshair
    canvas.get_tk_widget().config(cursor="crosshair")
    # Show label near the plot window
    if 'zoom_label' not in globals() or zoom_label is None:
        zoom_label = Label(plot_window, text="Now Click the Region to Zoom In", bg="yellow", font=("Arial", 12, "bold"))
    zoom_label.place(relx=0.5, rely=0.01, anchor="n")
    # Connect the matplotlib click event
    canvas.mpl_connect('button_press_event', on_image_click_zoom)

def on_image_click_zoom(event):
    global zoom_mode_enabled, zoom_center, zoom_label
    if not zoom_mode_enabled:
        return
    if ax is None or current_image_data is None:
        return
    if event.inaxes != ax:
        return
    x = int(event.xdata)
    y = int(event.ydata)
    zoom_center = (x, y)
    apply_zoom()
    zoom_mode_enabled = False
    print(f"Zoom center set to ({x}, {y}), size {zoom_size_x} x {zoom_size_y}.")
    # Hide the zoom label after clicking
    if 'zoom_label' in globals() and zoom_label is not None:
        zoom_label.place_forget()
        zoom_label = None

def apply_zoom():
    global zoom_center, zoom_size, current_image_data
    if zoom_center is None or current_image_data is None:
        ax.set_xlim(auto=True)
        ax.set_ylim(auto=True)
        canvas.draw_idle()
        return

    x, y = zoom_center
    size_x = zoom_size_x
    size_y = zoom_size_y
    half_x = size_x // 2
    half_y = size_y // 2

    # Calculate the bounds of the zoomed region
    x_start = max(0, x - half_x)
    x_end = min(current_image_data.shape[1], x + half_x)
    y_start = max(0, y - half_y)
    y_end = min(current_image_data.shape[0], y + half_y)

    ax.set_xlim(x_start, x_end)
    ax.set_ylim(y_end, y_start)  # y is inverted in imshow
    canvas.draw_idle()

def reset_zoom():
    global zoom_center
    zoom_center = None
    if ax is not None and current_image_data is not None:
        ax.set_xlim(0, current_image_data.shape[1])
        ax.set_ylim(current_image_data.shape[0], 0)
        canvas.draw_idle()
    else:
        print("No image data to reset zoom on.")

def update_zoom_size_from_entry(*args):
    global zoom_size_x, zoom_size_y
    try:
        val_x = int(zoom_size_x_var.get())
        if val_x > 0:
            zoom_size_x = val_x
    except Exception:
        pass
    try:
        val_y = int(zoom_size_y_var.get())
        if val_y > 0:
            zoom_size_y = val_y
    except Exception:
        pass

def show_image_plot_window():
    global windows_manually_closed
    
    # Reset all manual close flags when user explicitly shows all windows
    windows_manually_closed['image_plot'] = False
    
    # Show or create Image Plot window
    if 'plot_window' in globals() and plot_window is not None and plot_window.winfo_exists():
        plot_window.deiconify()
        plot_window.lift()
    else:
        # Create the Image Plot window if it doesn't exist
        show_plot_window()
        # Refresh the image display if we have current data
        if current_image_data is not None:
            # Use the current filename if available, otherwise use a generic title
            title = current_filename if current_filename else "Current Image"
            display_image(current_image_data, title)

def show_plot_window():
    global plot_window, fig, ax, canvas, toolbar, windows_manually_closed

    if 'plot_window' in globals() and plot_window.winfo_exists():
        plot_window.deiconify()
        plot_window.lift()
        windows_manually_closed['image_plot'] = False  # Reset manual close flag
        return

    plot_window = Toplevel(root)
    plot_window.title("Image Plot")

    # Add window close event handler
    def on_image_window_close():
        windows_manually_closed['image_plot'] = True
        plot_window.destroy()
    plot_window.protocol("WM_DELETE_WINDOW", on_image_window_close)

    fig = Figure()
    fig.subplots_adjust(left=0.04, right=0.96, top=0.95, bottom=0.07)
    ax = fig.add_subplot(111)

    plot_frame = Frame(plot_window)
    plot_frame.pack(fill=BOTH, expand=True)

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=BOTH, expand=True)

    # Set crosshair cursor - force it on mouse motion to override toolbar behavior
    def on_mouse_motion(event):
        if event.inaxes:
            canvas_widget.config(cursor="crosshair")
        else:
            canvas_widget.config(cursor="")
    
    canvas.mpl_connect('motion_notify_event', on_mouse_motion)

    # Move the toolbar to the bottom
    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()
    toolbar.pack(side='bottom', fill='x')

    if zoom_center is not None:
        apply_zoom()
    root.after(500, place_plot_window_right)

def place_plot_window_right():
    root.update_idletasks()  # Make sure geometry is updated
    root_x = root.winfo_x()
    root_y = root.winfo_y()
    root_width = root.winfo_width()
    plot_window.geometry(f"+{root_x + root_width + 2}+{root_y}")

def open_folder_browser():
    """Open a fast, directory-only browser without scanning file contents."""
    global current_folder

    browser = Toplevel(root)
    browser.title("Select Directory (Click +/▷ to Expand, Double Click to Select)")
    browser.geometry("600x450")

    path_var = StringVar()

    def list_roots():
        roots = []
        # Windows drive roots
        try:
            import string
            for d in string.ascii_uppercase:
                drive = f"{d}:\\"
                if os.path.exists(drive):
                    roots.append(drive)
        except Exception:
            pass
        # Fallback to current drive root
        if not roots:
            try:
                roots.append(os.path.splitdrive(os.getcwd())[0] + "\\")
            except Exception:
                pass
        # Linux root
        if os.path.exists("/"):
            roots.append("/")
        return roots

    def populate_tree(parent_path, parent_node):
        try:
            with os.scandir(parent_path) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        node = tree.insert(parent_node, 'end', text=entry.name, values=(entry.path,))
                        # Add a dummy child so the expand arrow appears (lazy loading)
                        tree.insert(node, 'end', text='')
        except Exception:
            pass

    def on_expand(event):
        item = tree.focus()
        if not item:
            return
        # If only dummy child present, clear and populate
        children = tree.get_children(item)
        if len(children) == 1 and tree.item(children[0], 'text') == '':
            tree.delete(children[0])
            path = tree.set(item, 'fullpath')
            # Use threading to prevent UI freeze on large directories
            def populate_async():
                try:
                    populate_tree(path, item)
                except Exception as e:
                    print(f"Error populating tree: {e}")
            threading.Thread(target=populate_async, daemon=True).start()

    def on_select(event):
        item = tree.focus()
        if not item:
            return
        path = tree.set(item, 'fullpath')
        path_var.set(path)

    def on_open(event=None):
        sel = path_var.get()
        if not sel:
            return
        
        nonlocal_selected = sel
        selected_dir_var.set(nonlocal_selected)

        # Prepare and load new folder without interrupting movie playback
        def do_load():
            nonlocal_folder = nonlocal_selected

            # Stop any active background processing to avoid conflicts
            try:
                global plotting_active, roi_integration_active, plotting_thread
                plotting_active = False
                roi_integration_active = False
                if plotting_thread is not None and plotting_thread.is_alive():
                    plotting_thread.join(timeout=2)
            except Exception:
                pass

            # Clear previous data
            files_data.clear()
            file_list.delete(0, END)
            image_slider.config(to=0)
            # try:
            #     last_roi_intensities.clear()
            #     last_roi_times.clear()
            #     processed_roi_files.clear()
            # except Exception:
            #     pass

            # Set new folder
            global current_folder
            current_folder = nonlocal_folder

            # Load files according to current mode
            if multiimage_mode.get():
                load_all_files_multi()
            else:
                load_all_files_single()

        safe_after(0, do_load)
        browser.destroy()

    # Helper: expand tree to a given full path without loading siblings unnecessarily
    def expand_to_path(path):
        try:
            if not path or not os.path.isdir(path):
                return
            # Find matching root
            for root_item in tree.get_children(''):
                root_path = tree.set(root_item, 'fullpath')
                if path.startswith(root_path):
                    parts = [p for p in os.path.normpath(path).split(os.sep) if p]
                    if root_path.endswith('\\'):
                        parts = parts[1:]
                    elif root_path == '/':
                        parts = parts
                    parent = root_item
                    for part in parts:
                        children = tree.get_children(parent)
                        if len(children) == 1 and tree.item(children[0], 'text') == '':
                            tree.delete(children[0])
                            p = tree.set(parent, 'fullpath')
                            populate_tree(p, parent)
                            children = tree.get_children(parent)
                        target = None
                        for ch in children:
                            if tree.item(ch, 'text') == part:
                                target = ch
                                break
                        if target is None:
                            break
                        tree.item(target, open=True)
                        parent = target
                    # Clean up dummy child and select final node
                    try:
                        final_children = tree.get_children(parent)
                        if len(final_children) == 1 and tree.item(final_children[0], 'text') == '':
                            tree.delete(final_children[0])
                    except Exception:
                        pass
                    tree.selection_set(parent)
                    tree.focus(parent)
                    tree.see(parent)
                    break
        except Exception:
            pass

    # UI layout
    top = Frame(browser)
    top.pack(fill='x')
    path_entry = Entry(top, textvariable=path_var)
    path_entry.pack(side='left', fill='x', expand=True, padx=4, pady=4)

    tree = ttk.Treeview(browser, columns=('fullpath',), displaycolumns=())
    tree.pack(fill='both', expand=True)
    tree.bind('<<TreeviewOpen>>', on_expand)
    tree.bind('<<TreeviewSelect>>', on_select)
    tree.bind('<Double-1>', on_open)

    # Populate roots lazily
    for root_path in list_roots():
        node = tree.insert('', 'end', text=root_path, values=(root_path,))
        tree.insert(node, 'end', text='')

    # Pressing Enter opens the typed directory (same as imager.py);
    def on_enter_open(event=None):
        p = path_var.get()
        if p:
            try:
                expand_to_path(p)
            except Exception:
                pass
        on_open()
    path_entry.bind('<Return>', on_enter_open)
    Button(top, text="Open", command=on_open).pack(side='left', padx=4)

    # Pre-expand to current folder if valid
    try:
        if current_folder and os.path.isdir(current_folder):
            path_var.set(current_folder)
            expand_to_path(current_folder)
            safe_after(0, lambda: expand_to_path(current_folder))
    except Exception:
        pass

    browser.transient(root)
    browser.grab_set()
    browser.focus_set()

# --- Multi-image mode checkbutton ---
def on_multiimage_mode_change():
    """Automatically load files when multiimage mode is toggled - all files if no keywords, filtered files if keywords exist"""
    if current_folder:
        keyword_input = file_filter_var.get().strip()
        if keyword_input:
            # Keywords exist - load filtered files
            if multiimage_mode.get():
                load_filtered_files_multi()
            else:
                load_filtered_files_single()
        else:
            # No keywords - load all files
            if multiimage_mode.get():
                load_all_files_multi()
            else:
                load_all_files_single()
    else:
        # No folder selected - clear everything
        file_list.delete(0, END)
        files_data.clear()
        image_slider.config(to=0)
        image_slider.set(0)
        image_slider.config(label="No files loaded")

def on_window_close():
    """Clean up resources before closing the application"""
    try:
        global is_playing
        is_playing = False
        cleanup_callbacks()
        cleanup_memory_caches()
    except Exception:
        pass  # Ignore errors during cleanup
    finally:
        try:
            root.quit()
            root.destroy()
        except Exception:
            pass

# --- GUI setup ---
root = Tk()
root.title("File Control Panel")
scale_mode = StringVar(value="log")

dir_frame = Frame(root)
dir_frame.pack(pady=5)

dir_button = Button(dir_frame, text="HDF5/TIFF File Directory", command=open_folder_browser)
dir_button.pack(side='left')

selected_dir_var = StringVar(value="No directory selected.")
dir_label = Label(dir_frame, textvariable=selected_dir_var, anchor="e", width=50)
dir_label.pack(side='left', padx=5)
Label(root, text="Load Files With: (e.g., sam2_001; sam2_002)").pack(pady=2)

filter_frame = Frame(root)
filter_frame.pack(pady=2)

file_filter_var = StringVar()
file_filter_entry = Entry(filter_frame, textvariable=file_filter_var, width=60)
file_filter_entry.pack(side='left', padx=2)
file_filter_entry.bind('<Return>', lambda event: [load_filtered_files_multi() if multiimage_mode.get() else load_filtered_files_single()])

Button(filter_frame, text="Load Filtered Files", command=lambda: load_filtered_files_multi() if multiimage_mode.get() else load_filtered_files_single()).pack(side='left', padx=2)

multiimage_mode = BooleanVar(value=False)  # If checked, use multi-image (v1p9_multi) logic
Checkbutton(root, text="A Single H5 File Contains Multiple Images?", variable=multiimage_mode, command=on_multiimage_mode_change).pack(pady=2)

image_slider = Scale(
    root, from_=0, to=0, orient=HORIZONTAL, length=480, label="No files loaded", command=lambda val: on_image_slider_multi(val) if multiimage_mode.get() else on_image_slider_single(val)
)
image_slider.pack(pady=5)
image_slider.set(0)  # Set initial position
# image_slider.bind("<ButtonRelease-1>", lambda event: on_image_slider_multi(image_slider.get()) if multiimage_mode.get() else on_image_slider_single(image_slider.get()))
image_slider.bind("<Button-1>", lambda event: pause_movie())

files_data = {}
output_folder = ""

file_frame = Frame(root)
file_frame.pack(pady=5, padx=10, fill=BOTH, expand=True)

scrollbar = Scrollbar(file_frame)
scrollbar.pack(side=RIGHT, fill=Y)

file_list = Listbox(file_frame, selectmode=SINGLE, yscrollcommand=scrollbar.set, width=60, height=10)
file_list.pack(padx=10, pady=5, fill=BOTH, expand=True)
scrollbar.config(command=file_list.yview)
file_list.bind('<<ListboxSelect>>', lambda event: manual_select_multi(event) if multiimage_mode.get() else manual_select_single(event))
file_list.bind("<Button-1>", lambda event: on_file_list_click(event) if multiimage_mode.get() else manual_select_single(event))

button_frame = Frame(root)
button_frame.pack(pady=5)

refresh_save_frame = Frame(button_frame)
refresh_save_frame.pack(pady=5)
Button(refresh_save_frame, text="Refresh File List", command=lambda: refresh_file_list_multi() if multiimage_mode.get() else refresh_file_list_single()).pack(side='left', padx=20)
Button(refresh_save_frame, text="Save Filtered as Tiff & Metafile", command=lambda: save_filtered_files_as_tiff_multi() if multiimage_mode.get() else save_filtered_files_as_tiff_single()).pack(side='left', padx=2)
Button(refresh_save_frame, text="Save All as Tiff & Metafile", command=lambda: save_all_files_in_directory_multi() if multiimage_mode.get() else save_all_files_in_directory_single()).pack(side='left', padx=2)

movie_frame = Frame(button_frame)
movie_frame.pack(pady=5)
Button(movie_frame, text="Play/Resume", command=lambda: play_movie_multi() if multiimage_mode.get() else play_movie_single()).pack(side='left', padx=2)
Button(movie_frame, text="Pause", command=pause_movie).pack(side='left', padx=(2, 20))
Button(movie_frame, text="Linear Scale", command=lambda: set_scale_mode("linear")).pack(side='left', padx=2)
Button(movie_frame, text="Log Scale", command=lambda: set_scale_mode("log")).pack(side='left', padx=2)

max_pixel_entry_frame = Frame(button_frame)
max_pixel_entry_frame.pack(pady=5)

max_pixel_var = StringVar()
max_pixel_entry = Entry(max_pixel_entry_frame, textvariable=max_pixel_var, width=10)
max_pixel_entry.pack(side='left')
max_pixel_entry.bind('<Return>', lambda event: set_max_pixel())

Button(max_pixel_entry_frame, text="Set Max Pixel", command=set_max_pixel).pack(side='left', padx=2)
Button(max_pixel_entry_frame, text="Reset Max Pixel", command=reset_max_pixel).pack(side='left', padx=2)
Button(max_pixel_entry_frame, text="Zoom In", command=enable_zoom_mode).pack(side='left', padx=(20, 2))
Button(max_pixel_entry_frame, text="Reset Zoom", command=reset_zoom).pack(side='left', padx=2)

zoom_size_frame = Frame(button_frame)
zoom_size_frame.pack(pady=5)

zoom_size_x_var = StringVar(value=str(zoom_size_x))
zoom_size_y_var = StringVar(value=str(zoom_size_y))
zoom_size_x_var.trace_add("write", update_zoom_size_from_entry)
zoom_size_y_var.trace_add("write", update_zoom_size_from_entry)
Label(zoom_size_frame, text="Zoom Size X (pixels):").pack(side='left')
zoom_size_x_entry = Entry(zoom_size_frame, textvariable=zoom_size_x_var, width=6)
zoom_size_x_entry.pack(side='left', padx=2)
zoom_size_x_entry.bind('<Return>', lambda event: [update_zoom_size_from_entry(), apply_zoom()])
Label(zoom_size_frame, text="Y:").pack(side='left')
zoom_size_y_entry = Entry(zoom_size_frame, textvariable=zoom_size_y_var, width=6)
zoom_size_y_entry.pack(side='left', padx=2)
zoom_size_y_entry.bind('<Return>', lambda event: [update_zoom_size_from_entry(), apply_zoom()])
Button(zoom_size_frame, text="Apply Zoom Size", command=lambda: [update_zoom_size_from_entry(), apply_zoom()]).pack(side='left', padx=4)

slider_frame = Frame(button_frame)
slider_frame.pack(pady=5)

Label(slider_frame, text="Minimum Pixel Intensity").pack()
vmin_slider = Scale(
    slider_frame, from_=0, to=1000, orient=HORIZONTAL, resolution=1,
    length=480, command=lambda _: on_update_plot()
)
vmin_slider.pack()
vmin_slider.bind("<Button-1>", lambda event: pause_movie())

Label(slider_frame, text="Maximum Pixel Intensity").pack()
vmax_slider = Scale(
    slider_frame, from_=0, to=1000, orient=HORIZONTAL, resolution=1,
    length=480, command=lambda _: on_update_plot()
)
vmax_slider.pack()
vmax_slider.bind("<Button-1>", lambda event: pause_movie())
plot_window_btn_frame = Frame(slider_frame)
plot_window_btn_frame.pack(pady=5)
Button(plot_window_btn_frame, text="Show Image Window", command=show_image_plot_window).pack(side='left', padx=5)
Button(plot_window_btn_frame, text="Hide Image Window", command=lambda: plot_window.withdraw()).pack(side='left', padx=5)

copyright_label = Label(root, text="Developed by NSLS-II, Brookhaven National Laboratory", font=("Helvetica", 8), anchor="center")
copyright_label.pack(side='bottom', pady=5)

# Set up proper window close handling
root.protocol("WM_DELETE_WINDOW", on_window_close)
root.mainloop()