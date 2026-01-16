# SPECULOOS Photometric Precision Prediction Tool

## Overview

`predict_target_precision.py` predicts the photometric precision achievable for SPECULOOS targets and optimizes detector positioning to maximize precision while accounting for bad pixels.

## What It Does

1. **Queries Gaia DR2** for the target star and potential comparison stars in the field
2. **Applies proper motion corrections** to propagate star positions to the current epoch
3. **Optimizes target placement** on the detector by:
   - Testing positions across the detector at configurable grid spacing using parallel processing
   - Filtering comparison stars by magnitude range and detector boundaries
   - Checking for bad pixel contamination in apertures
   - Computing predicted precision using a trained Decision Tree model
   - Selecting the position that maximizes precision while staying far from bad pixels and edges
4. **Generates visualizations and reports** showing optimal positioning strategy

## Installation

```bash
pip install astropy astroquery numpy pandas matplotlib scikit-learn joblib paramiko scipy
```

## Directory Structure

```
spirit_precision/
├── src/
│   ├── predict_target_precision.py      # Main entry point
│   ├── paths.py                         # Directory path definitions
│   ├── config.py                        # Configuration loading
│   ├── credentials.py                   # Server credentials (add to .gitignore)
│   ├── gaia_queries.py                  # Gaia DR2 catalogue queries
│   ├── coordinate_utils.py              # Coordinate transformations and WCS
│   ├── bad_pixel_handling.py            # Bad pixel map management
│   ├── comparison_star_selection.py     # Comparison star filtering logic
│   ├── precision_prediction.py          # Model inference and magnitude conversions
│   ├── optimization.py                  # Position optimization with parallelization
│   ├── visualization.py                 # Plot generation
│   ├── batch_processing.py              # Batch mode and resume functionality
│   └── utils.py                         # Utility functions
├── configs/
│   ├── config.json
│   └── precision_prediction_model.joblib
├── ref_images/
│   └── spirit.fits                      # Reference FITS for detector WCS
├── BPMs/
│   └── 1_BadPixelMap_YYYYMMDD.fits     # Downloaded bad pixel maps
├── target_lists/
│   └── targets.txt                      # Target lists for batch processing
└── runs/
    ├── {gaia_id}_{YYYYMMDD}/            # Single target runs
    └── batch_{YYYYMMDD}/                # Batch runs
```

## Code Structure

The codebase is modularized into functional components:

- **paths.py**: Centralizes all directory path definitions (configs, BPMs, reference images, output runs)
- **config.py**: Loads JSON configuration and extracts detector parameters from reference FITS
- **credentials.py**: Stores server credentials (excluded from git)
- **gaia_queries.py**: Handles Gaia DR2 queries with proper motion propagation and 2MASS crossmatching
- **coordinate_utils.py**: WCS creation, sky-to-pixel transformations, proper motion propagation
- **bad_pixel_handling.py**: Downloads latest BPM from server via SSH, loads BPM, checks aperture contamination
- **comparison_star_selection.py**: Filters comparison stars by magnitude, position, and bad pixel contamination
- **precision_prediction.py**: Runs trained Decision Tree model, handles magnitude conversions
- **optimization.py**: Parallelized grid search across detector positions
- **visualization.py**: Creates sky survey visualizations and precision maps
- **batch_processing.py**: Manages batch runs, CSV summaries, and resume functionality
- **utils.py**: Helper functions (type conversions, directory creation)

## Usage

### Single Target Mode

```bash
# Basic optimization
python src/predict_target_precision.py --target 1234567890 --optimize

# With all outputs
python src/predict_target_precision.py --target 1234567890 --optimize --save --viz --map

# Centered prediction (non-optimized)
python src/predict_target_precision.py --target 1234567890 --centered
```

### Batch Processing Mode

```bash
# Process multiple targets from a list
python src/predict_target_precision.py --batch targets.txt --optimize --save --viz --map

# Resume an interrupted batch from a specific date
python src/predict_target_precision.py --batch targets.txt --resume 20250120 --optimize --save --viz

# Resume and skip failed targets (don't retry them)
python src/predict_target_precision.py --batch targets.txt --resume 20250120 --skip-failed --optimize
```

**Target list format** (space or comma-separated, Gaia ID in 3rd column):
```
Sp0000-1245  00002867-1245153 2421137424841635840  0.1193751 -12.7542112 ...
Sp0000-0533  00003477-0533070 2443091236074493824  0.1449558  -5.5520516 ...
```

### Command-Line Arguments

**Target Selection (required, mutually exclusive):**
- `--target GAIA_ID` - Process single target by Gaia DR2 source ID
- `--batch FILENAME` - Process multiple targets from file in `target_lists/`

**Operation Modes:**
- `--optimize` - Run position optimization (recommended)
- `--centered` - Run centered prediction without optimization
- `--save` - Save results to JSON file
- `--viz` - Create sky survey visualization with optimal positioning
- `--map` - Create precision map showing precision across detector

**Batch Resume Options:**
- `--resume YYYYMMDD` - Resume batch from specified date (e.g., `--resume 20250120`)
- `--skip-failed` - Skip retrying failed targets when resuming (default: retry failed targets)
- `--force-config` - Override configuration validation errors when resuming (not recommended)

## Configuration File

`configs/config.json` controls detector parameters, bad pixel map source, and analysis settings:

```json
{
  "download_BPM_from_server": true,
  "reference_image": "spirit.fits",
  "detector": {
    "bad_pixel_map_path": "1_BadPixelMap.fits",
    "edge_padding_pixels": 10
  },
  "aperture": {
    "radius_pixels": 15
  },
  "optimization": {
    "grid_spacing_pixels": 3
  },
  "comparison_star_limits": {
    "fainter_limit": 4.2,
    "brighter_limit": -0.6
  },
  "j_to_zyj_conversion": {
    "slope": 0.9159,
    "intercept": 0.122
  },
  "quadratic_fit_coefficients": [0.03799352, -0.72984788, 0.36124136]
}
```

**Note:** `reference_image` and `bad_pixel_map_path` should be filenames only (not full paths). The code automatically looks in `ref_images/` and `BPMs/` directories.

## Credentials Setup (Optional)

For automatic BPM download from the server, create `src/credentials.py`:
```python
# src/credentials.py
SFTP_HOST = "your.server.here"
SFTP_USERNAME = "your_username"
SFTP_BASE_PATH = "/path/to/pipeline/output"
SFTP_TELESCOPE = "telescope_name"
```

Add to `.gitignore`:
```
src/credentials.py
```

If credentials are not provided, the tool will use the most recent BPM from the `BPMs/` directory.


---

## Decision Tree Summary
```
Command-line --download-bpm specified?
├─ Yes → Use command-line value
└─ No → Use config['download_BPM_from_server']

download_BPM_from_server == True?
├─ Yes → Credentials available?
│   ├─ Yes → Download attempt
│   │   ├─ Success → Use downloaded BPM ✓
│   │   └─ Fail → Continue to fallback
│   └─ No → Continue to fallback
└─ No → config['detector']['bad_pixel_map_path'] exists?
    ├─ Yes → Use that path ✓
    └─ No → Continue to fallback

Fallback: Find local BPM
├─ Dated BPMs exist? → Use most recent ✓
├─ Any .fits files exist? → Use first one ✓
└─ No files found → ERROR ✗

```
## Outputs

### Single Target Mode
Creates directory: `runs/{gaia_id}_{YYYYMMDD}/`

**1. Optimization Results JSON** (`optimization_{gaia_id}.json`)
```json
{
  "gaia_id": 1234567890,
  "status": "SUCCESS",
  "timestamp": "2025-01-20T14:30:45.123456",
  "optimal_x": 512.0,
  "optimal_y": 640.0,
  "detector_center_ra": 123.456789,
  "detector_center_dec": -45.678901,
  "target_ra": 123.456000,
  "target_dec": -45.677000,
  "target_offset_x": 14.0,
  "target_offset_y": -20.0,
  "target_offset_ra_arcsec": -4.81,
  "target_offset_dec_arcsec": 6.45,
  "precision": 0.001234,
  "n_comparison_stars": 5,
  "combined_mag": 12.34,
  "target_jmag": 10.5,
  "target_zyj": 10.6,
  "target_teff": 3500.0,
  "distance_to_hazard": 25.3,
  "reference_image": "spirit.fits",
  "bad_pixel_map": "1_BadPixelMap_20251020.fits",
  "processing_time_seconds": 45.2,
  "error_message": ""
}
```

**2. Precision Map FITS** (`precision_map_{gaia_id}_{ref}_{bpm}.fits`)
- 2D FITS image showing predicted precision at each detector position
- Grid spacing matches `grid_spacing_pixels` setting
- NaN values indicate invalid positions (bad pixels, no comparison stars, edge violations)

**3. Precision Map PNG** (`precision_map_{gaia_id}_{ref}_{bpm}.png`)
- Colormap visualization of precision across detector
- Red regions: invalid positions
- Colorbar: valid precision values (lower = better)
- Black dashed box: edge padding boundary
- Black X: optimal position

**4. Optimization Visualization PNG** (`optimization_viz_{gaia_id}_{ref}_{bpm}.png`)
- DSS2 Red sky survey background image
- Red star + circle: target with aperture
- Blue stars + circles: comparison stars with apertures and Gaia IDs
- Green dashed box: detector field of view at optimal position
- Green X: detector center position
- RA/Dec axes for absolute positioning

### Batch Processing Mode
Creates directory: `runs/batch_{YYYYMMDD}/`

**1. Batch Metadata JSON** (`batch_metadata.json`)

Records batch configuration for consistency when resuming:
```json
{
  "batch_date": "20250120",
  "created_timestamp": "2025-01-20T14:30:45.123456",
  "target_list": "targets.txt",
  "bad_pixel_map": "1_BadPixelMap_20250119.fits",
  "reference_image": "spirit.fits",
  "aperture_radius": 15,
  "grid_spacing": 3,
  "edge_padding": 10,
  "detector_width": 1024,
  "detector_height": 1280
}
```

**2. Batch Summary CSV** (`batch_summary_{YYYYMMDD}.csv`)

Consolidated results for all targets with columns:
- `gaia_id` - Gaia DR2 source ID
- `status` - SUCCESS or FAILED
- `timestamp` - ISO format timestamp
- `target_ra`, `target_dec` - Target sky coordinates (degrees)
- `optimal_x`, `optimal_y` - Optimal detector position (pixels)
- `detector_center_ra`, `detector_center_dec` - Detector center coordinates (degrees)
- `target_offset_x`, `target_offset_y` - Offset from detector center (pixels)
- `target_offset_ra_arcsec`, `target_offset_dec_arcsec` - Offset from center (arcseconds)
- `precision` - Predicted photometric precision
- `n_comparison_stars` - Number of usable comparison stars
- `combined_mag` - Combined magnitude of comparison stars
- `target_jmag` - Target J-band magnitude
- `target_zyj` - Target zYJ filter magnitude
- `target_teff` - Target effective temperature (K)
- `distance_to_hazard` - Distance to nearest bad pixel or edge (pixels)
- `reference_image` - Reference FITS file used
- `bad_pixel_map` - Bad pixel map file used
- `processing_time_seconds` - Processing time per target
- `error_message` - Error description (if failed)

**3. Individual Target Directories** (`{gaia_id}_{YYYYMMDD}/`)

Same outputs as single target mode for each successfully processed target.

## Resuming Interrupted Batches

Batch processing can be interrupted by network failures, connection drops, or other issues. The resume feature allows you to continue where you left off without reprocessing successful targets.

### How Resume Works

1. **Automatic State Tracking**: Each batch creates a `batch_metadata.json` file recording:
   - Bad pixel map used
   - Reference image
   - Critical parameters (aperture, grid spacing, edge padding, detector size)

2. **Consistency Validation**: When resuming, the code verifies that current parameters match the original batch. This prevents mixing results from different configurations.

3. **Progress Detection**: Reads `batch_summary_{YYYYMMDD}.csv` to identify:
   - `SUCCESS` targets (skip by default)
   - `FAILED` targets (retry by default)

4. **Appends Results**: New results are appended to the existing CSV, maintaining a complete record.

### Resume Examples

**Basic resume (retry failed targets):**
```bash
# Original batch started on Jan 20, interrupted
python src/predict_target_precision.py --batch targets.txt --optimize --save --viz

# Resume on Jan 21 or later
python src/predict_target_precision.py --batch targets.txt --resume 20250120 --optimize --save --viz
```

**Skip failed targets:**
```bash
# Don't retry targets that failed previously
python src/predict_target_precision.py --batch targets.txt --resume 20250120 --skip-failed --optimize
```

**Override configuration validation:**
```bash
# Force resume even if parameters changed (not recommended)
python src/predict_target_precision.py --batch targets.txt --resume 20250120 --force-config --optimize
```

### Resume Validation

The code validates that critical parameters match between original and resumed runs:
- Aperture radius
- Grid spacing
- Edge padding
- Detector dimensions

**If parameters differ**, the code will error with a message like:
```
ERROR: Configuration mismatch detected:
  - Aperture radius: 10 vs 15
  - Grid spacing: 5 vs 3

Cannot resume batch with different parameters.
Use --force-config to override (not recommended).
```

**Why this matters:** Mixing results from different grid spacings or aperture sizes would make the batch summary CSV inconsistent and scientifically invalid.

### Bad Pixel Map Consistency

When resuming, the code **always uses the original bad pixel map** from the initial run, even if:
- `download_BPM_from_server: true` in config
- A newer BPM is available on the server
- You're resuming days later

This ensures all targets in a batch use identical bad pixel maps for consistency.

## Bad Pixel Map Management

### Automatic Download from Server

When `download_BPM_from_server: true` in config:

1. Connects to server via SSH using stored keys
2. Searches for latest bad pixel map:
   - Checks `{base_path}/v2/{telescope}/output/{YYYYMMDD}/reduction/1_BadPixelMap.fits`
   - Tries v2, then v3 for each date
   - Goes back in time from yesterday to cutoff date (default: 2021-01-01)
3. Downloads to `BPMs/1_BadPixelMap_{YYYYMMDD}.fits`
4. Falls back to `bad_pixel_map_path` if download fails

**Requirements:**
- SSH key authentication already set up for the server
- `paramiko` Python package installed
- Credentials configured in `src/credentials.py`

### Manual Bad Pixel Map

When `download_BPM_from_server: false`:
- Uses file specified in `detector.bad_pixel_map_path`
- File should be in `BPMs/` directory
- FITS format: 0 = good pixel, 1 = bad pixel

## Algorithm Details

### Position Optimization

1. **Query Expansion**: Queries Gaia for a field ~2× detector size to ensure all potential comparison stars are captured regardless of target placement

2. **Parallel Grid Search**: Tests target positions at regular intervals (`grid_spacing_pixels`) across the detector within valid bounds (accounting for `edge_padding_pixels` and `aperture.radius_pixels`)
   - Uses all available CPU cores via `joblib` parallelization
   - Each worker independently tests one detector position
   - Typical speedup: 4-8× on multi-core systems

3. **Position Validation**: For each tested position:
   - Checks target aperture for bad pixel contamination
   - Projects comparison stars to detector coordinates
   - Filters comparison stars by:
     - Magnitude range relative to target
     - Position on detector (fully within bounds)
     - Clean apertures (no bad pixels)
   - Calculates precision using Decision Tree model

4. **Selection**: Chooses position with:
   - Best (minimum) predicted precision
   - If tied: maximum distance to nearest hazard (bad pixels or edges)

### Precision Prediction

Uses a trained Decision Tree regressor with features:
- Number of comparison stars
- Target zYJ magnitude
- Combined magnitude of comparison stars
- Target effective temperature

Model file: `configs/precision_prediction_model.joblib`

## Troubleshooting

**"No comparison stars found in magnitude range"**
- Target may be isolated or in sparse field
- Adjust `comparison_star_limits` in config to widen magnitude range

**"No valid positions found"**
- Detector may be too contaminated with bad pixels
- Try reducing `aperture.radius_pixels` or `edge_padding_pixels`
- Check that bad pixel map is correct

**"Target star not found in query results"**
- Gaia DR2 ID may be incorrect
- Target may not have 2MASS crossmatch (required for J-band magnitude)

**SSH connection fails for BPM download**
- Verify SSH key authentication is set up for the server
- Check `credentials.py` has correct hostname and username
- Set `download_BPM_from_server: false` and use local bad pixel map

**"Object of type PosixPath is not JSON serializable"**
- Path objects need converting to strings before JSON serialization
- This has been fixed in the modularized version

**Visualization shows wrong orientation**
- Survey image orientation may differ from detector
- This is cosmetic only; optimal positioning calculations are correct

**"Batch metadata not found" when resuming**
- The batch directory may be from an older version before metadata was saved
- Cannot safely resume - start a new batch

**"Configuration mismatch detected" when resuming**
- Current config differs from original batch parameters
- Either restore original config or use `--force-config` (not recommended)
- Mixing different parameters in one batch invalidates scientific consistency

**Resume completed but some targets still FAILED**
- Transient failures (network issues) may succeed on retry
- Persistent failures (no comparison stars, bad Gaia ID) will fail again
- Check `error_message` column in CSV for details
- Use `--skip-failed` on subsequent resumes to stop retrying persistent failures

**Gaia password warnings appearing multiple times**
- This is harmless - one warning per parallel worker
- Suppress with `warnings.filterwarnings('ignore', message='.*passwords.*inactivated.*')` at top of `gaia_queries.py`

## Performance Notes

- **Parallelization**: Grid search uses all CPU cores via `joblib`. Typical speedup: 4-8× on modern multi-core systems
- **Grid spacing**: Smaller values (1-2 pixels) are more thorough but slower. Values of 3-5 pixels provide good balance
- **Detector size**: Processing time scales with detector area and grid density. Typical 1024×1280 detector with spacing=3 takes ~30-60 seconds per target on modern hardware
- **Batch processing**: Processes targets sequentially. Use batch mode for unattended overnight runs
- **Gaia queries**: Can be slow (several minutes per target) due to complex joins with 2MASS. This is the main bottleneck, not the grid search

## Citation

If you use this tool in published research, please cite:
- Gaia DR2 catalog (Gaia Collaboration et al. 2018)
- 2MASS catalog (Skrutskie et al. 2006)
- SPECULOOS project (Delrez et al. 2018)