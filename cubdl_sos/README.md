# CUBDL speed-of-sound comparison

This tool estimates the global speed of sound of each CUBDL plane-wave acquisition with eight
methods, and compares each estimate with the CUBDL reference value (the "Optimized Speed of Sound" of the CUBDL
phantom details, the values used for the CUBDL evaluation).

The reference values come with the licensed CUBDL release and are not in this repository. Make
`results/reference_sos.csv` with the columns `file,reference_sos` (one row per file) from the
"Optimized Speed of Sound" column of the CUBDL phantom details before step 5.

| Method | Type | Objective |
|---|---|---|
| `nominal` | none | 1540 m/s (the value that all institutions submitted) |
| `grid_cf` | grid search | maximum of the mean coherence factor |
| `grid_bright` | grid search | maximum of the mean speckle brightness |
| `grid_sharp` | grid search | maximum of the image sharpness (normalized 4th moment) |
| `grid_phase` | grid search | maximum of the inter-angle phase coherence |
| `gd_cf` | gradient descent (this work) | coherence factor |
| `gd_bright` | gradient descent (this work) | speckle brightness |
| `gd_phase` | gradient descent | phase error between the images of angle pairs, averaged in patches (the loss type of differentiable-beamforming autofocus, Simson et al.; adapted here to compounded plane waves) |
| `gds_cf`, `gds_bright`, `gds_phase` | gradient descent on the objective smoothed with a Gaussian in c (width 10 to 0.5 m/s, graduated optimization) | the same three objectives |

`gd_phase` is not the code of Simson et al. Their method estimates a speed-of-sound map from
full synthetic-aperture data with a common-midpoint phase error. This tool uses the same type of
loss for a global speed of sound and plane-wave data. Describe it in this way in the paper.

All methods use the same differentiable delay-and-sum beamformer (JAX), with all transmit angles
compounded, a receive F-number of 1, and the same pixel grid. Gradient descent uses Adam on c,
starts at 1540 m/s, and gets the exact derivative dc by forward-mode differentiation.

## Folder contents

This folder is complete: copy it to any location and run it there. It needs only Docker (and, for
the GPU image, an NVIDIA GPU with its driver).

| Path | Function |
|---|---|
| `data/` | Put the CUBDL `.hdf5` / `.h5` files here (git ignores them) |
| `run.sh`, `run.ps1` | One command for each step (Git Bash or Linux; PowerShell) |
| `run_sos_comparison.py` | Main program: inspection, beamformer check, comparison |
| `cubdl_io.py` | Loads a CUBDL file; finds the array layout from the sizes |
| `das.py` | Differentiable plane-wave delay-and-sum and the four metrics |
| `summarize.py` | Table of errors, LaTeX table and scatter plot |
| `make_synthetic.py` | Synthetic test file in the CUBDL layout (known speed of sound) |
| `Dockerfile.gpu`, `Dockerfile`, `requirements.txt` | Docker images: GPU (CUDA 12) and CPU |
| `test/`, `results/` | Made by the run: test output and comparison results |

## Procedure

1. Copy this folder to the new location.
2. Put the CUBDL files in `data/`.
3. Start Docker Desktop.
4. Open a terminal in this folder and run the steps below. Use `bash run.sh <step>` in Git Bash
   or Linux, or `.
un.ps1 <step>` in PowerShell. The scripts use the GPU image if `nvidia-smi`
   finds a GPU, and the CPU image if not.

| Step | Command | Time | What to look at |
|---|---|---|---|
| 1 | `build` | 5 to 10 min, once | No errors |
| 2 | `test` | about 1 min | The first line shows `JAX device: cuda:0` (GPU). `grid_cf` and `gd_cf` are within about 5 m/s of 1500 |
| 3 | `check` | about 1 min | One file of each institution. The best NCC of each file is 0.6 or more, and each `test/check_<name>.png` shows the same targets in the two images |
| 4 | `run` | GPU: 30 to 70 s for each file (about 40 min for 49 files); CPU: about 5 to 10 times longer | One line of estimates for each file |
| 5 | `summary` | seconds | `results/summary.txt` |

`all` does steps 1 to 5 in sequence. `run` can stop and start again: it does not repeat the files
that are complete. To test a few files first: `bash run.sh run --files OSL002 MYO001`.

Options (environment variables):
- `CUBDL_DATA`: a different data folder, for example `CUBDL_DATA=/d/cubdl bash run.sh run`
  (PowerShell: `$env:CUBDL_DATA = "D:\cubdl"`).
- `CUBDL_CPU=1`: do not use the GPU.

If a check gives an NCC below 0.6, look at `test/check_<name>.png` before you trust the estimates of
that institution.

## Output

| File | Contents |
|---|---|
| `results/results.csv` | One row for each file and method: estimate, time, number of evaluations, selected convention, NCC of the check |
| `results/curves/<name>.json` | Metric value against c (grid), and the gradient-descent trajectories |
| `results/summary.txt` | MAE, median error, bias, maximum error, files within 10 and 20 m/s, time; MAE for each institution |
| `results/summary_table.tex` | The same table in LaTeX |
| `results/scatter.png` | Estimate against reference, one panel for each method |

## Direct use of the program

The scripts call `run_sos_comparison.py` in the Docker image. You can also call it directly:

```bash
python run_sos_comparison.py --inspect data/EUT001.hdf5          # print the file layout
python run_sos_comparison.py --check data/EUT001.hdf5            # beamformer check
python run_sos_comparison.py --data data --out results/results.csv
python summarize.py results/results.csv
```

This needs Python 3.11 and the packages in `requirements.txt` (for a GPU: `pip install "jax[cuda12]"`).

## Options

| Option | Default | Function |
|---|---|---|
| `--max-angles` | 0 (all) | Use only this number of angles (evenly spaced). Faster, but the landscapes become less smooth |
| `--nx`, `--nz` | 128, 160 | Maximum pixel grid for the estimation (sub-sampled from `pixel_positions`) |
| `--zmin` | none | Minimum depth (m); default: the CUBDL depth range |
| `--fnum` | 1.0 | Receive F-number |
| `--step` | 5 | Coarse grid step (m/s). A fine search with a 1 m/s step follows around each maximum |
| `--iters` | 100 | Maximum number of gradient-descent iterations |
| `--convention` | cubdl | `cubdl`: CUBDL loader conventions; `auto`: search |
| `--assume-max-angle` | 0 | Angle span (deg) for files with no angles (default 16 for INS) |
| `--max-offset` | 4e-6 | Range (s) of the time-offset search |
| `--patch`, `--pair-sep` | 16, T/5 | Patch length and angle separation of the phase metric |
| `--chunk` | 4096 (CPU), 16384 (GPU) | Pixels for each chunk. Decrease it if the GPU memory is not sufficient |

## CUBDL conventions

The five institutions store their data with different keys, probes and time references. By default
(`--convention cubdl`), the tool uses the geometry and the time-zero rule of the official CUBDL loader
(`datasets/PWDataLoaders.py` of the CUBDL code). The CUBDL reference speeds of sound were found with
these rules, so the comparison is fair. The pixel grid covers the aperture and the CUBDL depth range.

| Institution | Elements | Angles | Time zero (CUBDL) | Depth (mm) |
|---|---|---|---|---|
| OSL | `element_positions`, centred | `transmit_direction` | −`start_time` | 10–65 |
| MYO | L11-4v, pitch 0.3 mm (no positions in the file) | `angles` (rad) | aperture-edge term | 5–55 |
| UFL | LA533, pitch 0.245 mm (no positions in the file) | `angles` (deg) | −`channel_data_t0` | 10–50 |
| EUT | `element_positions`, centred | `transmit_direction` | aperture-edge term + 10/fc | 10–80 |
| INS | `element_positions`, centred | −16° to 16° (not in the file) | −`start_time` + aperture-edge term | 10–60 |

If the file contains a beamformed image with pixel positions (OSL, EUT, UFL), `check` reports the NCC
between that image and this beamformer, for information only. The images of the institutions are not
all made with the CUBDL conventions, so a low NCC is not always an error; look at the PNG.

`--convention auto` searches the convention instead (transmit reference, start-time sign, time offset,
angle span), by the NCC with the file image or, without an image, by the highest coherence factor over
all speeds of sound. Use it only for files of other sources.

## Known limits

- The grid range is 1400 to 1680 m/s. Gradient descent is also limited to this range.
- Plain gradient descent (`gd_*`) can stop at a small local maximum of the landscape, mainly when the
  start point is far from the optimum. The smoothed version (`gds_*`) avoids most of these cases.
- Gradient descent can stop at a local maximum. On a synthetic file with 11 angles, the phase
  objective stopped at a local maximum; with 31 angles, all gradient methods found the true value.
  The `curves/` files show the landscape of each file, so you can see where this occurs.
- For one scalar parameter, a grid search is not slower than gradient descent. The advantage of
  gradients is for many parameters at the same time. Do not claim a speed advantage for the global
  speed of sound.
- If a file does not contain `beamformed_data`, the tool cannot check the delay convention. It then
  uses the standard convention and prints a message.
