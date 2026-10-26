###### Acknowledgment



The team used Claude Code to assist with portions of this project, including code development and editing of the report text; all analysis decisions, results, and interpretations are the authors’ own and were verified by the team.

# Link to Google Doc version of report:
https://docs.google.com/document/d/1-0saHnEEm_DfWUmhjSHtMaCVOX7esOt2/edit?usp=sharing&ouid=107286652439241406113&rtpof=true&sd=true

# Gear spall prognostics — team notebook package (Group 09, SIADS 696)

Goal: estimate the severity of gear-tooth spall damage from externally mounted vibration sensors
(PHM 2026 Data Challenge). This package is a self-contained snapshot of the data extraction,
exploratory visualization and unsupervised-learning work, with the data needed to re-run all of it.

## Contents

```
spall\_prognostics.ipynb      the notebook (executed; outputs included). Parts: 1 data extraction,
                             2 visualization / EDA, 3 unsupervised learning + challenge submissions
requirements.txt             Python packages
data/                        everything the notebook reads (\~320 MB), see the table in the notebook intro
  channels.csv               units + HDF5 provenance for every column
  labels/                    per-tooth condition labels after every training run (1 healthy ... 4 severe)
  train|validation|test/     lf\_1s.parquet (1-s rows), particles.parquet (oil debris), manifest.csv,
                             submission templates
  train/lf\_sample/           12 raw HDF5 files (Exp-F Run-1) to demonstrate the flattening step
  train/photos\_sample/       4 tooth photos used as the damage rubric
  eda/photo\_proxy.csv        automated texture score per tooth photo (the 1 GB photo set is not shipped)
src/                         reference copies of the repo scripts the notebook was distilled from:
  phm\_data.py                download + un-nest the challenge shares (needs network; not run here)
  flatten\_hdf5.py            HDF5 -> parquet (the notebook re-implements its core on the sample files)
  spall\_hi.py                the unsupervised health-index pipeline
out/                         files the notebook writes (submission CSVs, label-free selection table)
```

## Run it

```bash
python -m venv .venv \&\& . .venv/bin/activate      # or conda / uv
pip install -r requirements.txt
jupyter lab spall\_prognostics.ipynb                # run from this folder: data/ must sit next to the notebook
```

Colab: upload the zip to the session (or mount Drive), then in a cell

```python
!unzip -q spall\_prognostics\_team.zip
%cd spall\_prognostics\_team
```

and run the notebook cells. Runtime is under a minute end to end; peak memory about 1.5 GB.

## Not included (too large)

Raw HDF5 archives (3.9 GB extracted), the high-frequency waveforms (\~20 GB per split) and the tooth
photos (1 GB). `src/phm\_data.py` recreates them from the challenge shares:

```bash
export PHM\_TEST\_SHARE\_PASSWORD=<password from the challenge>   # only for the test split
python src/phm\_data.py small \&\& python src/flatten\_hdf5.py
```

## Notes for collaborators

* Exp-A Run-2 is a byte copy of Run-1 in the source data; every table in the notebook blanks it.
* `test\_hours` is cumulative seconds with CI data within an experiment / 3600; it matches the
submission-template run-end times to within minutes.
* Labels are used only to evaluate models (Part 3), never to fit them or choose settings.
* The submission the notebook writes to `out/` (HI = `tooth`) is identical to the one in the repo.

