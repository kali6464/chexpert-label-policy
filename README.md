# CheXpert: label policy, backbone, and evaluation

Code for the paper *What Do CheXpert Comparisons Measure? Label Policy, Backbone, and Evaluation in a Controlled Study* (K. Ali and S. A. Ludwig, North Dakota State University).

The study trains DenseNet-121, DeiT-Base, and ConvNeXt-Base on CheXpert under several uncertain-label policies, with three seeds each. It evaluates them with paired bootstrap comparisons on the CheXpert test set and checks them on NIH ChestX-ray14.

## Data (not included)

No images or labels are distributed here. Obtain them from their providers under their own terms:

- CheXpert-v1.0: Stanford AIMI
- CheXpert test-set labels: CheXlocalize release
- NIH ChestX-ray14: NIH Clinical Center

`selection_patient_ids.txt` lists the 2,000 training patients held out for checkpoint selection (`make_split.py`, NumPy generator, seed 0).

## Setup

- Python with PyTorch and timm; see `requirements.txt`.
- Edit the paths at the top of `config.yaml` to point to your copy of the data.
- The `.pbs` files are PBS job scripts for the CCAST cluster. Adapt them to your scheduler, or run the Python commands inside them directly.

## Run order

| Step | Files | What it does |
|---|---|---|
| 1 | `make_split.py` | Holds out the 2,000-patient selection set |
| 2 | `preprocess_resize.py`, `resize.pbs` | Resizes all images once to 384 x 384 |
| 3 | `submit_grid.sh`, `train.pbs`, `train.py` | Main grid: 3 backbones x 3 policies x 3 seeds, plus U-Ones |
| 4 | `score_test.py`, `score_test.pbs` | Scores every run on the test set |
| 5 | `compare.py`, `compare_test.py` | Paired bootstrap comparisons; output in `test_comparisons.txt` |
| 6 | `policy.pbs`, `train_policy.py`, `learned_policy.py` | Learned per-label policy (secondary) |
| 7 | `download_nih.sh`, `prepare_nih.py`, `score_external.py`, `score_external.pbs`, `summary_nih_patients.pbs` | External check on NIH ChestX-ray14 (patient-level bootstrap) |
| 8 | `submit_ignore.sh`, `smoke_ignore.pbs`, `score_ignore.pbs` | Exploratory U-Ignore runs (`train.py` with `uncertainty_policy=ignore`) |
| 9 | `submit_sensitivity*.sh`, `train_sens*.pbs`, `score_sens*.pbs`, `summarize_sensitivity*.py` | Exploratory check of learning rate (0.5x, 2x) and a 6-epoch schedule |
| 10 | `fig*.py`, `fig_common.py` | Figures |

Shared code: `common.py` (label policies), `data_loader.py`, `model.py`, `evaluate.py`. `smoke*.pbs` are short test jobs.

`runs_sens/*/config_used.yaml` holds the exact configuration of each training-settings run.

## Main settings

AdamW, batch 32, 3 epochs (20,316 steps), 1,000 warm-up steps, cosine decay to 1% of peak, mixed precision, gradient clipping at 1.0, checkpoint chosen on the selection set every 1,000 steps. Peak learning rate and weight decay:

- DenseNet-121: 1e-4 and 1e-5
- DeiT-Base (`deit_base_distilled_patch16_384`): 3e-5 and 0.05
- ConvNeXt-Base (`convnext_base.fb_in22k_ft_in1k`): 5e-5 and 0.05

Seeds: 42, 43, 44.
