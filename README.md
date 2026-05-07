# Deep Learning Project 1

NumPy implementation for MNIST classification in Project 1 of Neural Network and
Deep Learning. The repository contains the code, report source, generated figures,
and reproducibility scripts. It intentionally does not include the MNIST dataset
or trained checkpoint pickle files.

## Repository Layout

```text
codes/
  mynn/                         Core NumPy layers, models, optimizer, runner
  test_train.py                 Train the Part A MLP or Part B CNN baseline
  test_model.py                 Evaluate saved MLP/CNN checkpoints on MNIST test set
  experiment_part_c_recipe.py   Optimization and regularization experiments
  analysis_visualization.py     Confusion matrices, misclassified samples, kernels
  figs/                         Learning curves used by the report
  part_c_results/               Result JSON files and visualization figures
report/
  report.tex                    Report source
  report.pdf                    Submitted report PDF
scripts/
  download_checkpoints.py       Download checkpoints from ModelScope
  prepare_modelscope_upload.py  Build a local ModelScope upload directory
  verify_submission.py          One-command sanity check for grading
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the provided MNIST gzip files under:

```text
codes/dataset/MNIST/
  train-images-idx3-ubyte.gz
  train-labels-idx1-ubyte.gz
  t10k-images-idx3-ubyte.gz
  t10k-labels-idx1-ubyte.gz
```

The dataset is not tracked in Git because the project PDF asks not to upload it.

## Download Checkpoints

The trained checkpoints are hosted on ModelScope:

```text
https://modelscope.cn/models/ParryY/Deep-Learning-Project1
```

Download and place them into the expected local paths with:

```bash
python scripts/download_checkpoints.py
```

The script first tries the ModelScope SDK if it is installed, then falls back to
`git clone https://www.modelscope.cn/ParryY/Deep-Learning-Project1.git`.

## Verify

After installing dependencies, placing the dataset, and downloading checkpoints:

```bash
python scripts/verify_submission.py
```

Expected checkpoint evaluation results:

```text
MLP baseline test accuracy: 0.9544
CNN baseline test accuracy: 0.9779
Best recipe CNN test accuracy: 0.9856
```

## Reproduce Main Experiments

Run from `codes/`:

```bash
cd codes
python test_train.py --model mlp
python test_train.py --model cnn
python test_model.py --model mlp
python test_model.py --model cnn
python experiment_part_c_recipe.py \
  --recipes sgd,momentum,sgd_step,adam,l2,dropout,recipe_combo \
  --epochs 5 --eval-batch-size 1000 \
  --out-dir part_c_results/recipe_full
python analysis_visualization.py
```

No deep learning framework modules are used for the neural network operators.
Core layers and optimizers are implemented with NumPy.
