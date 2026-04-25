# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')
import mynn as nn

import numpy as np
from struct import unpack
import gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from draw_tools.plot import plot
import pickle

# fixed seed for experiment
np.random.seed(309)

BASE_DIR = Path(__file__).resolve().parent
train_images_path = BASE_DIR / 'dataset' / 'MNIST' / 'train-images-idx3-ubyte.gz'
train_labels_path = BASE_DIR / 'dataset' / 'MNIST' / 'train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28).astype(np.float32)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open(BASE_DIR / 'idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / 255.0
valid_imgs = valid_imgs / 255.0

linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 256, 10], 'ReLU')
optimizer = nn.optimizer.SGD(init_lr=0.1, model=linear_model)
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, batch_size=128)

runner.train(
        [train_imgs, train_labs],
        [valid_imgs, valid_labs],
        num_epochs=5,
        log_iters=100,
        eval_iters=100,
        save_dir=BASE_DIR / 'best_models',
)

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

fig_dir = BASE_DIR / 'figs'
os.makedirs(fig_dir, exist_ok=True)
plt.savefig(fig_dir / 'part_a_mlp_learning_curve.png', dpi=200)
print(f"Best validation accuracy: {runner.best_score:.5f}")
