# Train either the Part A MLP baseline or the Part B CNN model on MNIST.
import argparse
import gzip
import os
from pathlib import Path
from struct import unpack

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')

import mynn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from draw_tools.plot import plot


BASE_DIR = Path(__file__).resolve().parent


def load_mnist_train(flatten):
    images_path = BASE_DIR / 'dataset' / 'MNIST' / 'train-images-idx3-ubyte.gz'
    labels_path = BASE_DIR / 'dataset' / 'MNIST' / 'train-labels-idx1-ubyte.gz'

    with gzip.open(images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32)
        if flatten:
            images = images.reshape(num, rows * cols)
        else:
            images = images.reshape(num, 1, rows, cols)

    with gzip.open(labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images / 255.0, labels


def split_train_valid(images, labels, valid_size, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(np.arange(labels.shape[0]))

    images = images[idx]
    labels = labels[idx]
    valid_images = images[:valid_size]
    valid_labels = labels[:valid_size]
    train_images = images[valid_size:]
    train_labels = labels[valid_size:]
    return train_images, train_labels, valid_images, valid_labels


def build_model(model_name, args, input_dim, max_classes):
    if model_name == 'mlp':
        model = nn.models.Model_MLP([input_dim, args.hidden_size, args.hidden_size_2, max_classes], 'ReLU')
        lr = args.lr if args.lr is not None else 0.1
        save_dir = args.save_dir or BASE_DIR / 'best_models' / 'mlp'
        fig_name = args.figure or 'part_a_mlp_learning_curve.png'
        eval_batch_size = args.eval_batch_size
    elif model_name == 'cnn':
        model = nn.models.Model_CNN(
            in_channels=1,
            input_size=28,
            conv_channels=args.conv_channels,
            second_conv_channels=args.second_conv_channels,
            kernel_size=args.kernel_size,
            pool_size=args.pool_size,
            num_classes=max_classes,
        )
        lr = args.lr if args.lr is not None else 0.05
        save_dir = args.save_dir or BASE_DIR / 'best_models' / 'cnn'
        fig_name = args.figure or 'part_b_cnn_learning_curve.png'
        eval_batch_size = args.eval_batch_size or 1000
    else:
        raise ValueError(f'Unknown model: {model_name}')

    return model, lr, Path(save_dir), fig_name, eval_batch_size


def parse_args():
    parser = argparse.ArgumentParser(description='Train MLP or CNN on MNIST.')
    parser.add_argument('--model', choices=['mlp', 'cnn'], default='mlp')
    parser.add_argument('--seed', type=int, default=309)
    parser.add_argument('--valid-size', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--log-iters', type=int, default=100)
    parser.add_argument('--eval-iters', type=int, default=100)
    parser.add_argument('--eval-batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--save-dir', type=Path, default=None)
    parser.add_argument('--figure', type=str, default=None)

    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--hidden-size-2', type=int, default=16)
    parser.add_argument('--conv-channels', type=int, default=4)
    parser.add_argument('--second-conv-channels', type=int, default=8)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--pool-size', type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    flatten = args.model == 'mlp'
    images, labels = load_mnist_train(flatten=flatten)
    train_images, train_labels, valid_images, valid_labels = split_train_valid(
        images,
        labels,
        valid_size=args.valid_size,
        seed=args.seed,
    )

    input_dim = train_images.shape[-1] if args.model == 'mlp' else 28 * 28
    max_classes = int(train_labels.max()) + 1
    model, lr, save_dir, fig_name, eval_batch_size = build_model(
        args.model,
        args,
        input_dim=input_dim,
        max_classes=max_classes,
    )

    optimizer = nn.optimizer.SGD(init_lr=lr, model=model)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=max_classes)
    runner = nn.runner.RunnerM(
        model,
        optimizer,
        nn.metric.accuracy,
        loss_fn,
        batch_size=args.batch_size,
    )

    runner.train(
        [train_images, train_labels],
        [valid_images, valid_labels],
        num_epochs=args.epochs,
        log_iters=args.log_iters,
        eval_iters=args.eval_iters,
        eval_batch_size=eval_batch_size,
        save_dir=save_dir,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes.reshape(-1)
    plot(runner, axes)
    fig.tight_layout(w_pad=3.0)

    fig_dir = BASE_DIR / 'figs'
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(fig_dir / fig_name, dpi=200, bbox_inches='tight')
    print(f"Model: {args.model}")
    print(f"Learning rate: {lr}")
    print(f"Best validation accuracy: {runner.best_score:.5f}")


if __name__ == '__main__':
    main()
