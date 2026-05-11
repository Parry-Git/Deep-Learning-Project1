# Evaluate a saved MLP or CNN model on the MNIST test set.
import argparse
import gzip
import sys
from pathlib import Path
from struct import unpack

import numpy as np
import numpy.core as _np_core
import numpy.core.numeric as _np_core_numeric
sys.modules.setdefault('numpy._core', _np_core)
sys.modules.setdefault('numpy._core.numeric', _np_core_numeric)

import mynn as nn


BASE_DIR = Path(__file__).resolve().parent


def load_mnist_test(flatten):
    images_path = BASE_DIR / 'dataset' / 'MNIST' / 't10k-images-idx3-ubyte.gz'
    labels_path = BASE_DIR / 'dataset' / 'MNIST' / 't10k-labels-idx1-ubyte.gz'

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


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MLP or CNN on MNIST.')
    parser.add_argument('--model', choices=['mlp', 'cnn'], default='mlp')
    parser.add_argument('--model-path', type=Path, default=None)
    parser.add_argument('--batch-size', type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    flatten = args.model == 'mlp'
    test_images, test_labels = load_mnist_test(flatten=flatten)

    if args.model == 'mlp':
        model = nn.models.Model_MLP()
        model_path = args.model_path or BASE_DIR / 'best_models' / 'mlp' / 'best_model.pickle'
    else:
        model = nn.models.Model_CNN()
        model_path = args.model_path or BASE_DIR / 'best_models' / 'cnn' / 'best_model.pickle'
    model.load_model(model_path)

    total_acc = 0.0
    total_num = 0
    for start in range(0, test_images.shape[0], args.batch_size):
        end = min(start + args.batch_size, test_images.shape[0])
        logits = model(test_images[start:end])
        batch_num = end - start
        total_acc += nn.metric.accuracy(logits, test_labels[start:end]) * batch_num
        total_num += batch_num

    print(total_acc / total_num)


if __name__ == '__main__':
    main()
