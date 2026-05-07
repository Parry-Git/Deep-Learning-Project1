import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.core as np_core
import numpy.core.numeric as np_core_numeric
import sys

import mynn as nn
from test_model import load_mnist_test


BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES = [str(i) for i in range(10)]

# Some saved models were pickled with NumPy 2.x module paths and are loaded here
# under NumPy 1.x in the huawei environment.
sys.modules.setdefault('numpy._core', np_core)
sys.modules.setdefault('numpy._core.numeric', np_core_numeric)


def predict_batches(model, images, batch_size):
    if hasattr(model, 'eval'):
        model.eval()
    logits = []
    for start in range(0, images.shape[0], batch_size):
        end = min(start + batch_size, images.shape[0])
        logits.append(model(images[start:end]))
    logits = np.concatenate(logits, axis=0)
    probs = nn.op.softmax(logits)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    return logits, probs, preds, confs


def confusion_matrix(labels, preds, num_classes=10):
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y, p in zip(labels, preds):
        mat[int(y), int(p)] += 1
    return mat


def plot_confusion(mat, out_path, title):
    row_sum = mat.sum(axis=1, keepdims=True)
    normalized = mat / np.maximum(row_sum, 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(10):
        for j in range(10):
            if mat[i, j] > 0:
                color = 'white' if normalized[i, j] > 0.5 else 'black'
                ax.text(j, i, str(mat[i, j]), ha='center', va='center', fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def class_accuracy(mat):
    return np.diag(mat) / np.maximum(mat.sum(axis=1), 1)


def top_confusions(mat, k=8):
    off_diag = mat.copy()
    np.fill_diagonal(off_diag, 0)
    pairs = []
    for true_label, pred_label in np.argwhere(off_diag > 0):
        pairs.append({
            'true': int(true_label),
            'pred': int(pred_label),
            'count': int(off_diag[true_label, pred_label]),
        })
    pairs.sort(key=lambda item: item['count'], reverse=True)
    return pairs[:k]


def plot_misclassified(images, labels, preds, confs, out_path, title, limit=25):
    wrong_idx = np.where(labels != preds)[0]
    if wrong_idx.size == 0:
        return []
    order = wrong_idx[np.argsort(confs[wrong_idx])[::-1]]
    selected = order[:limit]
    cols = 5
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.1))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for ax, idx in zip(axes, selected):
        ax.imshow(images[idx, 0], cmap='gray')
        ax.set_title(f'T:{labels[idx]} P:{preds[idx]}\nconf:{confs[idx]:.2f}', fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return selected.tolist()


def plot_model_comparison_examples(images, labels, mlp_preds, cnn_preds, combo_preds, out_path, limit=20):
    fixed = np.where((mlp_preds != labels) & (cnn_preds == labels))[0]
    still_wrong = np.where((cnn_preds != labels) & (combo_preds != labels))[0]
    selected = np.concatenate([fixed[:limit // 2], still_wrong[:limit - min(len(fixed), limit // 2)]])
    if selected.size == 0:
        return []
    cols = 5
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.2))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for ax, idx in zip(axes, selected):
        ax.imshow(images[idx, 0], cmap='gray')
        ax.set_title(
            f'T:{labels[idx]} M:{mlp_preds[idx]}\nC:{cnn_preds[idx]} R:{combo_preds[idx]}',
            fontsize=8,
        )
    fig.suptitle('Examples where model choices change the prediction')
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return selected.astype(int).tolist()


def plot_mlp_weights(model, out_path, limit=16):
    first_linear = next(layer for layer in model.layers if layer.optimizable)
    W = first_linear.params['W']
    norms = np.linalg.norm(W, axis=0)
    selected = np.argsort(norms)[::-1][:limit]
    cols = 4
    rows = int(np.ceil(limit / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = np.array(axes).reshape(-1)
    vmax = np.max(np.abs(W[:, selected]))
    for ax, unit in zip(axes, selected):
        ax.imshow(W[:, unit].reshape(28, 28), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'unit {unit}', fontsize=8)
        ax.axis('off')
    for ax in axes[len(selected):]:
        ax.axis('off')
    fig.suptitle('MLP first-layer weights with largest L2 norm')
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return selected.astype(int).tolist()


def optimizable_layers(model):
    return [layer for layer in model.layers if layer.optimizable]


def plot_cnn_kernels(model, out_path):
    conv_layers = [layer for layer in model.layers if layer.__class__.__name__ == 'conv2D']
    conv1 = conv_layers[0].params['W']
    conv2 = conv_layers[1].params['W'] if len(conv_layers) > 1 else None

    rows = 2 if conv2 is not None else 1
    cols = max(conv1.shape[0], conv2.shape[0] if conv2 is not None else 0)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.6))
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.reshape(-1):
        ax.axis('off')

    vmax1 = np.max(np.abs(conv1))
    for i in range(conv1.shape[0]):
        axes[0, i].imshow(conv1[i, 0], cmap='RdBu_r', vmin=-vmax1, vmax=vmax1)
        axes[0, i].set_title(f'conv1-{i}', fontsize=8)

    if conv2 is not None:
        # Use signed average over input channels to show the coarse pattern each
        # second-layer filter combines from first-layer responses.
        conv2_avg = np.mean(conv2, axis=1)
        vmax2 = np.max(np.abs(conv2_avg))
        for i in range(conv2_avg.shape[0]):
            axes[1, i].imshow(conv2_avg[i], cmap='RdBu_r', vmin=-vmax2, vmax=vmax2)
            axes[1, i].set_title(f'conv2-{i}', fontsize=8)

    fig.suptitle('CNN convolution kernels')
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def collect_cnn_activations(model, image):
    if hasattr(model, 'eval'):
        model.eval()
    X = image[None, :, :, :]
    outputs = {'input': X.copy()}
    conv_count = 0
    relu_count = 0
    pool_count = 0
    out = X
    for layer in model.layers:
        out = layer(out)
        name = layer.__class__.__name__
        if name == 'conv2D':
            conv_count += 1
            outputs[f'conv{conv_count}'] = out.copy()
        elif name == 'ReLU':
            relu_count += 1
            outputs[f'relu{relu_count}'] = out.copy()
        elif name == 'MaxPool2D':
            pool_count += 1
            outputs[f'pool{pool_count}'] = out.copy()
    return outputs


def plot_feature_maps(activations, out_path, title, channels_per_stage=4):
    stages = ['input', 'conv1', 'relu1', 'pool1', 'conv2', 'relu2']
    stages = [stage for stage in stages if stage in activations]
    cols = channels_per_stage
    rows = len(stages)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.7))
    axes = np.array(axes).reshape(rows, cols)
    for row, stage in enumerate(stages):
        data = activations[stage][0]
        if data.ndim == 2:
            data = data[None, :, :]
        scores = np.max(np.abs(data), axis=(1, 2))
        selected = np.argsort(scores)[::-1][:cols]
        for col in range(cols):
            ax = axes[row, col]
            ax.axis('off')
            if col < len(selected):
                channel = selected[col]
                fmap = data[channel]
                cmap = 'gray' if stage == 'input' else 'viridis'
                ax.imshow(fmap, cmap=cmap)
                ax.set_title(f'{stage} c{channel}', fontsize=8)
        axes[row, 0].set_ylabel(stage, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def activation_foreground_stats(activations):
    image = activations['input'][0, 0]
    foreground = image > 0.2
    stats = {}
    for stage, data in activations.items():
        if stage == 'input' or data.ndim != 4:
            continue
        fmap = np.mean(np.maximum(data[0], 0), axis=0)
        if fmap.shape != foreground.shape:
            y_scale = foreground.shape[0] // fmap.shape[0]
            x_scale = foreground.shape[1] // fmap.shape[1]
            fg = foreground.reshape(fmap.shape[0], y_scale, fmap.shape[1], x_scale).max(axis=(1, 3))
        else:
            fg = foreground
        fg_mean = float(np.mean(fmap[fg])) if np.any(fg) else 0.0
        bg_mean = float(np.mean(fmap[~fg])) if np.any(~fg) else 0.0
        stats[stage] = {
            'foreground_mean': fg_mean,
            'background_mean': bg_mean,
            'foreground_background_ratio': fg_mean / max(bg_mean, 1e-12),
        }
    return stats


def load_models(args):
    mlp = nn.models.Model_MLP()
    mlp.load_model(args.mlp_model)
    cnn = nn.models.Model_CNN()
    cnn.load_model(args.cnn_model)
    combo = nn.models.Model_CNN()
    combo.load_model(args.combo_model)
    return mlp, cnn, combo


def parse_args():
    parser = argparse.ArgumentParser(description='Error analysis and visualization for MNIST models.')
    parser.add_argument('--mlp-model', type=Path, default=BASE_DIR / 'best_models' / 'mlp' / 'best_model.pickle')
    parser.add_argument('--cnn-model', type=Path, default=BASE_DIR / 'best_models' / 'cnn' / 'best_model.pickle')
    parser.add_argument('--combo-model', type=Path, default=BASE_DIR / 'part_c_results' / 'recipe_full' / 'recipe_combo' / 'best_model.pickle')
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--out-dir', type=Path, default=BASE_DIR / 'part_c_results' / 'error_analysis')
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mlp, cnn, combo = load_models(args)
    flat_images, labels = load_mnist_test(flatten=True)
    images, _ = load_mnist_test(flatten=False)

    _, _, mlp_preds, mlp_confs = predict_batches(mlp, flat_images, args.batch_size)
    _, _, cnn_preds, cnn_confs = predict_batches(cnn, images, args.batch_size)
    _, _, combo_preds, combo_confs = predict_batches(combo, images, args.batch_size)

    model_outputs = {
        'mlp': (mlp_preds, mlp_confs),
        'cnn': (cnn_preds, cnn_confs),
        'recipe_combo': (combo_preds, combo_confs),
    }

    metrics = {}
    for name, (preds, confs) in model_outputs.items():
        mat = confusion_matrix(labels, preds)
        accuracy = float(np.mean(preds == labels))
        metrics[name] = {
            'accuracy': accuracy,
            'num_errors': int(np.sum(preds != labels)),
            'class_accuracy': class_accuracy(mat).round(6).tolist(),
            'top_confusions': top_confusions(mat),
        }
        plot_confusion(
            mat,
            args.out_dir / f'{name}_confusion_matrix.png',
            f'{name} confusion matrix (test accuracy {accuracy:.4f})',
        )
        selected = plot_misclassified(
            images,
            labels,
            preds,
            confs,
            args.out_dir / f'{name}_misclassified.png',
            f'{name} high-confidence misclassified examples',
        )
        metrics[name]['high_confidence_misclassified_indices'] = selected

    metrics['comparison_example_indices'] = plot_model_comparison_examples(
        images,
        labels,
        mlp_preds,
        cnn_preds,
        combo_preds,
        args.out_dir / 'model_comparison_examples.png',
    )

    metrics['mlp_weight_units'] = plot_mlp_weights(
        mlp,
        args.out_dir / 'mlp_first_layer_weights.png',
    )
    plot_cnn_kernels(cnn, args.out_dir / 'cnn_kernels.png')

    correct = np.where(combo_preds == labels)[0]
    correct = correct[np.argsort(combo_confs[correct])[::-1]]
    wrong = np.where(combo_preds != labels)[0]
    wrong = wrong[np.argsort(combo_confs[wrong])[::-1]]
    activation_indices = []
    if correct.size > 0:
        activation_indices.append(int(correct[0]))
    if wrong.size > 0:
        activation_indices.append(int(wrong[0]))

    metrics['activation_examples'] = []
    for idx in activation_indices:
        activations = collect_cnn_activations(combo, images[idx])
        out_path = args.out_dir / f'cnn_activations_idx_{idx}_true_{labels[idx]}_pred_{combo_preds[idx]}.png'
        plot_feature_maps(
            activations,
            out_path,
            f'CNN activations: index {idx}, true {labels[idx]}, pred {combo_preds[idx]}',
        )
        metrics['activation_examples'].append({
            'index': idx,
            'true': int(labels[idx]),
            'pred': int(combo_preds[idx]),
            'confidence': float(combo_confs[idx]),
            'path': str(out_path),
            'foreground_stats': activation_foreground_stats(activations),
        })

    with open(args.out_dir / 'analysis_summary.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
