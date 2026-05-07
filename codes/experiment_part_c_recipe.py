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

import mynn as nn
from draw_tools.plot import plot
from test_model import load_mnist_test
from test_train import load_mnist_train, split_train_valid


BASE_DIR = Path(__file__).resolve().parent


RECIPES = {
    'sgd': {
        'optimizer': 'sgd',
        'scheduler': 'none',
        'lr': 0.05,
        'weight_decay': 0.0,
        'dropout': 0.0,
        'early_stopping': None,
    },
    'momentum': {
        'optimizer': 'momentum',
        'scheduler': 'none',
        'lr': 0.05,
        'weight_decay': 0.0,
        'dropout': 0.0,
        'early_stopping': None,
    },
    'sgd_step': {
        'optimizer': 'sgd',
        'scheduler': 'step',
        'lr': 0.05,
        'weight_decay': 0.0,
        'dropout': 0.0,
        'early_stopping': None,
    },
    'momentum_multistep': {
        'optimizer': 'momentum',
        'scheduler': 'multistep',
        'lr': 0.05,
        'weight_decay': 0.0,
        'dropout': 0.0,
        'early_stopping': None,
    },
    'adam': {
        'optimizer': 'adam',
        'scheduler': 'none',
        'lr': 0.001,
        'weight_decay': 0.0,
        'dropout': 0.0,
        'early_stopping': None,
    },
    'l2': {
        'optimizer': 'momentum',
        'scheduler': 'none',
        'lr': 0.05,
        'weight_decay': 1e-4,
        'dropout': 0.0,
        'early_stopping': None,
    },
    'dropout': {
        'optimizer': 'momentum',
        'scheduler': 'none',
        'lr': 0.05,
        'weight_decay': 0.0,
        'dropout': 0.2,
        'early_stopping': None,
    },
    'early_stop': {
        'optimizer': 'momentum',
        'scheduler': 'none',
        'lr': 0.05,
        'weight_decay': 0.0,
        'dropout': 0.0,
        'early_stopping': 8,
    },
    'recipe_combo': {
        'optimizer': 'momentum',
        'scheduler': 'multistep',
        'lr': 0.05,
        'weight_decay': 1e-4,
        'dropout': 0.1,
        'early_stopping': 8,
    },
}


def make_model(args, recipe, num_classes):
    return nn.models.Model_CNN(
        in_channels=1,
        input_size=28,
        conv_channels=args.conv_channels,
        second_conv_channels=args.second_conv_channels,
        kernel_size=args.kernel_size,
        pool_size=args.pool_size,
        num_classes=num_classes,
        weight_decay_lambda=recipe['weight_decay'],
        dropout_rate=recipe['dropout'],
    )


def make_optimizer(model, recipe, args):
    name = recipe['optimizer']
    lr = recipe['lr']
    if name == 'sgd':
        return nn.optimizer.SGD(init_lr=lr, model=model)
    if name == 'momentum':
        return nn.optimizer.MomentGD(init_lr=lr, model=model, mu=args.momentum)
    if name == 'adagrad':
        return nn.optimizer.AdaGrad(init_lr=lr, model=model)
    if name == 'rmsprop':
        return nn.optimizer.RMSProp(init_lr=lr, model=model)
    if name == 'adam':
        return nn.optimizer.Adam(init_lr=lr, model=model)
    raise ValueError(f'Unknown optimizer: {name}')


def make_scheduler(optimizer, recipe, args, num_batches):
    name = recipe['scheduler']
    if name == 'none':
        return None
    if name == 'step':
        step_size = args.step_size or max(1, num_batches * 2)
        return nn.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=args.lr_gamma)
    if name == 'multistep':
        if args.milestones is not None:
            milestones = [int(item) for item in args.milestones.split(',') if item]
        else:
            milestones = [max(1, num_batches * 2), max(1, num_batches * 4)]
        return nn.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)
    if name == 'exp':
        return nn.lr_scheduler.ExponentialLR(optimizer, gamma=args.exp_gamma)
    raise ValueError(f'Unknown scheduler: {name}')


def evaluate_test(model_name, model_path, batch_size):
    test_images, test_labels = load_mnist_test(flatten=False)
    model = nn.models.Model_CNN()
    model.load_model(model_path)

    total_acc = 0.0
    total_num = 0
    for start in range(0, test_images.shape[0], batch_size):
        end = min(start + batch_size, test_images.shape[0])
        logits = model(test_images[start:end])
        batch_num = end - start
        total_acc += nn.metric.accuracy(logits, test_labels[start:end]) * batch_num
        total_num += batch_num
    return total_acc / total_num


def finite_values(values):
    return [float(v) for v in values if not np.isnan(v)]


def save_runner_plot(runner, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot(runner, axes)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def run_recipe(name, recipe, args, train_set, dev_set, num_classes, out_dir):
    np.random.seed(args.seed)
    model = make_model(args, recipe, num_classes)
    optimizer = make_optimizer(model, recipe, args)
    num_batches = int(np.ceil(train_set[0].shape[0] / args.batch_size))
    scheduler = make_scheduler(optimizer, recipe, args, num_batches)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=num_classes)
    runner = nn.runner.RunnerM(
        model,
        optimizer,
        nn.metric.accuracy,
        loss_fn,
        batch_size=args.batch_size,
        scheduler=scheduler,
    )

    save_dir = out_dir / name
    runner.train(
        train_set,
        dev_set,
        num_epochs=args.epochs,
        log_iters=args.log_iters,
        eval_iters=args.eval_iters,
        eval_batch_size=args.eval_batch_size,
        save_dir=save_dir,
        early_stopping_patience=recipe['early_stopping'],
        early_stopping_min_delta=args.early_stopping_min_delta,
    )

    model_path = save_dir / 'best_model.pickle'
    test_acc = evaluate_test(name, model_path, args.eval_batch_size)
    curve_path = out_dir / f'{name}_learning_curve.png'
    save_runner_plot(runner, curve_path)

    metrics = {
        'name': name,
        'optimizer': recipe['optimizer'],
        'scheduler': recipe['scheduler'],
        'initial_lr': recipe['lr'],
        'final_lr': float(optimizer.init_lr),
        'weight_decay': recipe['weight_decay'],
        'dropout': recipe['dropout'],
        'early_stopping_patience': recipe['early_stopping'],
        'best_validation_accuracy': float(runner.best_score),
        'test_accuracy': float(test_acc),
        'num_updates': len(runner.train_loss),
        'final_train_loss': float(runner.train_loss[-1]),
        'final_train_accuracy': float(runner.train_scores[-1]),
        'final_dev_loss': finite_values(runner.dev_loss)[-1],
        'final_dev_accuracy': finite_values(runner.dev_scores)[-1],
        'curve_path': str(curve_path),
        'model_path': str(model_path),
    }

    with open(out_dir / f'{name}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics, runner


def save_comparison_plot(runners, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, runner in runners.items():
        x = np.arange(len(runner.train_loss))
        axes[0].plot(x, runner.train_loss, linewidth=1.2, label=name)
        dev_x = np.where(~np.isnan(runner.dev_loss))[0]
        dev_y = np.array(runner.dev_loss)[dev_x]
        axes[0].plot(dev_x, dev_y, linestyle='--', linewidth=1.0)

        axes[1].plot(x, runner.train_scores, linewidth=1.2, label=name)
        dev_x = np.where(~np.isnan(runner.dev_scores))[0]
        dev_y = np.array(runner.dev_scores)[dev_x]
        axes[1].plot(dev_x, dev_y, linestyle='--', linewidth=1.0)
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('loss')
    axes[1].set_xlabel('iteration')
    axes[1].set_ylabel('accuracy')
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout(w_pad=3.0)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Part C optimization + regularization experiments.')
    parser.add_argument('--recipes', type=str, default='sgd,momentum,sgd_step,adam,l2,dropout,recipe_combo')
    parser.add_argument('--seed', type=int, default=309)
    parser.add_argument('--valid-size', type=int, default=10000)
    parser.add_argument('--train-limit', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--log-iters', type=int, default=100)
    parser.add_argument('--eval-iters', type=int, default=100)
    parser.add_argument('--eval-batch-size', type=int, default=1000)
    parser.add_argument('--quick', action='store_true')

    parser.add_argument('--conv-channels', type=int, default=4)
    parser.add_argument('--second-conv-channels', type=int, default=8)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--pool-size', type=int, default=2)

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr-gamma', type=float, default=0.5)
    parser.add_argument('--exp-gamma', type=float, default=0.999)
    parser.add_argument('--step-size', type=int, default=None)
    parser.add_argument('--milestones', type=str, default=None)
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0)
    parser.add_argument('--out-dir', type=Path, default=BASE_DIR / 'part_c_results' / 'recipe')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 1
        args.train_limit = args.train_limit or 2000
        args.valid_size = min(args.valid_size, 1000)
        args.log_iters = min(args.log_iters, 20)
        args.eval_iters = min(args.eval_iters, 20)

    np.random.seed(args.seed)
    images, labels = load_mnist_train(flatten=False)
    train_images, train_labels, valid_images, valid_labels = split_train_valid(
        images,
        labels,
        valid_size=args.valid_size,
        seed=args.seed,
    )
    if args.train_limit is not None:
        train_images = train_images[:args.train_limit]
        train_labels = train_labels[:args.train_limit]

    num_classes = int(train_labels.max()) + 1
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = [name.strip() for name in args.recipes.split(',') if name.strip()]
    metrics = []
    runners = {}
    for name in selected:
        if name not in RECIPES:
            raise ValueError(f'Unknown recipe: {name}')
        print(f'Running recipe: {name}')
        item, runner = run_recipe(
            name,
            RECIPES[name],
            args,
            [train_images, train_labels],
            [valid_images, valid_labels],
            num_classes,
            out_dir,
        )
        metrics.append(item)
        runners[name] = runner

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    save_comparison_plot(runners, out_dir / 'recipe_comparison.png')

    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
