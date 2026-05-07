import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.eval_batch_size = None

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []
        self.lr_history = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        eval_iters = kwargs.get("eval_iters", log_iters)
        self.eval_batch_size = kwargs.get("eval_batch_size", self.eval_batch_size)
        save_dir = kwargs.get("save_dir", "best_model")
        early_stopping_patience = kwargs.get("early_stopping_patience", None)
        early_stopping_min_delta = kwargs.get("early_stopping_min_delta", 0.0)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        best_score = 0
        global_iter = 0
        no_improve_count = 0
        should_stop = False

        for epoch in range(num_epochs):
            if hasattr(self.model, 'train'):
                self.model.train()
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            num_batches = int(np.ceil(X.shape[0] / self.batch_size))
            for iteration in range(num_batches):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)
                
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.lr_history.append(self.optimizer.init_lr)

                should_eval = (
                    global_iter % eval_iters == 0
                    or iteration == num_batches - 1
                )
                if should_eval:
                    dev_score, dev_loss = self.evaluate(dev_set)
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)

                    if dev_score > best_score + early_stopping_min_delta:
                        save_path = os.path.join(save_dir, 'best_model.pickle')
                        self.save_model(save_path)
                        print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                        best_score = dev_score
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        if (
                            early_stopping_patience is not None
                            and no_improve_count >= early_stopping_patience
                        ):
                            should_stop = True
                else:
                    self.dev_scores.append(np.nan)
                    self.dev_loss.append(np.nan)

                if (global_iter) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    if should_eval:
                        print(f"[Dev] loss: {dev_loss}, score: {dev_score}")

                global_iter += 1
                if should_stop:
                    print(f"early stopping at epoch {epoch}, iteration {iteration}")
                    break
            if should_stop:
                break
        self.best_score = best_score

    def evaluate(self, data_set):
        was_training = getattr(self.model, 'training', True)
        if hasattr(self.model, 'eval'):
            self.model.eval()
        X, y = data_set
        if self.eval_batch_size is not None and X.shape[0] > self.eval_batch_size:
            total_loss = 0.0
            total_score = 0.0
            total_num = 0
            for start in range(0, X.shape[0], self.eval_batch_size):
                end = min(start + self.eval_batch_size, X.shape[0])
                batch_X = X[start:end]
                batch_y = y[start:end]
                logits = self.model(batch_X)
                batch_loss = self.loss_fn(logits, batch_y)
                batch_score = self.metric(logits, batch_y)
                batch_num = end - start
                total_loss += batch_loss * batch_num
                total_score += batch_score * batch_num
                total_num += batch_num
            result = total_score / total_num, total_loss / total_num
            if was_training and hasattr(self.model, 'train'):
                self.model.train()
            return result

        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        if was_training and hasattr(self.model, 'train'):
            self.model.train()
        return score, loss
    
    def save_model(self, save_path):
        self.model.save_model(save_path)
