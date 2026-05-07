from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
        self.last_lr = optimizer.init_lr
    
    @abstractmethod
    def step():
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.last_lr = self.optimizer.init_lr
            self.step_count = 0

class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones=None, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.milestones = set(milestones or [])
        self.gamma = gamma
        self.global_step = 0

    def step(self) -> None:
        self.global_step += 1
        if self.global_step in self.milestones:
            self.optimizer.init_lr *= self.gamma
        self.last_lr = self.optimizer.init_lr

class ExponentialLR(scheduler):
    def __init__(self, optimizer, gamma=0.99) -> None:
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        self.optimizer.init_lr *= self.gamma
        self.last_lr = self.optimizer.init_lr
