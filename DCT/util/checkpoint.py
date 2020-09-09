from typing import Iterable
import torch
import os

class CheckPoint:
    def __init__(self, model: list, optimizer, mode: str="max", name: str="best", verbose: bool=True):
        self.mode = mode
        self.name = name
        self.verbose = verbose

        self.model = model
        self.optimizer = optimizer

        self.best_state = dict()
        self.last_state = dict()
        self.best_metric = 0 if mode == "max" else 100000
        self.epoch_counter = 0

        # Preparation
        if not isinstance(self.model, list):
            self.model = [self.model]

        self.create_directory()
        
    def create_directory(self):
        os.makedirs(os.path.dirname(self.name), exist_ok=True)

    def step(self, new_value):
        self.epoch_counter += 1
        
        # Save last epoch
        self.last_state = self._get_state(new_value)
        torch.save(self.last_state, self.name + ".last")

        if self._check_is_better(new_value):
            if self.verbose:
                print("\n better performance: saving ...")

            self.best_metric = new_value
            self.best_state = self._get_state(new_value)
            torch.save(self.best_state, self.name)
            
    def _get_state(self, new_value = None) -> dict:
        state = {
            "state_dict": [m.state_dict() for m in self.model],
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch_counter,
        }
        if new_value is not None:
            state["best_metric"] = new_value
            
        return state

    def save(self):
        torch.save(self._get_state, self.name + ".last")
        
    def load_best(self):
        if not os.path.isfile(self.name + ".last"):
            return

        data = torch.load(self.name)
        self._load_helper(data, self.best_state)
            
    def load_last(self):
        if not os.path.isfile(self.name + ".last"):
            return

        data = torch.load(self.name + ".last")
        self._load_helper(data, self.last_state)
        
    def _load_helper(self, state, destination):
        for k, v in data.items():
            destination = v
            
        self.optimizer.load_state_dict(self.best_state["optimizer"])
        self.epoch_counter = self.best_state["epoch"]
        self.best_metric = self.best_state["best_metric"]

        for i in range(len(self.model)):
            self.model[i].load_state_dict(self.best_state["state_dict"][i])

    def _check_is_better(self, new_value):
        if not isinstance(new_value, Iterable):
            new_value = [new_value]
            
        if not isinstance(self.best_metric, Iterable):
            self.best_metric = [self.best_metric]
            
        tester = lambda x, y: x > y
        if self.mode == "max":
             tester = lambda x, y: y > x
        
        return any(map(tester, self.best_metric, new_value))
