import torch
import os


class CheckPoint:
    def __init__(self, model, optimizer, mode: str="max", name: str="best", verbose: bool=True):
        self.mode = mode
        self.name = name
        self.verbose = verbose

        self.model = model
        self.optimizer = optimizer

        self.best_state = dict()
        self.last_state = dict()
        self.best_metric = 0 if mode == "max" else 100000
        self.epoch_counter = 0
        
        self.create_directory()
        
    def create_directory(self):
        os.makedirs(os.path.dirname(self.name), exist_ok=True)

    def step(self, new_value):
        self.epoch_counter += 1
        
        # Save last epoch
        self.last_state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_metric": new_value,
            "epoch": self.epoch_counter,
        }
        torch.save(self.last_state, self.name + ".last")

        if self._check_is_better(new_value):
            if self.verbose:
                print("\n better performance: saving ...")

            self.best_metric = new_value
            self.best_state = {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_metric": self.best_metric,
                "epoch": self.epoch_counter,
            }
            torch.save(self.best_state, self.name)
            
    def load_best(self):
        data = torch.load(self.name)
        
        for k, v in data.items():
            self.best_state[k] = v
            
        self.model.load_state_dict(self.best_state["state_dict"])
        self.optimizer.load_state_dict(self.best_state["optimizer"])
        self.epoch_counter = self.best_state["epoch"]
            
    def load_last(self):
        data = torch.load(self.name + ".last")
        
        for k, v in data.items():
            self.last_state[k] = v
            
        self.model.load_state_dict(self.last_state["state_dict"])
        self.optimizer.load_state_dict(self.last_state["optimizer"])
        self.epoch_counter = self.last_state["epoch"]

    def _check_is_better(self, new_value):
        if self.mode == "max":
            if self.best_metric <= new_value:
                return True
            return False

        if self.best_metric <= new_value:
            return False
        return True
