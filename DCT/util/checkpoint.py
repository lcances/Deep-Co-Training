from torch.utils.tensorboard import SummaryWriter
import torch
import os


class CheckPoint:
    def __init__(self, model: list, optimizer,
                 mode: str = "max", name: str = "best",
                 verbose: bool = True):
        self.mode = mode

        self.name = name
        self.verbose = verbose

        self.model = model
        self.optimizer = optimizer
        self.best_state = dict()
        self.last_state = dict()
        self.best_metric = None
        self.epoch_counter = 0

        # Preparation
        if not isinstance(self.model, list):
            self.model = [self.model]

        self.create_directory()

    def create_directory(self):
        os.makedirs(os.path.dirname(self.name), exist_ok=True)

    def step(self, new_value):
        if self.epoch_counter == 0:
            self.best_metric = new_value

        # Save last epoch
        self.last_state = self._get_state(new_value)
        torch.save(self.last_state, self.name + ".last")

        # save best epoch
        if self._check_is_better(new_value):
            if self.verbose:
                print("\n better performance: saving ...")

            self.best_metric = new_value
            self.best_state = self._get_state(new_value)
            torch.save(self.best_state, self.name)

        self.epoch_counter += 1

    def _get_state(self, new_value=None) -> dict:
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
        if not os.path.isfile(self.name):
            return

        data = torch.load(self.name)
        self._load_helper(data, self.best_state)

    def load_last(self):
        if not os.path.isfile(self.name + ".last"):
            return

        data = torch.load(self.name + ".last")
        self._load_helper(data, self.last_state)

    def _load_helper(self, state, destination):
        print(list(state.keys()))
        for k, v in state.items():
            destination[k] = v

        self.optimizer.load_state_dict(destination["optimizer"])
        self.epoch_counter = destination["epoch"]
        self.best_metric = destination["best_metric"]

        # Path to fit with previous version of checkpoint
        if not isinstance(destination["state_dict"], list):
            destination["state_dict"] = [destination["state_dict"]]

        for i in range(len(self.model)):
            self.model[i].load_state_dict(destination["state_dict"][i])

    def _check_is_better(self, new_value):
        assert len(self.best_metric.shape) == len(new_value.shape)

        # The case of 0-d tensor
        if len(self.best_metric.shape) == 0:
            if self.model == "max":
                return self.best_metric < new_value
            return self.best_metric > new_value

        # Multi-dimension tensor
        if self.mode == "max":
            return any(new_value > self.best_metric)

        return any(self.best_metric > new_value)

    
class mSummaryWriter(SummaryWriter):
    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix=''):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        self.history = dict()
        
    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        super().add_scalar(tag, scalar_value, global_step, walltime)
        
        if tag not in self.history:
            self.history[tag] = [scalar_value]
        else:
            self.history[tag].append(scalar_value)