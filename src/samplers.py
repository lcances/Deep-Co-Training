from torch.utils.data.sampler import Sampler
import math
import random
import numpy as np


class CoTrainingSampler(Sampler):
    METHODS = ["duplicate", "truncate", "random_truncate", "random_select"]
    
    def __init__(self, dataset, batch_size, nb_class: int = 10,
                 shuffle: bool = True, nb_view: int = 2, ratio: float = None, method: str = "duplicate"):
        """
        The CoTrainingSampler is used to get the index that gonna be used a batch for every views.
        It will output batch_indexes as follow:
        2 views --> (S1, S2, U)
        4 views --> (S1, S2, S3, S4, U)
        ...

        Where Sx will be of size SB and U of size UB

        :param batch_size: The total size of a batch
        :param supervised_size: The number of supervised files
        :param unsupervised_size: The number of unsupervised files
        :param nb_view: The number of views
        :param ratio:   The ratio of Supervised / All to apply. If non, a perfect ratio is computed
        :param method:  The methods used to acheive the requested ratio. "duplicate" will duplicate
                        supervised files, "truncate" will remove unsupervised files and,
                        "random_truncate" will remove random unsupervised files.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.supervised_size = len(dataset.y_S)
        self.unsupervised_size = len(dataset.y_U)
        self.nb_view = nb_view
        self.nb_class = nb_class
        self.shuffle = shuffle
        self.ratio = ratio if ratio is not None else self.supervised_size / (self.supervised_size + self.unsupervised_size)
        self.method = method
        
        # will hold the indexes for supervised and unsupervised sample.
        self.S_idx = []
        self.U_idx = []
        
        self._check_param()

        # Create list of index for S and U
        self._create_idx()
        
        # Create view
        self.views = [self.S_idx.copy() for _ in range(self.nb_view)]
        [random.shuffle(self.views[idx]) for idx in range(self.nb_view)]
        
        # Calc the supervised and unsupervised batch size
        if self.unsupervised_size != 0:

            # split ratio, keep total nb of file = batch_size
            self.nb_batch = int(math.floor((len(self.S_idx) + len(self.U_idx)) / batch_size))
            self.S_batch_size = math.floor(self.batch_size * self.ratio)
            self.U_batch_size = math.floor(self.batch_size * (1 - self.ratio))

        else:
            self.ratio = 1
            self.nb_batch = int(math.floor(len(self.S_idx) / batch_size))
            self.S_batch_size = self.batch_size
            self.U_batch_size = 0
        
    def _check_param(self):
        if self.method not in CoTrainingSampler.METHODS:
            raise ValueError("method %s unknown, only %s available" % (self.method, CoTrainingSampler.METHODS))

    def _create_idx(self):
        def duplicate_supervised():
            # calc the number of time the supervised part should be copied to keep the ratio
            if self.unsupervised_size != 0:
                nb_S = int((self.ratio * self.unsupervised_size) / (1 - self.ratio))

                if len(self.S_idx) < nb_S:
                    S_idx = self.S_idx.copy()
                    while len(self.S_idx) < nb_S:
                        self.S_idx += S_idx
                self.S_idx = self.S_idx[:nb_S]
        
        def truncate_unsupervised():
            # Calc the number of unsupervised file that must be kept to fulfill the ratio
            nb_U = int((self.supervised_size * (1 - self.ratio)) / self.ratio)
            self.U_idx = self.U_idx[:nb_U]
            
        def random_truncate_unsupervised():
            random.shuffle(self.U_idx)
            truncate_unsupervised()

        def random_select():
            nb_U = int((self.supervised_size * (1 - self.ratio)) / self.ratio)
            self.U_idx = list(range(self.unsupervised_size))
            self.U_idx = np.random.choice(self.U_idx, nb_U)

        def round_to_multiple():
            """Adjust the S and U size to be a multiple of number of class (below)"""
            valid_nb_S = len(self.S_idx) - (len(self.S_idx) % self.nb_class)
            valid_nb_U = len(self.U_idx) - (len(self.U_idx) % self.nb_class)

            self.S_idx = self.S_idx[:valid_nb_S]
            self.U_idx = self.U_idx[:valid_nb_U]
        
        if self.ratio == 1: self.ratio -= 0.001
        if self.ratio == 0: self.ratio += 0.001
            
        self.S_idx = list(range(self.supervised_size))
        self.U_idx = list(range(self.unsupervised_size))
        
        # mitigation methods
        if self.method == CoTrainingSampler.METHODS[0]: duplicate_supervised()
        if self.method == CoTrainingSampler.METHODS[1]: truncate_unsupervised()
        if self.method == CoTrainingSampler.METHODS[2]: random_truncate_unsupervised()
        if self.method == CoTrainingSampler.METHODS[3]: random_select()

        round_to_multiple()

        # Shuffle the indexes
        if self.shuffle:
            random.shuffle(self.S_idx)
            random.shuffle(self.U_idx)

    def __iter__(self):
        for b in range(self.nb_batch - 1):
            batch = []

            for v_idx in self.views:
                v_start = b * self.S_batch_size
                v_end = (b+1) * self.S_batch_size

                batch.append(v_idx[v_start:v_end])

            U_start = b * self.U_batch_size
            U_end = (b+1) * self.U_batch_size
            U_batch = self.U_idx[U_start:U_end]
            batch.append(U_batch)

            # If a list of list, Pytorch will take only the first element. work around is to send a 1-element tuple
            yield batch,

    def __len__(self):
        return self.nb_batch


if __name__ == '__main__':
    # s = CoTrainingSampler(64, 1478, 14852, ratio=None)
    s = CoTrainingSampler(32, 873, 8732-87, ratio=None)

    for i in s:
        print(i)
        print(s.S_batch_size)
        print(s.U_batch_size)
        break
