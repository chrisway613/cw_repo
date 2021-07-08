"""封装一个类，这个类可将一些状态量存储到内存或者输出到文件"""


import os
import torch

from copy import deepcopy


class StateCacher:
    def __init__(self, in_memory=True, cache_dir=None):
        self.cache = {}
        self.in_memory = in_memory
        self.cache_dir = cache_dir

        if not in_memory:
            if cache_dir is None:
                import tempfile

                self.cache_dir = tempfile.gettempdir()
            else:
                assert os.path.isdir(cache_dir), \
                    f"Given 'cache_dir' {cache_dir} is not a valid directory, please check it out!"

    def store(self, key, state_dict):
        if self.in_memory:
            self.cache.update({key: deepcopy(state_dict)})
        else:
            fn = os.path.join(self.cache_dir, f"state_{key}_{id}.pt")
            torch.save(state_dict, fn)
            # or use pickle
            # import pickle
            # pickle.dump(state_dict, open(fn, 'wb'))
            self.cache.update({key: fn})

    def retrieve(self, key):
        assert self.cache.get(key) is not None, f"Target {key} was not cached, please check it out!"

        if self.in_memory:
            return self.cache[key]
        else:
            fn = self.cache[key]
            if not os.path.isfile(fn):
                raise RuntimeError(f"Failed to load state in {fn}, file doesn't exist!")

            state_dict = torch.load(fn, map_location=lambda storage, location: storage)
            # or use pickle
            # import pickle
            # state_dict = pickle.load(open(fn, 'rb'))

            return state_dict

    def __del__(self):
        """Check whether there are unused cached files existing in `cache_dir`
        before this instance being destroyed."""

        if not self.in_memory:
            for k in self.cache:
                path = self.cache[k]
                if os.path.exists(path):
                    os.remove(path)
