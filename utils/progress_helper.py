from tqdm import tqdm

class ProgressHelper:
    def __init__(self, desc="处理进度", total=None, ncols=80, leave=True):
        self.desc = desc
        self.total = total
        self.ncols = ncols
        self.leave = leave

    def wrap(self, iterable):
        return tqdm(iterable, desc=self.desc, total=self.total, ncols=self.ncols, leave=self.leave)
