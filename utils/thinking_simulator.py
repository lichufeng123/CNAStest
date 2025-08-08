from tqdm import tqdm
import time
import random

class ThinkingSimulator:
    def __init__(self, min_seconds=1, max_seconds=2, desc="模拟思考中...", steps=100):
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.desc = desc
        self.steps = steps

    def run(self):
        total_time = random.uniform(self.min_seconds, self.max_seconds)
        delay_per_step = total_time / self.steps

        for _ in tqdm(range(self.steps), desc=self.desc, ncols=100):
            time.sleep(delay_per_step)
