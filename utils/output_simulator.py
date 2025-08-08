class OutputSimulator:
    @staticmethod
    def show(desc="输出中...", duration=3.0, steps=40):
        from tqdm import tqdm
        import time
        for _ in tqdm(range(steps), desc=desc, ncols=80):
            time.sleep(duration / steps)
