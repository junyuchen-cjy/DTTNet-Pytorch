import os
import re
import traceback
import pandas as pd
import wandb
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

class AbstractLoggerQuery:
    def get_value_steps(self, tag):
        raise NotImplementedError

    # def steps_to_epoch(self, steps):
    #     raise NotImplementedError

    def get_best(self, tag, method="min"):
        v, s = self.get_value_steps(tag)
        indexes = [i for i in range(len(s))]
        # sort the indexes by v
        if method == "min":
            indexes.sort(key=lambda x: v[x])
        else:
            indexes.sort(key=lambda x: -v[x])
        # get the top 5 indexes
        top5_idx = indexes[:3]
        # the the values
        top5_values = [v[i] for i in top5_idx]
        top5_steps = [s[i] for i in top5_idx]
        return top5_idx, top5_values, top5_steps

    def get_best_epochs(self, tag, method="min"):
        v, e = self.get_value_epochs(tag)
        indexes = [i for i in range(len(e))]
        # sort the indexes by v
        if method == "min":
            indexes.sort(key=lambda x: v[x])
        else:
            indexes.sort(key=lambda x: -v[x])
        # get the top 5 indexes
        top5_idx = indexes[:3]
        # the the values
        top5_values = [v[i] for i in top5_idx]
        top5_epochs = [e[i] for i in top5_idx]
        return top5_idx, top5_values, top5_epochs

    def get_last_ep(self, tag):
        v, s = self.get_value_steps(tag)
        return len(s) - 1

    def _bsearch(self, arr, target_step):
        l = 0
        r = len(arr) - 1
        while l < r:
            mid = (l + r + 1) // 2
            if arr[mid] <= target_step:
                l = mid
            else:
                r = mid - 1
        return l

    def steps_to_epoch(self, steps):
        ep_ep, ep_step = self.get_value_steps('epoch')
        mapped = []
        for vs in steps:
            idx = self._bsearch(ep_step, vs)
            mapped.append(ep_ep[idx])
        return mapped

    def get_value_epochs(self, tag):
        raise NotImplementedError


class TFLoggerQuery(AbstractLoggerQuery):
    def __init__(self, log_dir):
        self.event_acc = EventAccumulator(log_dir)
        self.event_acc.Reload()

    def get_value_steps(self, tag):
        event_list = self.event_acc.Scalars(tag)
        values = list(map(lambda x: x.value, event_list))
        step = list(map(lambda x: x.step, event_list))
        return values, step

    # def steps_to_epoch(self, steps):
    #     v, s = self.get_value_steps('epoch')
    #     # zip the two list
    #     return [int(v[s.index(step)]) for step in steps]

    def get_tags(self):
        return self.event_acc.Tags()["scalars"]


class WandbLoggerQuery(AbstractLoggerQuery):
    def __init__(self, api_key, path):
        self.api_key = api_key
        self.path = path

    def _get_run(self):
        return wandb.Api(api_key=self.api_key).run(self.path)

    def get_exp_name(self):
        run = self._get_run()
        return run.name

    def get_value_steps(self, tag):
        run = self._get_run()
        # run.history returns sampled data, do not use it
        # history = run.history(keys=[tag])
        history = run.scan_history(keys=[tag, "_step"])
        losses = [row for row in history]
        df = pd.DataFrame(losses)
        df = df.sort_values(by="_step")
        values = df[tag].values
        step = df["_step"].values
        return values, step

    def get_value_epochs(self, tag, sort_by="epoch"):
        run = self._get_run()
        history = run.scan_history(keys=[tag, "epoch"])
        losses = [row for row in history]
        df = pd.DataFrame(losses)
        df = df.sort_values(by=sort_by)
        values = df[tag].values
        epoch = df["epoch"].values
        return values, epoch



    def steps_to_runtime(self, steps):
        rt_ep, rt_step = self.get_value_steps('_runtime')
        mapped = []
        for vs in steps:
            idx = self._bsearch(rt_step, vs)
            secs = rt_ep[idx]
            # seconds to days hours minutes seconds
            m, s = divmod(secs, 60)
            h, m = divmod(m, 60)
            d, h = divmod(h, 24)
            mapped.append(f"{d:.0f}d {h:.0f}h {m:.0f}m {s:.0f}s")
        return mapped

    def epochs_to_runtime(self, epochs):
        rt_ep, ep_lst = self.get_value_epochs('_runtime')
        mapped = []
        for vs in epochs:
            idx = self._bsearch(ep_lst, vs)
            secs = rt_ep[idx]
            # seconds to days hours minutes seconds
            m, s = divmod(secs, 60)
            h, m = divmod(m, 60)
            d, h = divmod(h, 24)
            mapped.append(f"{d:.0f}d {h:.0f}h {m:.0f}m {s:.0f}s")
        return mapped

    def _secs_to_string(self, secs):
        # seconds to days hours minutes seconds
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        return f"{d:.0f}d {h:.0f}h {m:.0f}m {s:.0f}s"

    def avg_runtime_per_ep(self):
        epochs = [i for i in range(1, 100)]
        rt_ep, ep_lst = self.get_value_epochs('_runtime', sort_by="_runtime") # 坑：这里必须要按照runtime排序，因为epochs有很多相同的值
        mapped = []
        for vs in epochs:
            idx = self._bsearch(ep_lst, vs)
            secs = rt_ep[idx]
            mapped.append(secs)
        new = []
        for i in range(1, len(mapped)):
            new.append(mapped[i] - mapped[i-1])
        meansecs = np.mean(new)
        print(new)
        return self._secs_to_string(meansecs)


if __name__ == "__main__":
    tag = "val/usdr"
    api_key = "xxxxxxxxxxxx"
    path = "username/dtt_vocals/xxxxxxx"

    w_query = WandbLoggerQuery(api_key=api_key, path=path)
    best_sdr_idx, best_sdr_values, best_sdr_epochs = w_query.get_best_epochs(tag, method="max")
    print("idx", best_sdr_idx)
    print("values", best_sdr_values)
    print("epochs", best_sdr_epochs)
    print("runtime", w_query.epochs_to_runtime(best_sdr_epochs))