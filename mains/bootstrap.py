import time

import numpy as np
import pandas as pd
from sklearn.utils import resample

from logger import get_logger
from utils import msg_box, consuming_time


def bootstrapping(targets, outputs, metrics_epoch, test_metrics, repeat=1000):
    logger = get_logger("Bootstrapping")
    msg = msg_box("Bootstrap")
    msg += f"\nBootstrap for {repeat} times..."
    logger.info(msg)

    results = pd.DataFrame()
    test_metrics.reset()
    start = time.time()
    for number in range(repeat):
        ids = np.arange(len(outputs))
        sample_id = resample(ids)
        targets_ = targets[sample_id]
        outputs_ = outputs[sample_id]

        for met in metrics_epoch:
            test_metrics.epoch_update(met.__name__, met(targets_, outputs_))
        test_result = test_metrics.epoch_record
        test_result = test_result["mean"].rename(number)
        results = pd.concat((results, test_result), axis=1)

    msg = msg_box("result")

    end = time.time()
    total_time = consuming_time(start, end)
    msg += f"\nConsuming time: {total_time}."

    boot_result = pd.DataFrame()
    boot_result["CI_median"] = results.median(axis=1)
    boot_result["CI_low"] = results.quantile(q=0.025, axis=1)
    boot_result["CI_high"] = results.quantile(q=0.975, axis=1)
    boot_result["CI_half"] = (boot_result["CI_high"] - boot_result["CI_low"]) / 2
    msg += f"\n{boot_result}"

    logger.info(msg)
