"""
To dump and load data
"""
import glob
import logging
import os
import pickle

import numpy as np
import pandas as pd
from tick.base import TimeFunction

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class DataBox:
    """
    Methodes:
    --------
        load_pickles(self, filename=None, keyword=None)
            load pickles from self.path
            one of filename and keyword should be given
        save_pickles(self, obj, filename)
            save "obj" as "filename" to self.path

    """

    def __init__(self, path):
        """
        parameter:
        ----------
            path
        """
        self.path = path

    def load_pickles(self, filename=None, keyword=None):
        if filename is None and keyword is None:
            raise ValueError("")
        if filename is None:
            filenames = glob.glob(os.path.join(self.path, keyword))
            if len(filenames) > 1:
                logging.info("possible files %s" % filenames)
                return None
            else:
                filename = filenames[0]
                logging.info(filename)
        try:
            f = open(os.path.join(self.path, filename), "rb")
            file = pickle.load(f)
            f.close()
        except:
            file = pd.read_pickle(filename)
        return file

    def save_pickles(self, obj, filename):
        f = open(os.path.join(self.path, filename), "wb")
        pickle.dump(obj, f)
        f.close()


def print_info(n_iter: int, print_every: int, obj: float, rel_obj: float):
    """print information during training

    Args:
        n_iter (int): number of iterations
        print_every (int): print every [print_every] iterations
        obj (float): loss at current iteration
        rel_obj (float): relative loss at current iteration
                        (obj-prev_obj)/(prev_obj)

    Returns:
        string: message to print
    """
    if n_iter == 0:
        msg = "     n_iter  |  objective  |    rel_obj  \n"
        msg += "-----------------------------------------\n"
    else:
        msg = ""

    if n_iter % print_every == 0:
        msg += (
            str(n_iter).rjust(11)
            + "  |"
            + str(np.format_float_scientific(obj, exp_digits=2, precision=2)).rjust(11)
            + "  |"
            + str(np.format_float_scientific(rel_obj, exp_digits=2, precision=2)).rjust(
                11
            )
        )
        print(msg)
        return msg


def timefunc(ts_list: list, dt: float, max_time: float = None):
    """_summary_

    Args:
        ts_list (list): each element in the list is a list of timestamps, representing event times of a single node
        dt (float): subsampling interval

    Returns:
        list: list of TimeFunction objects
    """
    timefuncs = []
    if max_time is None:
        max_time = max([ts[-1] for ts in ts_list]) + dt

    for ts in ts_list:
        ts = np.insert(ts, 0, 0)
        ts = np.insert(ts, len(ts), max_time + dt)

        df = pd.DataFrame({"time": ts, "cum": range(0, len(ts))})
        timef = TimeFunction(
            (df.time.values, df.cum.values),
            dt=dt,
            inter_mode=TimeFunction.InterConstRight,
        )
        timefuncs.append(timef)
    return timefuncs
