"""Dump Dict List to File"""
import json
import pandas as pd

from dexpression_pytorch import settings


def dump_dict_list(history):
    df = pd.DataFrame(history)
    df.to_csv(settings.results("history.csv"))
