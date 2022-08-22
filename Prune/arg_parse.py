# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/21 21:09
# @Author : wzy 
# @File : arg_parse.py
# ---------------------------------------
import argparse


def parse_args():
    parse = argparse.ArgumentParser(description="The hyper-parameter of Prune")
    parse.add_argument('-b', '--bs', default=128)
    parse.add_argument('-l', '--lr', default=1e-3)
    parse.add_argument('-e', '--epoch', default=10)
    parse.add_argument('-cfg', '--config_list', default=[{
        'sparsity_per_layer': 0.8,
        'op_types': ['Conv2d']
    }])
    args = parse.parse_args()
    return args
