# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/21 20:37 
# @Author : wzy 
# @File : prune.py
# ---------------------------------------
import torch
from nni.compression.pytorch.pruning import L1NormPruner
from nni.compression.pytorch.speedup import ModelSpeedup
from arg_parse import parse_args
from torchinfo import summary
args = parse_args()


def pruning(model, device, config_list):
    print('==========================before pruning==========================')
    print(model)
    summary(model, (1,3,32,32))
    print('==================================================================')
    # 使用库中剪枝方法(此时只是包装结构，还未进行剪枝)
    pruner = L1NormPruner(model, config_list)
    # print(model)
    # 压缩模型并进行掩码
    model, masks = pruner.compress()
    # 展示每层结构的掩码稀疏度(此时还没尽兴剪枝)
    for name, mask in masks.items():
        print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))

    # 解开模型(拿到里面的单个结构)
    pruner._unwrap_model()
    # 真正的模型剪枝加速
    ModelSpeedup(model, torch.rand(1, 3, 32, 32).to(device), masks).speedup_model()

    print('==========================after pruning==========================')
    print(model)
    summary(model, (1,3,32,32))
    print('=================================================================')
    # pruner.export_model(model_path="./pts/prune.pth", mask_path="./pts/mask.pth")
    return model, masks


def pruning_main(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_prune, masks = pruning(model, device, args.config_list)
    return model_prune, masks
