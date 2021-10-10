import torch
import torch.nn as nn
import numpy as np
import random

str2kernel = lambda s: torch.tensor(list(map(int, s)), dtype=torch.float32) * 2 - 1
bnum_base = torch.tensor([2 ** i for i in range(9)]).to(torch.int32).unsqueeze(0)

patterns_str = [('%9s' % bin(i)[2:]).replace(' ', '0') for i in range(512)]
patterns_all = str2kernel(patterns_str[0]).unsqueeze(0)
for i in range(1, len(patterns_str)):
    patterns_all = torch.cat([patterns_all, str2kernel(patterns_str[i]).unsqueeze(0)], dim=0)  # [512, 9]


def patterns2bnum(k):
    k = ((torch.sign(k) + 1) / 2).view(-1, 9).to(torch.int32).cpu()
    bnum = torch.sum(k * bnum_base, dim=1).cpu().numpy()
    return bnum


def one_hot(labels, n_class):  # labels: [n_num] in [0, n_class)
    return torch.zeros(labels.shape[0], n_class).to(labels.device).\
        scatter(1, labels.unsqueeze(1), 1)


def get_sorted_patterns(weights, bit_num):
    _, idxs = weights2patterns(weights.view(-1, 9), patterns_all.to(weights.device))
    idxs = list(idxs.data.cpu().numpy())
    counts = [(i, idxs.count(i)) for i in range(512)]
    counts.sort(key=lambda x: (x[1], x[0]), reverse=True)
    counts = list(map(lambda x: x[0], counts[:2 ** bit_num]))
    patterns = patterns_all[counts]  # [2 ** bit_num, 9]
    return patterns


def get_random_patterns(bit_num):
    idxs = random.sample(list(range(512)), 2 ** bit_num)
    patterns = patterns_all[idxs]  # [2 ** bit_num, 9]
    return patterns


def weights2patterns(weights, patterns):
    # weights: [cin * cout, 9], patterns: [2 ** bit_num, 9]
    norm = torch.norm(weights.unsqueeze(1) - patterns.unsqueeze(0), dim=2)  # [cin * cout, 2 ** bit_num]
    idxs = norm.argmin(dim=1)  # [cin * cout]
    return patterns[idxs], idxs  # [cin * cout, 9], [cin * cout]


def remove_repetitive_patterns(patterns, bit_num):
    tmp_set, new_patterns = set(), None
    for i in range(2 ** bit_num):
        pattern = patterns[i].unsqueeze(0)
        bnum = patterns2bnum(pattern)[0]
        if bnum not in tmp_set:
            tmp_set.add(bnum)
            new_patterns = pattern if i == 0 else torch.cat([new_patterns, pattern], dim=0)
    return new_patterns


def conpensate_patterns(weights, patterns, bnum_set, bit_num):
    add_num = 0
    #patterns_resample = get_sorted_patterns(weights, bit_num).to(weights.device).detach()
    patterns_resample = get_random_patterns(bit_num).to(weights.device).detach()
    for i in range(patterns_resample.shape[0]):
        pattern = patterns_resample[i].unsqueeze(0)
        bnum = patterns2bnum(pattern)[0]
        if bnum not in bnum_set:
            patterns = torch.cat([patterns, pattern], dim=0)
            add_num += 1
        if len(bnum_set) + add_num == 2 ** bit_num:
            break
    return patterns
