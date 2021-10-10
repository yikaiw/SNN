import torch
import torch.nn as nn
from torch.autograd import Function
from .patterns import *


class Binarize(nn.Module):
    def __init__(self, bit_num):
        super(Binarize, self).__init__()
        self.bit_num = bit_num
        self.step = 0
        self.patterns = nn.Parameter(torch.ones([2 ** bit_num, 9], requires_grad=True))
        self.patterns.data = get_random_patterns(bit_num)
        self.memory = torch.sign(self.patterns)
        self.register_parameter('patterns', self.patterns)

    def check_conpensate_patterns(self, weights):
        bnum_set = set(patterns2bnum(self.patterns.data))
        if len(bnum_set) < 2 ** self.bit_num:
            patterns = remove_repetitive_patterns(self.patterns.data, self.bit_num)
            self.patterns.data = conpensate_patterns(weights, patterns, bnum_set, self.bit_num)
            new_bnum_set = set(patterns2bnum(self.patterns.data))
            print('conpensate_patterns:\n', bnum_set, '(%d)' % len(bnum_set), '->\n', \
                 new_bnum_set, '(%d)' % len(new_bnum_set))
        self.patterns.data = torch.clamp(self.patterns.data, -3.0, 3.0)

    def forward(self, weights):  
        # weights: [cin, cout, 3, 3], patterns: [2 ** bit_num, 9], memory: [2 ** bit_num, 9]
        i = (torch.abs(self.patterns) >= 1e-3).to(torch.float32)
        self.memory = torch.sign(self.patterns) * i + self.memory.to(i.device) * (1 - i)
        binary_weights, idxs = weights2patterns(weights.view(-1, 9), self.memory.to(weights.device))
        binary_weights = binary_weights.view(weights.shape)
        cliped_weights = torch.clamp(weights, -1.0, 1.0)

        self.step += 1
        if self.step % 25 == 0:
            idxs = one_hot(idxs, n_class=self.patterns.shape[0])  # [cin * cout, 2 ** bit_num]
            selected_patterns = torch.matmul(idxs, self.patterns).view(weights.shape)
            binary_weights = binary_weights.detach() - cliped_weights.detach() + cliped_weights \
                - selected_patterns.detach() + selected_patterns
            self.check_conpensate_patterns(weights)
        else:
            binary_weights = binary_weights.detach() - cliped_weights.detach() + cliped_weights \
        return binary_weights
