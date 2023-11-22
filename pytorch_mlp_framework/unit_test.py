import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

import sys
sys.path.append('/Users/arthakhouri/Desktop/UoE/Machine Learning Practical/MLPC_2')
from pytorch_mlp_framework.model_architectures import *

class TestConvolutionBlockBN(unittest.TestCase):
    def test_forward(self):
        
        block = ConvolutionBlockBN(input_shape, num_filters, kernel_size, padding, bias, dilation)
        x = torch.randn(input_shape)
        output = block.forward(x)

        self.assertEqual(output.shape, x.shape)

class TestConvolutionalReductionBlockBN(unittest.TestCase):
    def test_forward(self):
        
        block = ConvolutionalReductionBlockBN(input_shape, num_filters, kernel_size, padding, bias, dilation, reduction_factor)
        x = torch.randn(input_shape)
        output = block.forward(x)

        self.assertEqual(output.shape[2:], tuple(s // reduction_factor for s in x.shape[2:]))

class TestConvolutionBlockBNRC(unittest.TestCase):
    def test_forward(self):

        block = ConvolutionBlockBNRC(input_shape, num_filters, kernel_size, padding, bias, dilation)
        x = torch.randn(input_shape)
        output = block.forward(x)

        self.assertEqual(output.shape, x.shape)

if __name__ == '__main__':
    input_shape = (1, 3, 32, 32)
    num_filters = 16
    kernel_size = 3
    padding = 1
    bias = False
    dilation = 1
    reduction_factor = 2
    unittest.main()