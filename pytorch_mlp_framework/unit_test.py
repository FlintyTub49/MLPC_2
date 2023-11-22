import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

import sys
sys.path.append('/Users/arthakhouri/Desktop/UoE/Machine Learning Practical/MLPC_2')
from pytorch_mlp_framework.model_architectures import *

class ConvolutionBlocksTests(unittest.TestCase):
    def test_convolution_block_bn(self):
        """
        Test the ConvolutionBlockBN Class by checking if the output shape is correct
        """
        block = ConvolutionBlockBN(input_shape, num_filters, kernel_size, padding, bias, dilation)
        input_tensor = torch.randn(input_shape)

        output = block.forward(input_tensor)
        self.assertEqual(output.shape, torch.Size([1, num_filters, 32, 32]))

    def test_convolutional_reduction_block_bn(self):
        """
        Test the ConvolutionalReductionBlockBN Class by checking if the output shape is correct
        """
        block = ConvolutionalReductionBlockBN(input_shape, num_filters, kernel_size, padding, bias, dilation, reduction_factor)
        input_tensor = torch.randn(input_shape)

        output = block.forward(input_tensor)
        self.assertEqual(output.shape, torch.Size([1, num_filters, 16, 16]))

    def test_convolution_block_bn_rc(self):
        """
        Test the ConvolutionBlockBNRC Class by checking if the output shape is correct
        """
        input_shape = (1, num_filters, 32, 32)
        block = ConvolutionBlockBNRC(input_shape, num_filters, kernel_size, padding, bias, dilation)
        input_tensor = torch.randn(input_shape)

        output = block.forward(input_tensor)
        self.assertEqual(output.shape, torch.Size([1, num_filters, 32, 32]))

if __name__ == '__main__':    
    input_shape = (1, 3, 32, 32)
    num_filters, kernel_size = 32, 3
    padding, dilation = 1, 1
    bias = False
    reduction_factor = 2

    unittest.main()