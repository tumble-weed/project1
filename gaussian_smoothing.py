import torch
import numpy as np
import math
from torch import nn
def _gaussian_kernel(channels=3,kernel_size=15,sigma = 3,padding='symmetric'):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    
    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                        torch.exp(
                            -torch.sum((xy_grid - mean)**2., dim=-1) /\
                            (2*variance)
                        )
    # print(colored('chacnging gaussian kernel to mean','red'));gaussian_kernel = torch.ones_like(gaussian_kernel)

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    
    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)
    
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    if padding == 'symmetric':
        padding = torch.nn.ReflectionPad2d(kernel_size//2)
    else:
        raise NotImplementedError
    complete = torch.nn.Sequential(
        padding,        
        gaussian_filter,
        )
    return complete
    #------------------------

def gaussian_filter(t,kernel_size=15,sigma = 3,padding='symmetric'):
    layer = _gaussian_kernel(channels=t.shape[1],kernel_size=kernel_size,sigma=sigma,padding=padding)
    layer.to(t.device)
    out = layer(t)
    return out
    
def anti_aliasing_filter(image,output_shape,
                                padding='symmetric',truncate=4.0):
    input_shape = image.shape[-2:]
    factors = (np.asarray(input_shape, dtype=float) /
                    np.asarray(output_shape, dtype=float))
    anti_aliasing_sigma = np.max(np.maximum(0, (factors - 1) / 2))
    
    half_kernel_size = int(truncate * anti_aliasing_sigma + 0.5)
    print(half_kernel_size)
    kernel_size = half_kernel_size*2 + 1 
    print(kernel_size)
    # mode = 'symmetric' ! 'constant'
    image = gaussian_filter(image, kernel_size = kernel_size,
    sigma=anti_aliasing_sigma, padding=padding)
    return image
    pass    
