import torch
import numpy as np
from termcolor import colored
from .gaussian_smoothing import anti_aliasing_filter
from .activation import *
class MaskedModel(torch.nn.Module):
    @staticmethod
    def _attach_masks(layers, masking_condition, mask_at = 'output'):
        # input_dict = {}
        # output_dict = {}
        # def keep_input_output(self,input,output)
        #     input_dict[self] = output
        #     input_dict[self] = output
        masked_layers = []
        for li,l in enumerate(layers):
            print(l.__class__)
            if masking_condition(li,layers):
                masked_layers.append(l)
        return masked_layers
        pass

    def __init__(self,model,masking_condition,mask_at='output'):
        super(MaskedModel,self).__init__()
        if isinstance(model,MaskedModel):
            print('input is already Masked')
            self.model = model.model
            self.mask_dict = model.mask_dict
            self.mask_at = model.mask_at
            self.model_layers = model.model_layers
            self.masking_condition = model.masking_condition
            self.masked_layers  = model.masked_layers
            return 
        # self.model = model        
        # self.current_masks = {}
        self.mask_dict = None
        self.mask_at = mask_at
        self.model_layers = list(model.children())
        print(self.model_layers)
        self.masking_condition = masking_condition        
        # self.mask_shapes = {}
        cls = self.__class__
        self.masked_layers = cls._attach_masks(self.model_layers,masking_condition,self.mask_at)
        pass
    # def set_masks(self,mask):
    #     self.current_masks = {}
    #     self.mask_dict = mask_dict
    #     prev_li = -1
    #     for li,(l,s) in enumerate(self.mask_shapes.items()):
    #         self.current_masks[l] = torch.nn.interpolate(mask_dict,s)
    #     pass
    
    '''
    @staticmethod
    def _masked_forward(l,x,mask_dict,mask_at):
    '''
    def forward(self,x):
        self.current_masks  = {}
        for li,l in enumerate(self.model_layers):
            if l not in self.masked_layers or self.mask_dict is None:
                x = l(x)
                continue
            #TODO search for shape being 4 dimensions
            
            if self.mask_at == 'input':
                # print('applying mask')
#                 raise NotImplementedError
                if li in self.mask_dict:
                    mask_l = self.mask_dict[li]
                else:
                    mask_l = anti_aliasing_filter(self.mask_dict['input'],x.shape[-2:],padding='symmetric')
                    mask_l = torch.nn.functional.interpolate(self.mask_dict['input'],x.shape[-2:])
                    mask_l = ste(mask_l); print(colored('using STE','blue','on_yellow'))
                # mask_l = noisy_ste(mask_l)
                x = mask_l * x
                self.current_masks[l] = mask_l
            x = l(x)
            if self.mask_at == 'output':
                

                if li in self.mask_dict:
                    mask_l = self.mask_dict[li]
                else:
                    mask_l = anti_aliasing_filter(self.mask_dict['input'],x.shape[-2:],padding='symmetric')
#                     debug_dict['smooth_mask'] = mask_l
                    mask_l = torch.nn.functional.interpolate(mask_l,x.shape[-2:],mode='bilinear')
                    mask_l = ste(mask_l); print(colored('using STE','blue','on_yellow'))
                # mask_l = noisy_ste(mask_l)
                self.current_masks[l] = mask_l
                
                x = mask_l * x
                
        return x

    def set_mask(self,**kwargs):
        self.mask_dict = kwargs
        pass        
    def children(self):
        return self.model_layers
    
    # @staticmethod
    # def masked_forward(model,x,input_mask):
    #     for part in model.children():
    #         if not isinstance(part,MaskedModel):
    #             x = part(x)
    #             continue
    #         x,masks_used = part(x,input_mask)
    #     return x,masks_used
    pass
