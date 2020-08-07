from termcolor import colored
import torch
import numpy as np
import itertools
from .masked_model import MaskedModel

def keep_feat(layer,input,output):
    layer.feat = output
    pass


def is_attach_here(lix,layers):
    layers = list(layers)
    this_l = layers[lix]
    if lix == 0:
        return True
    
    prev_l = layers[lix-1]
    if this_l.feat.shape != prev_l.feat.shape:
        if not isinstance(this_l,torch.nn.MaxPool2d):
            return True
    return False
def get_up_for_l(l,l_1):
    this_shape = l.feat.shape
    target_shape = l_1.feat.shape
    upsample_shape = (target_shape[0],this_shape[1]) + target_shape[2:]
    # up = torch.nn.Upsample(size=upsample_shape)
    up = torch.nn.Upsample(size=target_shape[2:],mode='bilinear')
    return up
def get_lateral_for_l(l,nchan):
    l_shape = l.feat.shape
    conv1x1 = torch.nn.Conv2d(l_shape[1],nchan,(1,1),1)
    return conv1x1
#--------------------------------
def get_mask_only_first_relu():
    print(colored('masking first relu','yellow'))
    seen = [False]
    def mask_only_first_relu(li,layers):
        layers = list(layers)
        l = layers[li]
        if not seen[0]:
            if isinstance(l,torch.nn.modules.activation.ReLU):
                seen[0] = True
                return True
        return False
    return mask_only_first_relu
def get_mask_layer_at_offset(offset = -12#-6#
                                ):
    print(colored(f'masking last relu ({offset})','red'))
    def mask_layer_at_offset(li,layers):
        layers = list(layers)
        if li == len(layers) + offset:
            print(colored(layers[li],'red','on_yellow'))
            return True
        return False
    return mask_layer_at_offset

def add_top_down_path(alexnet):
    # masked_convs = []
    conv_backbone =  alexnet.features
    feature_layers = list(conv_backbone.children())
    for l in feature_layers:
        l.register_forward_hook(keep_feat)
    #-------------------------------
    device = next(alexnet.parameters()).device
    dummy_forward = torch.tensor(np.zeros((1,3,227,227))).float().to(device)
    _ = alexnet(dummy_forward)
    running_chan = feature_layers[-1].feat.shape[1]
    #-------------------------------
    # alexnet.features,masked_convs = replace_with_masked_conv(alexnet.features)
    
    alexnet.features = MaskedModel(alexnet.features,
                                #    get_mask_only_first_relu() ,
                                    get_mask_layer_at_offset(), 
                                    mask_at = 'output')
    _ = alexnet(dummy_forward)
    conv_backbone =  alexnet.features
    feature_layers = list(conv_backbone.children())    


    layers = list(conv_backbone.children())
    n_layers = len(layers)
    #-------------------------------
    upsamples = {}
    laterals = {}
    up = get_up_for_l(layers[n_layers-1],layers[n_layers-2])
    upsamples[n_layers-1]=up
    current_area = None#layers[n_layers-1].feat.shape[-1]
    print(current_area)
    top_down_layer_ixs = []#[n_layers-1]
    # print(layers[top_down_layer_ixs[-1]].feat.shape)
    # n_found = 0
    
    for lix in range(n_layers):
            11111111111
            if  is_attach_here(lix,layers):
                top_down_layer_ixs.append(lix)
                print(colored(f'attach at {layers[lix]}','green'))
                continue
            print(colored(f'ignoring {layers[lix]}','red'))
    # for lix in range(n_layers-1,-1,-1): #skip first layer
    #     print(f'1:{lix}')
    #     l = layers[lix]
    #     if l.feat.shape[-1] == current_area:
    #         continue      
    #     # top_down_layer_ixs[n_found] = lix
    #     top_down_layer_ixs = [lix] + top_down_layer_ixs
    #     current_area = l.feat.shape[-1]
    #     # n_found += 1
    #     print(l.feat.shape,current_area)

    for i in range(len(top_down_layer_ixs)-1,0,-1):
        lix = top_down_layer_ixs[i]
        lix_1 = top_down_layer_ixs[i-1]
        #---------------------------
        l = layers[lix]
        l_1 = layers[lix_1]
        # l_1 = layers[lix-1]
        # target_n_chan = l_1.shape[1]
        up = get_up_for_l(l,l_1)
        print(l.feat.shape,up)      
        upsamples[lix]= up
        if lix < top_down_layer_ixs[-1]:
            lateral = get_lateral_for_l(l,running_chan)
            laterals[lix]=lateral
        # print(l)
    remaining = top_down_layer_ixs[0]
    laterals[remaining] = get_lateral_for_l(layers[remaining],running_chan)

    22222222222222 # commenting STE
    refine = torch.nn.Sequential(torch.nn.Upsample(size=(227,227)),
                                    torch.nn.ZeroPad2d(1),
        torch.nn.Conv2d(running_chan,1,(3,3)),
        torch.nn.InstanceNorm2d(running_chan),
                        # torch.nn.Conv2d(running_chan,1,(1,1),1),

                                # Average(),
                                #  SignumSTE(),
                                    torch.nn.Sigmoid()
                                    )

    refine_layers = list(refine.children())
    def record_pre_mask(self,input,output):
        global debug_dict
        debug_dict['pre_mask'] = input 
        debug_dict['mask'] = output
    refine_layers[-1].register_forward_hook(record_pre_mask)

    print(laterals.keys(),upsamples.keys())
    for part in [upsamples,laterals,refine]:
        if isinstance(part,dict):
            part = part.values()
        for l in part:
            l.to(device)
    return upsamples,laterals,refine,top_down_layer_ixs
# debugprint = print
debugprint = lambda *args:None

class AlexNetFPN():
    def __init__(self,alexnet):
        self.alexnet = alexnet
        self.forward_layers = list(alexnet.features.children())
        self.upsamples,self.laterals,self.refine,self.top_down_layer_ixs = add_top_down_path(self.alexnet)
        # self.instance_norms = {'lateral':{},'upsample':{}}
        # for ix_of_ix,lix in enumerate(self.laterals):
        #     self.instance_norms 
        self.instance_norms = {'lateral':{lix:torch.nn.InstanceNorm2d(
                                            l.out_channels) for lix,l in self.laterals.items()},
                                'upsample':{lix:torch.nn.InstanceNorm2d(
                                            l.out_channels) for lix,l in self.laterals.items()}}        
    def __call__(self,im,detach_bottom_up=False):
        self.alexnet.features.mask_dict = None
        initial_scores = self.alexnet(im)
        n_forward = len(self.forward_layers)
        # x = 0
        l_feat = self.forward_layers[self.top_down_layer_ixs[-1]].feat
        if detach_bottom_up:
                l_feat = l_feat.detach().clone()
        x = l_feat
        if detach_bottom_up:
            x = x.detach().clone()
        # lat = lambda x:x
        n_top_down_layers = len(self.top_down_layer_ixs)
        for i in range(n_top_down_layers-1,0,-1):
            debugprint('1',x.shape)
            fix = self.top_down_layer_ixs[i]
            f_next_ix = self.top_down_layer_ixs[i-1] 
            upsample = self.upsamples[fix]
            x_up = upsample(x)
            debugprint('2',x_up.shape)
            f_next = self.forward_layers[f_next_ix]
            lat = self.laterals[f_next_ix]
            # f_1 = self.forward_layers[fix-1]
            debugprint('3',f_next.feat.shape,x.shape)
            f_next_feat = f_next.feat
            if detach_bottom_up:
                f_next_feat = f_next_feat.detach().clone()
            f_next_lat = lat(f_next.feat)
            debugprint('4',f_next_lat.shape)
            x = self.instance_norms['upsample'][f_next_ix](x_up) + self.instance_norms['lateral'][f_next_ix](f_next_lat)
            # x = 00000000000
            # if i==1:
                # print(colored(f'x.max(),x.min()','yellow'))
            # if i>1:
            x = torch.nn.functional.relu(x)
            # lat = self.laterals[fix]
        # import pdb;pdb.set_trace()
        print(colored(f'{x.max()},{x.min()}','yellow'))
        mask = self.refine(x)
        # mask = x
        print(colored(f'{mask.max()},{mask.min()}','yellow'))
        refined_scores = self.masked_forward(im,{'input' : mask})
        return initial_scores,refined_scores,mask
    def masked_forward(self,im,mask_dict):
        self.alexnet.features.mask_dict = mask_dict
        scores = self.alexnet(im)
        self.alexnet.features.mask_dict = None
        return scores,self.alexnet.features.current_masks
    def parameters(self):
        parameters = [r.parameters() for r in self.laterals.values()] + [self.refine.parameters()]
        # parameters = [self.laterals[i].parameters() for i in [12,11,4]] + [self.refine.parameters()]
        return itertools.chain(*parameters)
        # return itertools.chain(*)
    def eval(self):
        self.alexnet.eval()
        for l in self.laterals.values():
            l.eval()
        self.refine.eval()
        pass
    def clear_mask(self,):
        self.alexnet.features.mask_dict = None
    # return reverse,reverse_layers
    
