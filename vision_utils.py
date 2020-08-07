
import torchvision
import numpy as np
vgg_mean = (0.485, 0.456, 0.406)
vgg_std = (0.229, 0.224, 0.225)
def get_vgg_transform_detransform(imsize=(227,227)):

    
    model_mean,model_std = vgg_mean,vgg_std 
    preprocess = torchvision.transforms.Compose([torchvision.transforms.Resize(imsize),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean = model_mean,std=model_std)
                                            ])
    denormalize = lambda t,vgg_mean=vgg_mean,vgg_std=vgg_std:(t * torch.tensor(vgg_std).view(1,3,1,1).to(t.device)) + torch.tensor(vgg_mean).view(1,3,1,1).to(t.device)
    return preprocess,denormalize

def get_preprocess_unsqueeze_to_device(preprocess,device):
    def preprocess_unsqueeze_to_device(im_pil):
        ref = preprocess(im_pil)
        ref = ref.unsqueeze(0)
        ref = ref.to(device)
        return ref
    return preprocess_unsqueeze_to_device

def convert_image_np(inp): 
    """Convert a Tensor to numpy image.""" 
    inp = inp.numpy().transpose((1, 2, 0)) 
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    inp = std * inp + mean 
    inp = np.clip(inp, 0, 1) 
    return inp
