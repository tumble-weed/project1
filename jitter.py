import numpy as np
def position_jitter(t,jitter_radius,imsize):
    if jitter_radius > 0:
        x_offset = np.random.choice(range(jitter_radius))    
        y_offset = np.random.choice(range(jitter_radius))
        # print(x_offset,y_offset)    
        return t[:,:,y_offset:y_offset+imsize[0],x_offset:x_offset+imsize[1]]
    assert (t.shape[-2] == imsize[0]) and (t.shape[-1] == imsize[1]), 'HW shape of tensor should be same as imsize for jitter_radius=0'
    return t
