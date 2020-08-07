import numpy as np
def position_jitter(t,jitter_radius,imsize):
    x_offset = np.random.choice(range(jitter_radius))    
    y_offset = np.random.choice(range(jitter_radius))
    # print(x_offset,y_offset)    
    return t[:,:,y_offset:y_offset+imsize[0],x_offset:x_offset+imsize[1]]
