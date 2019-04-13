import numpy as np
import tensorflow as tf
import cv2
import json
import math
def target_handle(target,H,W,B,C):
    batch_size = len(target)
    template = np.zeros([batch_size,H,W,B,5+C])
    #template[:,0:7,0:8,:,:] = 1.0
    for batch_index,batch in enumerate(target):
        for index,data in enumerate(batch['boxInfo']):
            xy = np.floor(np.multiply(data['coords'][0:2],[W,H]))
            xy = xy.astype(int)
            template[batch_index,xy[1],xy[0],:,4] = 1.0
            a = math.modf(np.multiply(data['coords'][0],[W]))
            b = math.modf(np.multiply(data['coords'][1],[H]))
                                                              
            template[batch_index,xy[1],xy[0],:,0] = a[0]
            template[batch_index,xy[1],xy[0],:,1] = b[0]

            template[batch_index,xy[1],xy[0],:,2:4] = data['coords'][2:4]
            template[batch_index,xy[1],xy[0],:,5:C+5] = data['probs']
    template = np.reshape(template,[-1,H*W,B,5+C])
    template = template.astype(np.float32)
    return template

