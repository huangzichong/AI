import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import os
from model_darknet19 import darknet
from decode import decode
from utils import preprocess_image, postprocess, draw_detection, generate_colors
from config import anchors, class_names
from loss import compute_loss
import math
import json
from handle_target import target_handle

C=80    #分类数
W=13   #特征图宽度
H=13   #特征图高度
B=5    #边框数
Itle = 1000  #迭代次数

model_path = os.path.join('yolo2_model','yolo2_coco.ckpt')    #加载模型路径
anchors = [[1.235,4.576],
           [4.160,6.162],
           [2.235,4.576],
           [2.235,4.576],
           [2.235,4.576]]

image_name ='car2.jpg'
image_file = os.path.join('images',image_name)   #images/car2.jpg
image = cv2.imread(image_file)  #read the image, images/car2.jpg
image_shape = image.shape[:2]

input_size = (416,416)

image_cp = preprocess_image(image)  #图像预处理，resize image, normalization归一化， 增加一个在第0的维度--batch_size


image_name ='test.jpg'
image_file = os.path.join('images',image_name)   #images/car2.jpg
image2 = cv2.imread(image_file)  #read the image, images/car2.jpg
image_shape = image2.shape[:2]

input_size = (416,416)



image_cp = preprocess_image(image)  #图像预处理，resize image, normalization归一化， 增加一个在第0的维度--batch_size
image_cp2 = preprocess_image(image2) 
image_batch = [image_cp,image_cp2]

image_cp = np.reshape(image_batch,[-1,input_size[0],input_size[1],3])


tf_image = tf.placeholder(tf.float32,[2,input_size[0],input_size[1],3])  #定义placeholder
model_output = darknet(tf_image)  #网络的输出
model_output = tf.reshape(model_output,[-1,H*W,B,5+C])

'''
#测试用
predictions = tf.reshape(model_output,[-1,H,W,B,(5+C)])
coords = tf.reshape(predictions[:,:,:,:,0:4],[-1,H*W,B,4]) # reshape成[batch_size, H*W, 5, 4]
coords_xy = tf.nn.sigmoid(coords[:,:,:,0:2])  # 0-1，xy是相对于cell左上角的偏移量, 计算得出的是偏移量
coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4])*anchors/np.reshape([W,H],[1,1,1,2])) # 0-1，除以特征图的尺寸13，解码成相对于整张图片的wh
coords = tf.concat([coords_xy,coords_wh],axis=3) # [batch_size, H*W, B, 4]
# 置信度
confs = tf.nn.sigmoid(predictions[:,:,:,:,4])    #经过一个sigmoid函数，使其落在[0-1]之间
confs = tf.reshape(confs,[-1,H*W,B,1]) # 每个边界框一个置信度，每个cell有B个边界框，reshape成[batch_size, H*W, 5, 1]
#  类别概率
probs = tf.nn.softmax(predictions[:,:,:,:,5:]) # 网络最后输出是"得分"，通过softmax变成概率
probs = tf.reshape(probs,[-1,H*W,B,C])   #reshape成[batch_size, H*W, B, C]
# prediction汇总
preds = tf.concat([coords,confs,probs],axis=3) #  reshape成[-1, H*W, B, (4+1+C)]

with open("./boxes.json",'r') as load_f:
     target = json.load(load_f)
result = target_handle(target,H,W,B,C)

targets = {'coords':[],'confs':[],'probs':[]}
targets['coords'] = result[:,:,:,0:4]
print(targets['coords'][:,99,:,:])
targets['confs'] = np.reshape(result[:,:,:,4:5],[-1,H*W,B])
targets['probs'] = result[:,:,:,5:5+C]
_coords = targets["coords"]  # ground truth [-1, H*W, B, 4]，真实坐标xywh
_probs = targets["probs"]  # class probability [-1, H*W, B, C] ，类别概率——one hot形式，C维
_confs = targets["confs"]  # 1 for object, 0 for background, [-1, H*W, B]，置信度，每个边界框一个
# ground truth计算IOU-->_up_left, _down_right
_wh = tf.pow(_coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
_areas = _wh[:, :, :, 0] * _wh[:, :, :, 1]
_centers = _coords[:, :, :, 0:2] #center 为中心坐标
_up_left, _down_right = _centers - (_wh * 0.5), _centers + (_wh * 0.5)
# ground truth汇总
truths = tf.concat([_coords, tf.expand_dims(_confs, -1), _probs], 3)

targets = np.zeros([1,H*W,B,5+C])
targets[:,99,4,84] = 1.0
targets[:,99,4,0:4] = [0.683,0.559,0.095,0.352]
targets[:,99,4,4] = 1.0

loss = tf.pow(model_output - truths, 2)
#loss = tf.reduce_sum(loss, axis=[1, 2, 3])
loss = tf.reduce_mean(loss)
# Optimizer
optimizer = tf.train.AdamOptimizer(0.0001)
 # Gradient Clipping
gradients = optimizer.compute_gradients(loss)
capped_gradients = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gradients if grad is not None]  #梯度剪裁
train_op = optimizer.apply_gradients(capped_gradients,name='train_op')
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(Itle):
        print_loss = sess.run(loss,{tf_image:image_cp})
        sess.run(train_op,{tf_image:image_cp})
        a = sess.run(model_output,{tf_image:image_cp})
        #c = sess.run(truths,{tf_image:image_cp})
        #print(c[0,99,:,:])
        print(a[0,99,4,:])
        print(print_loss)

'''


#获取target边框属性
with open("./boxes.json",'r') as load_f:
     target = json.load(load_f)
result = target_handle(target,H,W,B,C)

targets = {'coords':[],'confs':[],'probs':[]}
targets['coords'] = result[:,:,:,0:4]
targets['confs'] = np.reshape(result[:,:,:,4:5],[-1,H*W,B])
targets['probs'] = result[:,:,:,5:5+C]

output_sizes = input_size[0]//32, input_size[1]//32
sprob=np.ones([1,1,1,80])
sconf=np.ones([1,1,1,1])
snoob=np.ones([1,1,1,1])
scoor=np.ones([1,1,1,4])

train_op,loss,preds,confs_loss = compute_loss(model_output,targets,anchors,(sprob,sconf,snoob,scoor),num_classes=80)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(Itle):
        print_loss = sess.run(loss,{tf_image:image_cp})
        sess.run(train_op,{tf_image:image_cp})
        print(print_loss)
        a,b,c = decode(model_output=preds,output_sizes=output_sizes, num_class=C,anchors=anchors)    
        a1 = sess.run(a,{tf_image:image_cp})
        b1 = sess.run(b,{tf_image:image_cp})
        c1 = sess.run(c,{tf_image:image_cp})
        d = sess.run(confs_loss,{tf_image:image_cp})
        print(d)

#print(result)
#output_sizes = input_size[0]//32, input_size[1]//32 # 特征图尺寸是图片下采样32倍

#这个函数返回框的坐标（左上角-右下角），目标置信度，类别置信度
#output_decoded = decode(model_output=model_output,output_sizes=output_sizes, num_class=len(class_names),anchors=anchors)


