import numpy as np
from numba import jit
import random
import cv2
import os

class adamOp(object):
  #构造函数
  def __init__(self, gradient,learn):
    #self 类似this指针
    # self.name 定义类属性
    shape = gradient.shape
    self.momentum = 0.1
    self.alpha = learn
    self.time = 0
    self.mt = np.zeros(shape)
    self.vt = np.zeros(shape)

class normaOp(object):
  #构造函数
  def __init__(self):
    #self 类似this指针
    self.running_mean=0.0
    self.running_var=0.0
    self.hat_x=0.0
    self.avg=0.0
    self.var=0.0
    self.x=0.0
    self.gamma=0.1
    self.beta=0.1
    self.out_media=0.0
    
def adam(adamOper,theta,gradient):
    # 初始化
    b1 = 0.9  # 算法作者建议的默认值
    b2 = 0.999  # 算法作者建议的默认值
    e = 0.00000001  #算法作者建议的默认值
    adamOper.mt = b1 * adamOper.mt + (1 - b1) * gradient
    adamOper.vt = b2 * adamOper.vt + (1 - b2) * (gradient**2)
    mtt = adamOper.mt / (1 - (b1**(adamOper.time + 1)))
    vtt = adamOper.vt / (1 - (b2**(adamOper.time + 1)))
    vtt_sqrt = np.sqrt(vtt)  
    theta = theta - adamOper.alpha * mtt / (vtt_sqrt + e)
    adamOper.time=adamOper.time+1
    return theta

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def sigmoid_derivative(y):
    ds = y * (1 - y)
    return ds

def relu(x):
    return np.where(x>0,x,0.1*(np.exp(x)-1))

def relu_derivative(x,y):
    return np.where(x>0,1,0.1+y)

def cost_derivative(y_pre,y):
    return y_pre-y
    
@jit(nopython = True,cache = True,nogil = True)
def arrayDot(a,b):
    length=len(a)
    c=0
    for i in range(0,length):
        c+=a[i]*b[i]
    return c    
    
@jit(nopython = True,cache = True,nogil = True)
def conv_jit(x,w,x_shape,w_shape,step=1,padding=0,dropout=0):
    filter_sum=0
    batch_size=x_shape[0]
    index_i=w_shape[0]
    index_j=(x_shape[1]-w_shape[1])/step+1
    index_k=(x_shape[2]-w_shape[2])/step+1
    result=[[[[0.00000 for i in range(0,index_i)]for k in range(0,index_k)] for j in range(0,index_j)] for h in range(0,batch_size)]
    for h in range(0,batch_size):
        for i in range(0,index_i):
            for j in range(0,index_j):
                for k in range(0,index_k):
                    for o in range(0,w_shape[1]):
                        for p in range(0,w_shape[2]):                   
                            filter_sum=filter_sum+arrayDot(x[h][j+o][k+p],w[i][o][p])
                    rand = random.uniform(0, 1)
                    if dropout==1:
                       if rand >0.2:
                          result[h][j][k][i]=filter_sum
                       else:
                          result[h][j][k][i]=0.0
                    else:
                       result[h][j][k][i]=filter_sum
                    filter_sum=0
    return result

@jit(nopython = True,cache = True,nogil = True)
def conv_back_jit(forward,back,forward_shape,back_shape,step=1,padding=0):
    batch_size=forward_shape[0]
    index_i=back_shape[3]
    index_j=(forward_shape[1]-back_shape[1])/step+1
    index_k=(forward_shape[2]-back_shape[2])/step+1
    filter_sum=0.0
    result=[[[[[0.00000 for l in range(0,forward_shape[3])]for k in range(0,index_k)]for j in range(0,index_j)] for i in range(0,index_i)]for h in range(0,batch_size)]
    for h in range(0,batch_size):
        for i in range(0,index_i):
            for r in range(0,forward_shape[3]):
                for j in range(0,index_j):
                    for k in range(0,index_k):
                        for o in range(0,back_shape[1]):
                            for p in range(0,back_shape[2]):
                                filter_sum=filter_sum+forward[h][j+o][k+p][r]*back[h][o][p][i]
                        result[h][i][j][k][r]=filter_sum
                        filter_sum=0.0
    return result

@jit(nopython = True,cache = True,nogil = True)
def conv_cita_jit(cita,w,cita_shape,w_shape,step=1):
    batch_size=cita_shape[0]
    index_i=w_shape[0]
    index_j=(cita_shape[1]-w_shape[1])/step+1
    index_k=(cita_shape[2]-w_shape[2])/step+1
    w_i=w_shape[1]
    w_j=w_shape[2]
    filter_sum=0
    result=[[[[0.00000 for i in range(0,w_shape[3])]for k in range(0,index_k)] for j in range(0,index_j)] for h in range(0,batch_size)]
    resultTemp=[[[[[0.00000 for i in range(0,w_shape[3])]for k in range(0,index_k)] for j in range(0,index_j)]for t in range(0,w_shape[0])]for h in range(0,batch_size)]
    for h in range(0,batch_size):
        for i in range(0,index_i):
            for r in range(0,w_shape[3]):
                for j in range(0,index_j):
                    for k in range(0,index_k):
                        for m in range(0,w_i):
                            for n in range(0,w_j):
                                filter_sum=filter_sum+cita[h][j+m][k+n][i]*w[i][w_i-m-1][w_j-n-1][r]
                        resultTemp[h][i][j][k][r]=filter_sum
                        filter_sum=0

    for h in range(0,batch_size):
        for r in range(0,w_shape[3]):
            for j in range(0,index_j):
                for k in range(0,index_k):
                    for i in range(0,index_i):
                        result[h][j][k][r] = result[h][j][k][r]+resultTemp[h][i][j][k][r]
    return result

@jit(nopython = True,cache = True,nogil = True)
def maxpool_jit(x,step,x_shape):
    batch_size=x_shape[0]
    index_i=x_shape[1]
    index_j=x_shape[2]
    index_k=x_shape[3]
    max_=0
    result=[[[[0.0000 for k in range(0,index_k)]for j in range(0,index_j)]for i in range(0,index_i)]for h in range(0,batch_size)]
    result2=[[[[0.0000 for k in range(0,index_k)]for j in range(0,index_j/2)]for i in range(0,index_i/2)]for h in range(0,batch_size)]
    for h in range(0,batch_size):
        for k in range(0,index_k):
            i=0
            while i < index_i:
                j=0
                while j < index_j:
                    for m in range(0,step):
                        for n in range(0,step):
                            if x[h][i+m][j+n][k]>max_:
                               max_=x[h][i+m][j+n][k]
                    for m in range(0,step):
                        for n in range(0,step):
                            if x[h][i+m][j+n][k]==max_:
                               result[h][i+m][j+n][k]=1.0
                            else:
                               result[h][i+m][j+n][k]=0.0
                    a=int(i/step)
                    b=int(j/step)
                    result2[h][a][b][k]=max_           
                    max_=0           
                    j=j+2        
                i=i+2
    return result2,result

def maxpool(x,step=2):
    a,b=maxpool_jit(list(x),step,x.shape)
    return np.array(a),np.array(b)
    
def conv(x,w,step=1,padding=0,dropout=0):
    a=np.pad(x,((0,0),(padding,padding),(padding,padding),(0,0)),'constant')
    result=conv_jit(list(a),list(w),a.shape,w.shape,step,padding,dropout)
    return np.array(result)
    
def conv_back(forward,back,step=1,padding=0):
    result=conv_back_jit(list(forward),list(back),forward.shape,back.shape,step,padding)
    return np.mean(result,axis=0)
        
def conv_cita(cita,w):
    padding_ud=w.shape[1]-1
    padding_lr=w.shape[2]-1
    cita_=np.pad(cita,((0,0),(padding_ud,padding_ud),(padding_lr,padding_lr),(0,0)),'constant')
    result=conv_cita_jit(list(cita_),list(w),cita_.shape,w.shape)
    return result

def batch_normalization(x,normaOper,mode='train'):
    momentum=0.9
    eps=1e-5
    normaOper.x=x
    if mode == 'train':
      normaOper.mean = np.mean(x,axis = 0)
      normaOper.var = np.var(x,axis = 0)
      normaOper.running_mean = normaOper.running_mean * momentum + (1-momentum) * normaOper.mean
      normaOper.running_var = normaOper.running_var * momentum + (1-momentum) * normaOper.var
      normaOper.out_media = (x-normaOper.mean)/np.sqrt(normaOper.var + eps)
      out = (normaOper.out_media + normaOper.beta) * normaOper.gamma
    elif mode == 'test':
      out = (x-normaOper.running_mean)/np.sqrt(normaOper.running_var+eps)
      out = (out + normaOper.beta) * normaOper.gamma     
    return out

def batch_normalization_back(normaOper,dout):
    momentum=0.9
    eps=1e-5
    dout_media = dout * normaOper.gamma
    dgamma = np.sum(dout * (normaOper.out_media + normaOper.beta),axis = 0)
    dbeta = np.sum(dout * normaOper.gamma,axis = 0)
    dx = dout_media / np.sqrt(normaOper.var + eps)
    dmean = -np.sum(dout_media / np.sqrt(normaOper.var+eps),axis = 0)
    dstd = np.sum(-dout_media * (normaOper.x - normaOper.mean) / (normaOper.var + eps),axis = 0)
    dvar = 1./2./np.sqrt(normaOper.var+eps) * dstd
    dx_minus_mean_square = dvar / normaOper.x.shape[0]
    dx_minus_mean = 2 * (normaOper.x-normaOper.mean) * dx_minus_mean_square
    dx += dx_minus_mean
    dmean += np.sum(-dx_minus_mean,axis = 0)
    dx += dmean / normaOper.x.shape[0]
    normaOper.gamma=normaOper.gamma-0.1*dgamma
    normaOper.beta=normaOper.beta-0.1*dbeta
    return dx
     
def get_batch(class_num,size_each_class,image_size=(240,240)):
    y=['' for i in range(0,class_num)]
    result=[0.00 for i in range(0,class_num*5)]
    g=[[0.00 for j in range(0,class_num)]for i in range(0,class_num*5)]
    for i in range(0,class_num):
        y[i]=str(i+1)+'_'+str(random.randint(1,size_each_class))
        image_file = os.path.join('imageForTrain',y[i]+'.bmp')   #images/car2.jpg
        image2 = cv2.imread(image_file)  #read the image, images/car2.jpg
        image2 = imageCrop(image2)
        for j in range(0,5):
            result[i*5+j]=preprocess_image(image2[j],image_size)
            g[i*5+j][int(class_num-i-1)]=1.0
    return np.expand_dims(np.array(result),axis=-1),np.array(g)

def preprocess_image(image,image_size):
    # 复制原图像
    image_cp = np.copy(image).astype(np.float32)

    # resize image
    image_rgb = cv2.cvtColor(image_cp,cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(image_rgb,image_size)

    # normalize归一化
    image_normalized = image_resized.astype(np.float32) / 225.0

    # 增加一个维度在第0维——batch_size
    #image_expanded = np.expand_dims(image_normalized,axis=0)
    return image_normalized

def imageCrop(image):
    cropImg=[0,0,0,0,0]
    cropImg[0] = np.repeat(np.repeat(image[0:120,0:120],2,0),2,1)
    cropImg[1] = np.repeat(np.repeat(image[0:120,120:240],2,0),2,1)
    cropImg[2] = np.repeat(np.repeat(image[120:240,0:120],2,0),2,1)
    cropImg[3] = np.repeat(np.repeat(image[120:240,120:240],2,0),2,1)
    cropImg[4] = image
    return cropImg
  
w=np.random.normal(0,1,[10,7,7,1])*np.sqrt(1/(49+1)) #116x116
w2=np.random.normal(0,1,[10,4,4,10])*np.sqrt(1/(160+1)) #56x56
w3=np.random.normal(0,1,[10,4,4,10])*np.sqrt(1/(160+1)) #27x27
w4=np.random.normal(0,1,[10,8,8,10])*np.sqrt(1/(640+1)) #10x10
w5=np.random.normal(0,1,(1000,3))*np.sqrt(2/(1000+3)) #DNN weight

adam_w=adamOp(w,0.001)
adam_w2=adamOp(w2,0.0007)
adam_w3=adamOp(w3,0.0006)
adam_w4=adamOp(w4,0.0002)
adam_w5=adamOp(w5,0.0001)

norma=normaOp()
norma2=normaOp()
norma3=normaOp()
norma4=normaOp()

k=0
for i in range(0,1000):
    '''
    r=np.load('normaOp.npz')
    print(r['norma'][()].beta)
    '''
    x,y=get_batch(3,57)
    a=conv(x,w)  #234x234
    a_n=batch_normalization(a,norma,mode='train')
    a_=relu(a_n)
    a__,a_pool_der=maxpool(a_,step=2) #117x117
    
    b=conv(a__,w2,padding=0) #114x114
    b_n=batch_normalization(b,norma2,mode='train')
    b_=relu(b_n)
    
    b__,b_pool_der=maxpool(b_,step=2) #57x57
    
    c=conv(b__,w3) #54x54
    c_n=batch_normalization(c,norma3,mode='train')
    c_=relu(c_n)
    c__,c_pool_der=maxpool(c_,step=2) #27x27

    d=conv(c__,w4) #20x20
    d_n=batch_normalization(d,norma4,mode='train')
    d_=relu(d_n)
    d__,d_pool_der=maxpool(d_,step=2) #10x10
    dnn=np.reshape(d__,[d__.shape[0],d__.shape[1]*d__.shape[2]*d__.shape[3]]) #将卷积输出转为1维
    dnn *= np.random.binomial([np.ones((dnn.shape[0],dnn.shape[1]))],1-0.4)[0]#全连接层加入dropout
    dnn_=np.dot(dnn,w5)
    dnn__=sigmoid(dnn_)
    print(dnn__)
    cost=cost_derivative(dnn__,y)
    cita1=sigmoid_derivative(dnn__)*cost
    
    #deta=np.dot(np.expand_dims(cita1,axis=0).T,np.expand_dims(dnn,axis=0))
    deta=np.dot(cita1.T,dnn) 
    cita2=np.reshape(np.dot(cita1,w5.T),d__.shape)
    cita2=relu_derivative(d_n,d_)*np.repeat(np.repeat(cita2,2,1),2,2)*d_pool_der
    cita2=batch_normalization_back(norma4,cita2)
    w5=adam(adam_w5,w5,deta.T)
    
    deta=conv_back(c__,cita2)
    cita3=conv_cita(cita2,w4)
    cita3=relu_derivative(c_n,c_)*np.repeat(np.repeat(cita3,2,1),2,2)*c_pool_der
    cita3=batch_normalization_back(norma3,cita3)
    w4=adam(adam_w4,w4,deta)

    deta=conv_back(b__,cita3)
    cita4=conv_cita(cita3,w3)
    cita4=relu_derivative(b_n,b_)*np.repeat(np.repeat(cita4,2,1),2,2)*b_pool_der
    cita4=batch_normalization_back(norma2,cita4)
    w3=adam(adam_w3,w3,deta)

    deta=conv_back(a__,cita4)
    cita5=conv_cita(cita4,w2)
    cita5=relu_derivative(a_n,a_)*np.repeat(np.repeat(cita5,2,1),2,2)*a_pool_der
    cita5=batch_normalization_back(norma,cita5)
    w2=adam(adam_w2,w2,deta)
    
    deta=conv_back(x,cita5)
    w=adam(adam_w,w,deta)
    if i%100==0:
       print("save")
       np.savez("weight.npz", w=w, w2=w2, w3=w3, w4=w4, w5=w5)
       np.savez("normaOp.npz",norma=norma, norma2=norma2, norma3=norma3, norma4=norma4)
    



