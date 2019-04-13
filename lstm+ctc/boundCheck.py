#coding=utf-8  
import cv2 as cv
import os
import matplotlib.pyplot as plt

if __name__=="__main__":
    path='./NumImages/lstm_ctc'  #os.getcwd()
    list = os.listdir(path) #列出文件夹下所有的目录与文件
    for index in range(0,len(list)):
        file = path+'/'+list[index]
        print(file)
        file_name = file
        sobel_img = cv.imread(file_name, cv.IMREAD_GRAYSCALE)
        sobel_img= cv.resize(sobel_img,(256,32))
        x= cv.Sobel(sobel_img,cv.CV_16S,1,0)#x方向边缘检测
        y= cv.Sobel(sobel_img,cv.CV_16S,0,1)
        #x,y方向分辨转换为uint8(8位无符号)类型
        ux = cv.convertScaleAbs(x)
        uy = cv.convertScaleAbs(y)
        #x,y合并
        sb_img = cv.addWeighted(ux,0.5,uy,0.5,0)

        #设置阈值，如果原值大于100，设为0，因为边缘为白色要变成黑色，如果小于100，设为255ֵ
        retval,sb_img = cv.threshold(sb_img,100,255.0, cv.THRESH_BINARY_INV)
        
        #cv.imshow("result",sb_img)
        #cv.waitKey(0)  
        cv.destroyAllWindows()
        cv.imwrite('./NumImages2/'+list[index],sb_img)
