#coding=utf-8

import os
basePath = "E:\\wingIde\\Tensorflow1.1\\food\\data\\train\\yuxiangrousi\\"
#获取目录下所有文件存入表中\
index = 1
for n in range(1,2):
    #path=basePath+str(n)+"\\"
    path=basePath
    f =os.listdir(path)
    n = 0
    for i in f:
    #设置旧文件名
        oldName=path+f[n]   
        #设置新文件名
        newName = path+'yuxiangrousi_'+str(index)+'.jpg'
        #用os模块中rename方法对文件改名
        os.rename(oldName,newName)
        print(oldName,'======>',newName)
        n+=1
        index+=1
