import os,os.path    
import shutil,string  

dir = "F:\Python27\work\deal-file\pos"  
outdir = "pos"  
label = " 1"  

fileList = os.listdir(dir)  
#列出dir目录下的目录和文件  

fileinfo = open('list.csv','w')  
#将结果保存在list.csv中  

for i in fileList:  
    curname = os.path.join(outdir, i)  
    print (curname)  
    fileinfo.write(curname +  ' 1' + '\n') #这里 1 为正样本的标签  
    #print i  
fileinfo.close()  