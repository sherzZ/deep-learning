#coding=utf-8
import re
import urllib.request
from collections import deque
import threading
from time import ctime,sleep

def getHtml(url):
    request = urllib.request.Request(url)
    page = urllib.request.urlopen(request)
    #print(type(page))
    #print(page.geturl())
    #print(page.info())  # 头部信息
    html = page.read()
    with open("E:\wingIde\spider\douyu.txt",'wb') as tf:
        tf.write(html)
    html=html.decode('utf-8')  
    #html = json.loads(html)
    return html

# 熊猫tv
def get_image_panda(html, path,startIndex=1):
    regex = r'<img (.*?)>'
    pattern = re.compile(regex,re.I|re.S|re.M) # 将正则表达式编译成pattern对象
    data=pattern.findall(html) # 以列表形式返回所有匹配的子串
    num=startIndex
    for d in data:
        data_original = d.split(" ")[2]
        img_url = data_original.split("\"")[1]
        print(img_url)
        image = download_page(img_url)
        with open(path+"%s.jpg"%num,'wb') as fp:
            fp.write(image)
            num+=1
            print("正在下载第%s张图片"%num)
    return num

# 斗鱼tv 
def get_image_douyu(html,path,startIndex=1):
    regex = r'<img (.*?)>'
    pattern = re.compile(regex,re.I|re.S|re.M) # 将正则表达式编译成pattern对象
    data=pattern.findall(html) # 以列表形式返回所有匹配的子串
    num=startIndex
    for d in data:
        regex = r'^data-original=(.*?)$'
        pattern = re.compile(regex,re.I|re.S) # 将正则表达式编译成pattern对象
        data2 =pattern.findall(d)
        for d2 in data2:
            img_url = d2.split(" ")[0].split("\"")[1]
            print(img_url)
            image = download_page(img_url)
            with open(path+"%s.jpg"%num,'wb') as fp:
                fp.write(image)
                num+=1
                print("正在下载第%s张图片"%num)
        #print(d)
    return num

def download_page(url):
    request = urllib.request.Request(url)
    response=urllib.request.urlopen(request)
    data = response.read()
    return data
    

def getImage(num,path):
    num =num;
    # 熊猫
    html=getHtml(url)
    num=get_image_panda(html,path,num)    
    
    #斗鱼
    html_douyu=getHtml(url_douyu)
    num=get_image_douyu(html_douyu,path,num)   
    
    #斗鱼颜值
    html_douyu=getHtml(url_douyu_yanzhi)
    num = get_image_douyu(html_douyu,path,num)  
    return num

class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False
#value=1
url = "http://www.panda.tv/cate/yzdr"
url_douyu = "https://www.douyu.com/directory/game/xx"
url_douyu_yanzhi = "https://www.douyu.com/directory/game/yz"

num =1
path = "F:\\爬图图片\\"
while True:
    num = getImage(num,path)
    sleep(300)

       

#for case in switch(value):
    #if(case(1)):
        ## 熊猫
        #html=getHtml(url)
        #get_image_panda(html,239) 
    #if(case(2)):
        ##斗鱼
        #html_douyu=getHtml(url_douyu)
        #get_image_douyu(html_douyu,365)   
    #if(case(3)):
        ##斗鱼颜值
        #html_douyu=getHtml(url_douyu_yanzhi)
        #get_image_douyu(html_douyu,245)         