#encoding=utf-8
import os
import tensorflow as tf
from PIL import Image

cwd = os.getcwd()
#图片文件夹目录，可有多层
classes = {'tensor'}
def create_record():
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for index, name in enumerate(classes):
        class_path = cwd +"/"+ name+"/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224))
            img_raw = img.tobytes() #将图片转化为原生bytes
            example = tf.train.Example(
               features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
               }))
            writer.write(example.SerializeToString())
    writer.close()
    
'''
data = create_record()
'''