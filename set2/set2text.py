#encoding=utf-8
import numpy as np
import os

set_path = '/home/lord/PycharmProjects/Kesci/set2/BOT_Image_Testset 2.txt'
# 用于存储图片路径名相关
image_lists = []
full_image_path_lists = []
label_lists = []

fhandle = open(set_path)
count = 0
while True:
    line = fhandle.readline()
    # print line
    line_parts = line.split('\t')
    image_name = line_parts[0] + '.jpg'
    image_label = line_parts[1]
    count += 1
    # print image_name
    image_lists.append(image_name)
    label_lists.append(image_label)
    if count == 10052:
        break
fhandle.close()

print len(image_lists), len(label_lists)
print image_lists[:3]
print label_lists[:3]
print image_lists[-3:10052]
print label_lists[-3:10052]

prefix_path = '/home/lord/Downloads/KesciData/solved'
full_images = []     # 包含图片完整路径以及标签的字符串链表
for i in xrange(8496):
    full_image_path = prefix_path + '/' + image_lists[i]
    full_image_path_lists.append(full_image_path)
    if os.path.exists(full_image_path):
        per_line =  full_image_path + ' ' + str(label_lists[i])
        full_images.append(per_line)