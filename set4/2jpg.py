#encoding=utf-8
'''
用于把图片变成jpg格式？
'''
import os
import numpy
import Image

datadir = "/home/lord/Downloads/kescitest/Testset 7"
savedatadir = "/home/lord/Downloads/kescitest/Testset_7"

if not os.path.exists(savedatadir):
    os.mkdir(savedatadir)

# change the image mode
for eachimage in os.listdir(datadir):
   judgejpg = eachimage.split('.')
   img = Image.open(datadir+'/' + eachimage)
   if img.mode != 'RGB' :
       img = img.convert('RGB')
   #print img.format , img.size , img.mode
   #if os.path.exists(savedatadir + '/' + classname ) == False:
   #	os.mkdir(savedatadir + '/' + classname)
   temp = judgejpg[0].split(' - ')
   print savedatadir + '/' + ''.join(temp) + '.jpg'
   img.save(savedatadir + '/' + ''.join(temp) + '.jpg')
