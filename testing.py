#encoding=utf-8
from cnn_util import *
import numpy as np
import os

vgg_model = '/home/lord/caffe-master/Workspace/Kesci/baseline/VGG_19_model/vggmodel_iter_100000.caffemodel'
vgg_deploy = '/home/lord/caffe-master/Workspace/Kesci/baseline/vgg_19_deploy.prototxt'

# vgg_model = '/home/lord/caffe-master/models/bvlc_reference_caffenet/VGG_ILSVRC_19_layers.caffemodel'
# vgg_deploy = '/home/lord/PycharmProjects/video2text/VGG/VGG_ILSVRC_19_layers_deploy.prototxt'

feat_path = '/home/lord/PycharmProjects/Kesci/feats.npy'    # 存储所有图片抽取特征之后的地址

# cnn = CNN(model=vgg_model, deploy=vgg_deploy, width=224, height=224)    # 声明一个cnn卷积得部署网络

test_image_path = '/home/lord/PycharmProjects/Kesci/test.txt'
image_lists = []
image_labels = []
fhandle = open(test_image_path)
count = 0
# 读取文本信息
while 1:
    line = fhandle.readline()
    # print line.split(' ')[0]
    image_lists.append(line.split(' ')[0])
    image_labels.append(line.split(' ')[1].replace('\n', ''))   # 第二个输出由于带有\n换行符，所以提前进行替换消除
    count += 1
    if count == 982:
        break
fhandle.close()
print len(image_lists)
print image_labels
# 抽取图片相应特征
# if not os.path.exists(feat_path):
    # feats = cnn.get_features(image_lists, layers='n_fc8', layer_sizes=[12])
    # np.save(feat_path, feats)
#
feats = np.load(feat_path)
print feats.shape
pred_labels = np.argmax(feats, axis=1)
print np.argmax(feats, axis=1)
# accuracy = ( np.argmax(feats, axis=1) == image_labels)

# new_image_labels = []
# for image_label in image_labels:
#     image_label = image_label.replace('\n', '')
#     # print image_label
#     new_image_labels.append(image_label)
# print new_image_labels

same_count = 0
for i in xrange(count):
    if pred_labels[i] == int(image_labels[i]):
        same_count += 1
    else:
        print 'Diff image, Number: %d ... the real image is %d, pred it into %d.' %(i, int(image_labels[i]), pred_labels[i], )
        print feats[i][pred_labels[i]], feats[i][int(image_labels[i])]
# 输出准确率
print 1.0 * same_count / 982

animal2num = {'天竺鼠': 0, '鬣狗': 1, '松鼠': 2, '梅花鹿': 3, '花栗鼠': 4, '狗': 5,
              '狼': 6, '驯鹿': 7, '狐狸': 8, '黄鼠狼': 9, '猫': 10, '长颈鹿': 11}

def softmax(x):
    """
    Compute the softmax function for each row of the input x.
    :param x:
    :return:
    """
    if len(x.shape) > 1:
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp

    return x

# print np.sum(softmax(feats), axis=1)

# 获取前面两个label以及相应得概率
original_feats = np.load(feat_path)
soft_feats = softmax(feats)
label_1 = 0
prob_1 = 0
count = 0
for i in xrange(soft_feats.shape[0]):
    # 0-11
    label_1 = np.argmax(soft_feats[i])
    prob_1 = np.max(soft_feats[i]) * 100
    label_2 = 0
    prob_2 = 0
    for j in xrange(soft_feats.shape[1]):
        cur_prob = soft_feats[i][j] * 100
        if cur_prob > prob_2 and cur_prob < prob_1:
            label_2 = j
            prob_2 = cur_prob
    print original_feats[i]
    print 'The first label is %d, the prob is %f' %(label_1, prob_1, )
    print 'The second label is %d, the prob is %f' %(label_2, prob_2, )
    print soft_feats[i]
    count += 1
    if count == 20:
        break