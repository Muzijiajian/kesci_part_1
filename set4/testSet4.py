#encoding=utf-8
from cnn_util import *
import numpy as np
import os

resnet152_model = '/home/lord/caffe-master/Workspace/Kesci/resnet152/resnet_152_v2_iter_100000.caffemodel'
resnet152_deploy = '/home/lord/caffe-master/Workspace/Kesci/resnet152/ResNet-152-deploy.prototxt'
feat_path = '/home/lord/PycharmProjects/Kesci/set4/one_feats.npy'
cnn = CNN(model=resnet152_model, deploy=resnet152_deploy, width=224, height=224)

image_lists = []
image_name_cut = []
setone_image_path = '/home/lord/Downloads/kescitest/Testset 4_'
images = os.listdir(setone_image_path)
count = 0

for image in images:
    if image.endswith('jpg'):
        if image.startswith('._'):
            pass
        else:
            count += 1
            image = setone_image_path + '/' + image
            image_lists.append(image)
            tmp = image.split('.')[0]
            cut_image_name = tmp.split('/')[6]
            image_name_cut.append(cut_image_name)
print count
print image_name_cut
print os.path.exists(image_lists[3])
# 抽取图片相应特征
if not os.path.exists(feat_path):
    feats = cnn.get_features(image_lists, layers='fc1000_', layer_sizes=[12])
    np.save(feat_path, feats)

feat_path = '/home/lord/PycharmProjects/Kesci/set4/one_feats.npy'
feats = np.load(feat_path)
print feats.shape

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

# 获取前面两个label以及相应得概率
original_feats = np.load(feat_path)
soft_feats = softmax(feats)
label_1 = 0
prob_1 = 0
count = 0
# 用于保存的数据数组
label_1_lists = []
label_2_lists = []
prob_1_lists = []
prob_2_lists = []
for i in xrange(soft_feats.shape[0]):
    # 0-11
    label_1 = np.argmax(soft_feats[i])
    prob_1 = np.max(soft_feats[i])
    label_2 = 0
    prob_2 = 0
    for j in xrange(soft_feats.shape[1]):
        cur_prob = soft_feats[i][j]
        if cur_prob > prob_2 and cur_prob < prob_1:
            label_2 = j
            prob_2 = cur_prob
    print original_feats[i]
    print 'The first label is %d, the prob is %f' %(label_1, prob_1, )
    print 'The second label is %d, the prob is %f' %(label_2, prob_2, )
    label_1_lists.append(label_1)
    label_2_lists.append(label_2)
    prob_1 = format(prob_1, '.6f')      # 保留结果后边6位小数点
    prob_2 = format(prob_2, '.6f')
    prob_1_lists.append(prob_1)
    prob_2_lists.append(prob_2)
    print soft_feats[i]
    count += 1
    if count == 10475:
        break

# 将对应数据写入文本
with open('/home/lord/PycharmProjects/Kesci/set4/pred.txt', 'w') as fp:
    print len(image_name_cut), len(label_1_lists), len(prob_1_lists), len(label_2_lists), len(prob_2_lists)
    for i in xrange(10475):
        fp.write(image_name_cut[i] + '\t' + str(label_1_lists[i]) + '\t' + str(prob_1_lists[i]) + '\t' + str(label_2_lists[i]) + '\t' + str(prob_2_lists[i]) + '\n')