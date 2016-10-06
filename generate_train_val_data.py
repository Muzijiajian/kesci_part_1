#encoding=utf-8
import os
import numpy as np

animal2num = {'天竺鼠': 0, '鬣狗': 1, '松鼠': 2, '梅花鹿': 3, '花栗鼠': 4, '狗': 5, '狼': 6,
              '驯鹿': 7, '狐狸': 8, '黄鼠狼': 9, '猫': 10, '长颈鹿': 11}


def getCurClass(dataname):
    '''
    用于获取对应的动物类别
    :param dataname:
    :return:
    '''
    cur_class = -1
    if dataname == 'guinea pig':
        cur_class = 0
    if dataname == 'squirrel':
        cur_class = 1
    if dataname == 'sikadeer':
        cur_class = 2
    if dataname == 'fox':
        cur_class = 3
    if dataname == 'Dog':
        cur_class = 4
    if dataname == 'wolf':
        cur_class = 5
    if dataname == 'cat':
        cur_class = 6
    if dataname == 'chipmunk':
        cur_class = 7
    if dataname == 'giraffe':
        cur_class = 8
    if dataname == 'reindeer':
        cur_class = 9
    if dataname == 'hyena':
        cur_class = 10
    if dataname == 'weasel':
        cur_class = 11
    return cur_class

# dataset_path = '/home/lord/Downloads/KesciData/PreData'
# all_datasets = os.listdir(dataset_path)

# cur_class = 0
# full_images = []
#
# for per_datastes in all_datasets:
#     fullpath_per_datastes = os.path.join(dataset_path, per_datastes)
#     image_lists = os.listdir(fullpath_per_datastes)
#     count = 0
#     for image in image_lists:
#         if image.endswith('.jpg'):
#             # 不知道为什么会出现这一问题
#             if image.startswith('._'):
#                 pass
#             else:
#                 count += 1
#                 new_image_name = image + ' ' + str(cur_class)
#                 full_image_path = fullpath_per_datastes + '/' + new_image_name
#                 full_images.append(full_image_path)
#             # print new_image_name
#     cur_class += 1
#     print per_datastes, 'images', count
# print 'Total images are:', len(full_images)
# # Shuffle the image lists
# np.random.shuffle(full_images)
# train_dataset = full_images[:5000]
# val_dataset = full_images[5000:]

# 这里用于正式的训练集合训练
dataset_path = '/home/lord/Downloads/KesciData/bot_train'
all_datasets = os.listdir(dataset_path)

cur_class = 0
full_images = []

for per_datastes in all_datasets:
    if per_datastes.endswith('rar'):
        pass
    else:
        # 文件夹开头的文件
        cur_class = getCurClass(per_datastes)
        print 'Dealing with', per_datastes, 'Class is', cur_class
        fullpath_per_datastes = os.path.join(dataset_path, per_datastes)
        image_lists = os.listdir(fullpath_per_datastes)
        count = 0
        for image in image_lists:
            if image.endswith('.jpg'):
                # 不知道为什么会出现这一问题
                if image.startswith('._'):
                    pass
                else:
                    count += 1
                    new_image_name = image + ' ' + str(cur_class)
                    full_image_path = fullpath_per_datastes + '/' + new_image_name
                    full_images.append(full_image_path)
                # print new_image_name
        cur_class += 1
        print per_datastes, 'images', count
print 'Total images are:', len(full_images)
# Shuffle the image lists
np.random.shuffle(full_images)
train_dataset = full_images[:118000]
val_dataset = full_images[118000:]

with open('/home/lord/PycharmProjects/Kesci/train.txt', 'w') as fp:
    for per_line in train_dataset:
        fp.write(per_line + '\n')
with open('/home/lord/PycharmProjects/Kesci/test.txt', 'w') as fp:
    for per_line in val_dataset:
        fp.write(per_line + '\n')
# np.savetxt('/home/lord/PycharmProjects/Kesci/train.txt', train_dataset)
# np.savetxt('/home/lord/PycharmProjects/Kesci/test.txt', val_dataset)






























