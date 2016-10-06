#encoding=utf-8
import numpy as np
'''
用于计算个人生成的文本与标准标签的得分
score = top1 + 0.4*top2 + 2*hide_tasks
'''

def compute_score(true_data_file, input_result_file):
    true_dict = {}
    with open(true_data_file, 'r') as fp:
        count = 0
        for line in fp.readlines():
            count += 1
            line_lists = line.split('\t')
            file_name = line_lists[0]
            file_label = line_lists[1]
            file_is_hidden = line_lists[3].rstrip('\n')
            # 获取非隐藏任务图片
            if '0' in file_is_hidden:
                # print 'haha'
                true_dict[file_name] = file_label

    # 计算准确率
    top1_correct = 0
    top2_correct = 0
    with open(input_result_file, 'r') as fp:
        for line in fp.readlines():
            line_list = line.split('\t')
            # print line_list
            file_name = line_list[0]
            try:
                # print line_list[1], true_dict[file_name]
                if line_list[1] == true_dict[file_name]:
                    top1_correct += 1
                if line_list[3] == true_dict[file_name]:
                    top2_correct += 1
            except:
                continue

    top1_ratio = 1.0 * top1_correct / count
    top2_ratio = 1.0 * top2_correct / count

    print "The top1: {0} and top2: {1}".format(top1_ratio, top2_ratio)
    print "The sum score: {0}".format(top1_ratio + 0.4*top2_ratio)



if __name__ == '__main__':
    real_label = '/home/lord/Downloads/kescitest/BOT_Image_Testset 7.txt'
    # my_label = '/home/lord/PycharmProjects/Kesci/set1/final_pred.txt'
    my_label = '/home/lord/Downloads/kescitest/crops/alex_pred7.txt'
    change_label = '/home/lord/Downloads/kescitest/crops/google_pred7.txt'
    new_label = '/home/lord/Downloads/kescitest/crops/crop_pred2.txt'
    compute_score(real_label, my_label)
    compute_score(real_label, change_label)
    compute_score(real_label, new_label)