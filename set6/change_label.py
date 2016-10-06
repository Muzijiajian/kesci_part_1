#encoding=utf-8
'''
由于softmax只是最大化当前类别的概率，并未充分利用top2对物体相近之间的性质
'''
import numpy as np



original_pred_dir = '/home/lord/PycharmProjects/Kesci/set6/pred.txt'
image_names = []
pred_top1 = []
prob_top1 = []
pred_top2 = []
prob_top2 = []

prob_gate = 0.8    # 表示置信度低于多少进行特征替换

with open(original_pred_dir, 'r') as fread:
    lines = fread.readlines()
    for line in lines:
        line = line.rstrip('\n')
        line_apart = line.split('\t')
        image_names.append(line_apart[0])
        pred_top1.append(line_apart[1])
        prob_top1.append(line_apart[2])
        pred_top2.append(line_apart[3])
        prob_top2.append(line_apart[4])


print pred_top1[10108], prob_top1[10108], pred_top2[10108], prob_top2[10108]
print len(pred_top2)

count = 0
new_pred_top2 = []
new_pred_top1 = []
for pred in pred_top1:
    pred  = int(pred)
    prob = float(prob_top1[count])
    if pred == 0:
        if prob  < 0.5:
            new_pred_top1.append(1)
            new_pred_top2.append(pred)
        else:
            new_pred_top1.append(pred_top1[count])
            new_pred_top2.append(pred_top2[count])
    if pred == 1:
        if prob > 0.99:
            new_pred_top2.append(7)
        else:
            new_pred_top2.append(pred_top2[count])
        new_pred_top1.append(pred_top1[count])
    if pred == 2 :
        if prob > 0.99:
            new_pred_top2.append(9)
        else:
            new_pred_top2.append(pred_top2[count])
        new_pred_top1.append(pred_top1[count])
    if pred == 3:
        new_pred_top2.append(pred_top2[count])
        new_pred_top1.append(pred_top1[count])
    if pred == 4:
        new_pred_top2.append(pred_top2[count])
        new_pred_top1.append(pred_top1[count])
    if pred == 5:
        new_pred_top2.append(pred_top2[count])
        new_pred_top1.append(pred_top1[count])
    if pred == 6:
        new_pred_top2.append(pred_top2[count])
        new_pred_top1.append(pred_top1[count])
    if pred == 7:
        if prob > 0.6:
            new_pred_top2.append(1)
        else:
            new_pred_top2.append(pred_top2[count])
        new_pred_top1.append(pred_top1[count])
    if pred == 8:
        new_pred_top2.append(pred_top2[count])
        new_pred_top1.append(pred_top1[count])
    if pred == 9:
        if prob > 0.99:
            new_pred_top2.append(4)
        else:
            new_pred_top2.append(pred_top2[count])
        new_pred_top1.append(pred_top1[count])
    if pred == 10:
        new_pred_top2.append(pred_top2[count])
        new_pred_top1.append(pred_top1[count])
    if pred == 11:
        new_pred_top2.append(pred_top2[count])
        new_pred_top1.append(pred_top1[count])
    count += 1


print len(new_pred_top2)
new_pred_dir = '/home/lord/PycharmProjects/Kesci/set6/new_pred.txt'
with open(new_pred_dir, 'w') as fwrite:
    for i in xrange(count):
        fwrite.write(image_names[i] + '\t' + str(new_pred_top1[i]) + '\t' + str(prob_top1[i]) + '\t' + str(new_pred_top2[i]) + '\t' + str(prob_top2[i]) + '\n')