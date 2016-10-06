#encoding=utf-8
import caffe
import caffe.draw
from caffe.proto import caffe_pb2
from google.protobuf import text_format

# 网络结构文件
input_net_proto_file = '/home/lord/softwares/caffe/examples/coco_caption/lrcn.prototxt'
# 输出图片文件
output_image_file = '/home/lord/PycharmProjects/video2text/net_pic.jpg'
# 网络结构排列方式： LR、TB、RL等
rankdir = 'LR'
# 读取网络
net = caffe_pb2.NetParameter()
text_format.Merge(open(input_net_proto_file).read(), net)
# 绘制网络
print ('Drawing net to %s' % output_image_file)
caffe.draw.draw_net_to_file(net, output_image_file, rankdir)
print 'Done...'