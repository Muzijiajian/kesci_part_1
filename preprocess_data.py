#encoding=utf-8
import cv2
import os
import sys
from PIL import Image

def processGifImage(infile, outfile):
    try:
        im = Image.open(infile)
        im.save(outfile, 'PNG')
    except IOError:
        print "Cant load", infile
        # sys.exit(1)

def All_To_JPG(file_path, save_path):
    '''
    注意gif图片的格式需要进行二次转换
    :param file_path:
    :return:
    '''
    images_lists = os.listdir(file_path)
    count = 0
    for image in images_lists:
        # 找到不是以jpg结尾得图片
        if not image.endswith('.jpg'):
            # Dealing with the file name
            old_image_path = os.path.join(file_path, image)
            print 'Differ!!', image
            tmp_image = image.split('.')[0]
            if image.endswith('.gif'):
                # Situation with gif: First to png then to jpg
                # 将结尾替换为png
                new_gif_image = tmp_image + '.png'
                gif_outfile = file_path + '/' + new_gif_image
                processGifImage(old_image_path, gif_outfile)
                # 将结尾替换为jpg
                new_image = tmp_image + '.jpg'
                new_image_path = save_path + '/' + new_image
                im = cv2.imread(gif_outfile)
                cv2.imwrite(new_image_path, im)
            else:
                # 将结尾替换为jpg
                new_image = tmp_image + '.jpg'
                new_image_path = save_path + '/' + new_image
                # print new_image_path
                # Read the original image
                im = cv2.imread(old_image_path)
                # Change the image type to new image path
                cv2.imwrite(new_image_path, im)
            count += 1
    print 'Total dealed %d members.' %(count)



if __name__ == '__main__':
    images_path = '/home/lord/Downloads/KesciData/test_part/test_preprocess'
    save_path = '/home/lord/Downloads/KesciData/test_part/test_save'
    All_To_JPG(images_path, save_path)