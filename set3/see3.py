#encoding=utf-8
'''
尝试实践googlenet测试时候将一张图片分别裁剪为144张224*224的子图
'''
import numpy as np
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize
from skimage.transform import rescale
from skimage.viewer import ImageViewer
import os

global crop_images_4

def rescale_image_and_mirror(image, crop_dims):
    '''
    Rescale the image into a crop_dims, without any loss
    :param image: (H x W x K) ndarrays
    :param crop_dims: (height, width) tuple for the crops.
    :return: (2 x H x W x K) ndarrays of crops
    '''
    im_shape = np.array(image.shape)
    scaled_image = rescale(image, crop_dims)
    crops = np.empty(2, crop_dims[0], crop_dims[1], im_shape[-1])
    crops[0] = scaled_image[:, :, :]
    crops[1] = scaled_image[:, ::-1, :]

    return crops

def oversmaple(image, crop_dims):
    """
    Crop image into the four corners, center and their mirrored versions.
    Parameters
    ----------
    image : (H x W x K) ndarrays
    crop_dims : (height, width) tuple for the crops.
    Returns
    -------
    crops : (10 x H x W x K) ndarray of crops.
    """
    # Dimensions and center
    im_shape = np.array(image.shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i+crop_dims[0], j+crop_dims[1])
            curr += 1

    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([-crop_dims / 2.0, crop_dims / 2.0])
    crops_ix = np.tile(crops_ix, (2, 1))    # Get a copy of all five positions, then do mirror
    # Extract crops
    crops = np.empty((12, crop_dims[0], crop_dims[1], im_shape[-1]), dtype=np.float32)
    ix = 0
    for crop in crops_ix:
        crops[ix] = image[crop[0]:crop[2], crop[1]:crop[3], :]
        ix += 1
    crops[ix - 5: ix] = crops[ix - 5:ix, :, ::-1, :]  # flip for mirrors
    tmp_image = resize(image, (crop_dims[0], crop_dims[1], im_shape[-1]))
    crops[10] = tmp_image[: ,:, :]
    crops[11] = tmp_image[:, ::-1, :]
    return crops

def resize_image(im, new_dims, interp_order=1):
    """
        Resize an image array with interpolation.
        Parameters
        ----------
        im : (H x W x K) ndarray
        new_dims : (height, width) tuple of new dimensions.
        interp_order : interpolation order, default is linear.
        Returns
        -------
        im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    if im.shape[-1] == 1 or im.shape[-1] == 3:
        im_min, im_max = im.min(), im.max()
        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images
            # in [0, 1].
            im_std = (im - im_min) / (im_max - im_min)
            resized_std = resize(im_std, new_dims, order=interp_order)
            resized_im = resized_std * (im_max - im_min) + im_min
        else:
            # the image is a constant -- avoid divide by 0
            ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                           dtype=np.float32)
            ret.fill(im_min)
            return ret
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im.astype(np.float32)

def load_image(filename, color=True):
    """
        Load an image converting from grayscale or alpha as needed.
        Parameters
        ----------
        filename : string
        color : boolean
            flag for color format. True (default) loads as RGB while False
            loads as intensity (if image is already grayscale).
        Returns
        -------
        image : an image with type np.float32 in range [0, 1]
            of size (H x W x 3) in RGB or
            of size (H x W x 1) in grayscale.
        """
    img = skimage.img_as_float(skimage.io.imread(filename, as_grey=not color)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            # If gray, then copy to RGB
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        # if has alpha, then remove
        img = img[:, :, :3]
    return img


def get_img_longside(img):
    '''
    Dicide with side is the longer image side
    :param img:
    :return: 0 for height, 1 for width
    '''
    height, width = img.shape[0], img.shape[1]
    if height >= width:
        long_side = 0
    else:
        long_side = 1
    return long_side

def process(image_dir):
    '''

    :param image_dir:
    :return:
    '''
    img = load_image(image_dir)     # <type 'numpy.ndarray'>
    print 'Original image shape:', img.shape
    global flag
    if img.shape[0] > 352 or img.shape[1] > 352:
        flag = 1
        first_crop = [256, 288, 320, 352]
        crop_images_1 = []
        long_side = get_img_longside(img)
        height, width = img.shape[0], img.shape[1]
        for i in xrange(len(first_crop)):
            if height < width:
                tmp_dim = (first_crop[i], width)
            else:
                tmp_dim = (height, first_crop[i])
            crop_images_1.append(resize_image(img, tmp_dim))
            # print crop_images_1[i].shape
        # Second crop time, get up&middle&bottom images based on the long side
        crop_images_2 = []
        count = 0  # value to iterate first crop
        for image in crop_images_1:
            # short_side sequencely equals [256, 288, 320, 352]
            short_side = first_crop[count]
            divide = int((img.shape[long_side] - short_side) / 3)
            if long_side:
                tmp_image_first = image[:, :short_side, :]
                tmp_image_middle = image[:, divide:divide + short_side, :]
                tmp_image_bottom = image[:, img.shape[long_side] - short_side:, :]
            else:
                tmp_image_first = image[:short_side, :, :]
                tmp_image_middle = image[divide:divide + short_side, :, :]
                tmp_image_bottom = image[img.shape[long_side] - short_side:, :, :]
            crop_images_2.append(tmp_image_first)
            crop_images_2.append(tmp_image_middle)
            crop_images_2.append(tmp_image_bottom)
            count += 1
        # Final step: resample each image into 12 images
        global crop_images_4
        crop_images_4 = []
        # crop_images_3_part2 = []
        crop_dims = (224, 224)
        for image in crop_images_2:
            crop_images_4.append(oversmaple(image, crop_dims))
            # crop_images_3_part2.append(rescale_image_and_mirror(image, crop_dims))
        # Get the avage image then save it into local
        avg_image = np.zeros((crop_dims[0], crop_dims[1], img.shape[-1]), dtype=np.float32)
        for image_tuple in crop_images_4:
            avg_image += np.mean(image_tuple, axis=0)
        # Average image then save it into local
        avg_image /= 12.0
    else:
        # Image size is limited
        flag = 0
        avg_image = resize(img, (224, 224, 3))

    return avg_image, crop_images_4, flag

def arraylist2array(array_lists):
    together_array = np.zeros((144, 224, 224, 3))
    count = 0
    for each_array in array_lists:
        for i in xrange(12):
            together_array[count, :, :, :] = each_array[i, :, :, :]
            count += 1
    return together_array

def save_144(images_array_lists, save_dir):
    '''
    For given one image, save it into a directory file that contains separate parts images
    :param images_array_lists:
    :param save_dir:
    :return:
    '''
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    images_array = arraylist2array(images_array_lists)
    for i in xrange(images_array.shape[0]):
        image_name = str(i) + '.jpg'
        image_save_dir = save_dir + '/' + image_name
        skimage.io.imsave(image_save_dir, images_array[i, :, :, :])

if __name__ == '__main__':
    # image_dir = '/home/lord/Downloads/kescitest/test_crop/0a00f354468043b798f1c2907406adc6.jpg'
    # avg_images = process(image_dir)
    # save_path = '/home/lord/PycharmProjects/Kesci/only/test.png'
    # skimage.io.imsave(save_path, avg_images)

    images_dir = '/home/lord/Downloads/kescitest/Testset_4'
    # images_dir = '/home/lord/Downloads/kescitest/test_crop'
    save_images_dir = '/home/lord/Downloads/kescitest/crops/crops_set4'
    all_images = os.listdir(images_dir)

    if not os.path.exists(save_images_dir):
        os.mkdir(save_images_dir)
        print 'Created save images file.'
    for image in all_images:
        if not image.endswith('.jpg'):
            pass
        else:
            image_fullpath = os.path.join(images_dir, image)
            print image_fullpath
            # Deal with the image&save it into file
            avg_image, crop_images, flag = process(image_fullpath)
            save_image_fullpath = os.path.join(save_images_dir, image)
            if not flag:
            # Just resized images
                skimage.io.imsave(save_image_fullpath, avg_image)
            else:
            # store the image into the local file
                image_name = image.split('.')[0]
                image_name = os.path.join(save_images_dir, image_name)
                save_144(crop_images, image_name)