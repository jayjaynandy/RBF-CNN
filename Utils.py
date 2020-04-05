import numpy as np
import scipy

#%%
def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    """
    transform images from [-1.0, 1.0] to [min_value, max_value] of dtype
    """
#    assert \
#        np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
#        and (images.dtype == np.float32 or images.dtype == np.float64), \
#        'The input images should be float64(32) and in the range of [-1.0, 1.0]!'
    if dtype is None:
        dtype = images.dtype
    return (images * (max_value - min_value) + min_value).astype(dtype)

#%%
def imwrite(image, path):

    if image.ndim == 3 and image.shape[2] == 1:  # for gray image
        image = np.array(image, copy=True)
        image.shape = image.shape[0:2]
    return scipy.misc.imsave(path, to_range(image, 0, 255, np.uint8))

#%%
def immerge(images, row, col):
    h, w = images.shape[1], images.shape[2]
    if images.ndim == 4:
        img = np.zeros((h * row, w * col, images.shape[3]))
    elif images.ndim == 3:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    return img

#%%
    
def npzLoad(npzFile = 'cifar.npz', dataFile = 'x_train', labelFile = 'y_train'): 
    cifar = np.load(npzFile)
    data = cifar[dataFile]
    label = cifar[labelFile]
    data = data.astype('float32')
    return data, label