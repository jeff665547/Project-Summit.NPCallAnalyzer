import numpy as np

def quantile_norm(data):
    temp = data.argsort(axis = 0)
    rank = np.empty_like(temp)
    for i in range(data.shape[1]):
        rank[temp[:,i],i] = np.arange(data.shape[0])
    return np.sort(data, axis = 0).mean(axis = 1)[rank]

def auto_contrast(images, thres = 40000, bins = 256):  
    image = images[0] if isinstance(images, list) else images
    if image.dtype == np.uint8:
        cmax = bins = 256
        cmin = 0
    elif image.dtype == np.uint16:
        cmax = bins = 65536
        cmin = 0
    elif image.dtype == np.float32 or image.dtype == np.float64:
        cmax = max(image.max() for image in images)
        cmin = 0
    else:
        raise Exception('undefined pixel type: {}'.format(image.dtype)) 
    h, x = np.histogram(image.ravel(), bins = bins, range = (cmin, cmax))
    if isinstance(images, list):
        for image in images[1:]:
            h += np.histogram(image.ravel(), bins = bins, range = (cmin, cmax))[0]
    c = len(images) if isinstance(images, list) else 1
    b = np.where(h > c * np.product(image.shape) / thres)[0]
    return x[b[0]], x[b[-1] + 1]