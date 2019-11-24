import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
import util

def bilinear(im, win_size=20, scale=None, height=None, width=None):
        
    h, w, chan = im.shape
    
    if height and width:
        h_s, w_s = h/height, w/width
    elif scale:
        h_s, w_s = 1/scale, 1/scale
        height, width = h/h_s, w/w_s
    else:
        print('Must specify either scale or height/width')
        return im
    
    im2 = []
    for kk in np.arange(chan):
        vert = []
        for ii in np.arange(0, h, win_size):
            hor = []
            for jj in np.arange(0, w, win_size):
                a, b, c, d = ii, min(ii+win_size,h), jj, min(jj+win_size,w)
                tmp_im = im[a:b, c:d, kk]
                x1, y1 = np.arange(a, b), np.arange(c, d)
                f = scipy.interpolate.interp2d(x1, y1, tmp_im.T, kind='linear')
                x2, y2 = np.arange(a, b, h_s), np.arange(c, d, w_s)
                tmp_im2 = f(x2, y2).T
                hor.append(tmp_im2)

            hor = np.hstack(hor)
        
            vert.append(hor)
        vert = np.vstack(vert)
        
        im2.append(vert)
    im2 = np.stack(im2, axis=2)
    
    if np.max(im) > 2:
        im2 = np.clip(im2, 0, 255)
    else:
        im2 = np.clip(im2, 0, 1)
    
    return im2

def bicubic(im, win_size=20, scale=None, height=None, width=None):
    
    h, w, chan = im.shape
    
    if height and width:
        h_s, w_s = h/height, w/width
    elif scale:
        h_s, w_s = 1/scale, 1/scale
        height, width = h/h_s, w/w_s
    else:
        print('Must specify either scale or height/width')
        return im
    
    im2 = []
    for kk in np.arange(chan):
        vert = []
        for ii in np.arange(0, h, win_size):
            hor = []
            for jj in np.arange(0, w, win_size):
                a, b, c, d = ii, min(ii+win_size,h), jj, min(jj+win_size,w)
                tmp_im = im[a:b, c:d, kk]
                x1, y1 = np.arange(a, b), np.arange(c, d)
                f = scipy.interpolate.interp2d(x1, y1, tmp_im.T, kind='cubic')
                x2, y2 = np.arange(a, b, h_s), np.arange(c, d, w_s)
                tmp_im2 = f(x2, y2).T
                hor.append(tmp_im2)

            hor = np.hstack(hor)
        
            vert.append(hor)
        vert = np.vstack(vert)
        
        im2.append(vert)
    im2 = np.stack(im2, axis=2)
    
    if np.max(im) > 2:
        im2 = np.clip(im2, 0, 255)
    else:
        im2 = np.clip(im2, 0, 1)
    
    return im2

def plot_images(im1, im2):
    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
    ax[0].imshow(im1)
    ax[0].axis('off')
    ax[0].set_title("Original image: {} x {}".format(im1.shape[0], im1.shape[1]), fontsize=10)
    ax[1].imshow(im2)
    ax[1].axis('off')
    ax[1].set_title("New image: {} x {}".format(im2.shape[0], im2.shape[1]), fontsize=10)
    plt.show()

def compare_images(im_name, true_im_name, interp_func):
    
    im = skimage.io.imread(im_name)
    im = skimage.color.rgba2rgb(im)

    true_im = skimage.io.imread(true_im_name)
    true_im = skimage.color.rgba2rgb(true_im)
    im2 = interp_func(im, height=true_im.shape[0], width=true_im.shape[1], win_size=20)
    
    assert true_im.shape == im2.shape, "Interpolated Image not same size as True Image"
    
    mse = util.mse(true_im, im2)
    psnr = util.psnr(true_im, im2)
    ssim, _ = skimage.measure.compare_ssim(true_im, im2, full=True, multichannel=True)

    print("Resizing {} x {} to {} x {} using {}".format(im.shape[0], \
          im.shape[1], im2.shape[0], im2.shape[1], interp_func.__name__))
    print("Mean-Squared Error is {:.4f}".format(mse))
    print("Peak Signal-to-Noise Ratio is {:.4f}".format(psnr))
    print("Structural Similarity Index is {:.4f}".format(ssim))
    plot_images(im, im2)

def main():
    
    im_name = '../data/wild_animal300.png'
    true_im_name = '../data/wild_animal450.png'
    
    print("Want Lower MSE, and Higher PSNR and SSIM\n")
    
    compare_images(im_name, true_im_name, bilinear)
    compare_images(im_name, true_im_name, bicubic)
    
if __name__ == "__main__":
    main()

