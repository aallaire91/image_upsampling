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

def plot_images(original_im, ground_truth_im, upsampled_im,title=None ):

    fig, ax = plt.subplots(1, 3, sharey=True, sharex=True)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    if(title is not None):
        fig.suptitle("{}".format(title),fontsize=24)

    ax[0].imshow(original_im)
    ax[0].axis('off')
    ax[0].set_title("Original Image: {} x {}".format(original_im.shape[0], original_im.shape[1]), fontsize=12)
    ax[1].imshow(ground_truth_im)
    ax[1].axis('off')
    ax[1].set_title("Ground Truth Image: {} x {}".format(ground_truth_im.shape[0], ground_truth_im.shape[1]), fontsize=12)
    ax[2].imshow(upsampled_im)
    ax[2].axis('off')
    ax[2].set_title("Upsampled Image: {} x {}".format(upsampled_im.shape[0], upsampled_im.shape[1]), fontsize=12)
    plt.show()

def compare_images(original_im_name, ground_truth_im_name, interp_func):
    
    original_im = skimage.io.imread(original_im_name)
    original_im = skimage.color.rgba2rgb(original_im)

    ground_truth_im = skimage.io.imread(ground_truth_im_name)
    ground_truth_im = skimage.color.rgba2rgb(ground_truth_im)
    upsampled_im = interp_func(original_im, height=ground_truth_im.shape[0], width=ground_truth_im.shape[1], win_size=20)
    
    assert ground_truth_im.shape == upsampled_im.shape, "Interpolated Image not same size as True Image"
    
    mse = util.mse(ground_truth_im, upsampled_im)
    psnr = util.psnr(ground_truth_im, upsampled_im)
    ssim, _ = skimage.measure.compare_ssim(ground_truth_im, upsampled_im, full=True, multichannel=True)

    print("Resizing {} x {} to {} x {} using {} interpolation.".format(original_im.shape[0], \
          original_im.shape[1], upsampled_im.shape[0], upsampled_im.shape[1], interp_func.__name__))
    print("\nEvaluation Metrics:")
    print("Mean-Squared Error is {:.4f}".format(mse))
    print("Peak Signal-to-Noise Ratio is {:.4f}".format(psnr))
    print("Structural Similarity Index is {:.4f}".format(ssim))
    plot_images(original_im, ground_truth_im,upsampled_im,interp_func.__name__)

def main():
    
    im_name = 'data/wild_animal300.png'
    true_im_name = 'data/wild_animal600.png'
    
    print("Want Lower MSE, and Higher PSNR and SSIM\n")
    
    compare_images(im_name, true_im_name, bilinear)
    compare_images(im_name, true_im_name, bicubic)
    
if __name__ == "__main__":
    main()

