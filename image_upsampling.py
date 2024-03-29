import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
import util
import os

def biquadratic_histospline(original_im_name,original_im,scale=2):
    upsampled_im = []
    h, w, c = original_im.shape
    upsampled_im_name = original_im_name[0:original_im_name.rfind('_', 0, \
                                                original_im_name.rfind('_'))]
    upsampled_im_name = upsampled_im_name + '_' + str(int(w*scale)) + '_' + \
                                        str(int(h*scale)) + '_histospline.png'
    upsampled_im = plt.imread(upsampled_im_name)

    return upsampled_im
def bilinear(im, win_size=20, scale=None, height=None, width=None):
        
    h, w, chan = im.shape
    
    if height and width:
        h_s, w_s = h/height, w/width
        h_s = np.round(h_s, decimals=2)
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
    
    im2 = np.clip(im2, 0, 1)
    
    return im2

def nedi(im, win_size=20, scale=None, height=None, width=None):
    
    h, w, chan = im.shape
    
    if int(height/h) == 4:
        flag = True
    else:
        flag = False
        
    scale = 2
    height, width = h*2, w*2
    im2s = []
    
    for kk in np.arange(chan):
        im1 = im[:,:,kk]
        
        xt = []
        yt = []
        for ii in range(1, h-1):
            for jj in range(1, w-1):
                yt.append(im1[ii][jj])
                if (ii+jj)%2==0 :
                    xt.append([im1[ii-1][jj-1],im1[ii-1][jj+1],\
                               im1[ii+1][jj-1],im1[ii+1][jj+1]])
                else :
                    xt.append([im1[ii][jj-1],im1[ii][jj+1],\
                               im1[ii+1][jj],im1[ii-1][jj]])
    
        x = np.array(xt)
        y = np.array(yt)
        
        b = np.linalg.pinv(x.T @ x) @ x.T @ y
        
        im2 = np.zeros((height, width))
        for ii in range(h):
            for jj in range(w):
                im2[scale*ii][scale*jj] = im1[ii][jj]
        
        for ii in range(1,height-2,scale):
            for jj in range(1,width-2,scale):
                im2[ii][jj] = np.matmul([im2[ii-1][jj-1],\
                                          im2[ii-1][jj+1],\
                                          im2[ii+1][jj-1],\
                                          im2[ii+1][jj+1]],b)
        
        for ii in range(1,height-1,scale):
            im2[ii][-1] = (im2[ii-1][-2] + im2[ii+1][-2])/2
                     
        for jj in range(1,width-1,2):
            im2[-1][jj] = (im2[-2][jj-1] + im2[-2][jj+1])/2
            
        for ii in range(1,height-2,scale):
            for jj in range(2,width-1,scale):
                im2[ii][jj] = np.matmul([im2[ii][jj-1],\
                                       im2[ii][jj+1],\
                                       im2[ii-1][jj],\
                                       im2[ii+1][jj]],b)
        
        for ii in range(2,height-1,scale):
            for jj in range(1,width-2,scale):
                im2[ii][jj] = np.matmul([im2[ii][jj-1],\
                                           im2[ii][jj+1],\
                                           im2[ii-1][jj],\
                                           im2[ii+1][jj]],b)
        for ii in range(2,height-1,scale):
            im2[ii][-1] = (im2[ii-1][-2] + im2[ii+1][-2])/2
        
        for jj in range(1,width-1,scale):
            im2[0][jj] = (im2[1][jj-1] + im2[1][jj+1])/2
        
        for ii in range(1,height-2,scale):
            im2[ii][0] = (im2[ii-1][1] + im2[ii+1][1])/2
            
        for jj in range(2,width-1,scale):
            im2[-1][jj] = (im2[-2][jj-1] + im2[-2][jj+1])/2
            
        im2[0][-1] = im2[1][-2]
        im2[-1][0] = im2[-2][1]
        
        im2s.append(im2)
    
    im2 = np.stack(im2s, axis=2)
    im2 = np.clip(im2, 0, 1)
    
    if flag:
        im2 = nedi(im2, height=height, width=width)
    
    return im2
    

def plot_images(original_im, ground_truth_im, upsampled_im,title=None):

    fig, ax = plt.subplots(1, 3, sharey=True, sharex=True)
#    mng = plt.get_current_fig_manager()
#    mng.window.showMaximized()

    if(title is not None):
        fig.suptitle("{}".format(title),fontsize=24)

    ax[0].imshow(original_im)
    ax[0].axis('off')
    ax[0].set_title("Original Image: {} x {}".format(original_im.shape[0], \
          original_im.shape[1]), fontsize=12)
    ax[1].imshow(ground_truth_im)
    ax[1].axis('off')
    ax[1].set_title("Ground Truth Image: {} x {}".format(\
          ground_truth_im.shape[0], ground_truth_im.shape[1]), fontsize=12)
    ax[2].imshow(upsampled_im)
    ax[2].axis('off')
    ax[2].set_title("Upsampled Image: {} x {}".format(upsampled_im.shape[0], \
          upsampled_im.shape[1]), fontsize=12)
    plt.show()

def compare_images(original_im_name, ground_truth_im_name, interp_func, \
                   plot_flag = False):
    
    original_im = skimage.io.imread(original_im_name)
    if original_im.shape[2]:
        original_im = skimage.color.rgba2rgb(original_im)
    
    ground_truth_im = skimage.io.imread(ground_truth_im_name)
    if ground_truth_im.shape[2]:
        ground_truth_im = skimage.color.rgba2rgb(ground_truth_im)


    if np.max(original_im) > 2:
        original_im = original_im / 255.0

    original_im = original_im.astype(np.float64)

    if np.max(ground_truth_im) > 2:
        ground_truth_im = ground_truth_im / 255.0

    ground_truth_im = ground_truth_im.astype(np.float64)
    
    if original_im.shape[0] == 129:
        original_im = original_im[:128,:128]
    
    upsampled_im = []
    if ('histospline' in interp_func.__name__ ):
        scale = ground_truth_im.shape[0]/original_im.shape[0]
        upsampled_im = interp_func(original_im_name,original_im,scale=scale )
    else:
        upsampled_im = interp_func(original_im, height=ground_truth_im.shape[0], \
                                   width=ground_truth_im.shape[1], win_size=20)
    
    assert ground_truth_im.shape == upsampled_im.shape, \
        "Interpolated Image not same size as True Image"

    if np.max(upsampled_im) > 2:
        upsampled_im = upsampled_im / 255.0

    upsampled_im = upsampled_im.astype(np.float64)
    
    if plot_flag:
        mse = util.mse(ground_truth_im, upsampled_im)
        psnr = util.psnr(ground_truth_im, upsampled_im)
        ssim, _ = skimage.measure.compare_ssim(ground_truth_im, upsampled_im, \
                                               full=True, multichannel=True)
    
        print("Resizing {} x {} to {} x {} using {} interpolation.".format(\
              original_im.shape[0], \
              original_im.shape[1], upsampled_im.shape[0], upsampled_im.shape[1], \
              interp_func.__name__))
        print("\nEvaluation Metrics:")
        print("Mean-Squared Error is {:.4f}".format(mse))
        print("Peak Signal-to-Noise Ratio is {:.4f}".format(psnr))
        print("Structural Similarity Index is {:.4f}".format(ssim))
        plot_images(original_im, ground_truth_im,upsampled_im,interp_func.__name__)
        print("--------------")
    return upsampled_im

def main():
    
    images = os.listdir('data')
    objects = ['rainbow', 'fishercat', 'giraffe', 'hippo', 'tiger']
    funcs = [bilinear, bicubic, nedi, biquadratic_histospline]
    for obj in objects:
        res = [i for i in images if obj in i]
        res = [i for i in res if 'png' in i]
        res = [i for i in res if not 'histospline' in i]
        
        if obj == 'rainbow':
            img300 = 'data/' + [i for i in res if '128' in i][0]
            img600 = 'data/' + [i for i in res if '256' in i][0]
            img1200 = 'data/' + [i for i in res if '512' in i][0]
        else:
            img300 = 'data/' + [i for i in res if '300' in i][0]
            img600 = 'data/' + [i for i in res if '600' in i][0]
            img1200 = 'data/' + [i for i in res if '1200' in i][0]
        
        for interp_func in funcs:
            
            scale2 = compare_images(img300, img600, interp_func)
            fname = 'results/' + obj + '_' + interp_func.__name__ + '_' + \
                    str(2) + '.png'
            
            if obj == 'rainbow':
                scale2 = scale2[:75,:75]
                
            plt.imsave(fname, scale2)
            print(fname)
            
            scale4 = compare_images(img300, img1200, interp_func)
            fname = 'results/' + obj + '_' + interp_func.__name__ + '_' + \
                    str(4) + '.png'
            
            if obj == 'rainbow':
                scale4 = scale4[:150,:150]
                    
            plt.imsave(fname, scale4)
            print(fname)
    
if __name__ == "__main__":
    main()

