import numpy as np

def mse(im1, im2):
	err = np.sum((im1.astype("float") - im2.astype("float")) ** 2)
	err /= float(im1.shape[0] * im1.shape[1])
	return err

def psnr(im1, im2):
    mse1 = mse(im1, im2)
    if mse1 == 0:
        return 100

    return 20 * np.log10(1 / np.sqrt(mse1))