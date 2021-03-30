import random
from skimage.color import rgb2lab,lab2rgb
from skimage.transform import resize
from skimage.measure import compare_ssim,compare_psnr
import tensorflow as tf


def evaluate_quality(val_origin,result):
    num = result.shape[0]
    psnr,ssim = 0,0
    if val_origin.shape!=result.shape:
        print 'nothing to do'
    else:
        print '======+++evaluating+++======>'
    cur = np.zeros((256, 256, 3))
    for i in range(num):
        r=result[i]
        cur[:,:,0] = r[:,:,0]
        cur[:,:,1:] = r[:,:,1:]
        r = lab2rgb(cur)
        r *= 255.

        r = r.astype(np.uint8)
        v = val_origin[i].astype(np.uint8)
        ssim += compare_ssim(v,r, multichannel=True,data_range=255.0)
        psnr += compare_psnr(v,r,data_range=255.0)
    
    ssim /= num
    psnr /= num
    print 'result:psnr---',psnr,'---ssim:',ssim
    return psnr,ssim