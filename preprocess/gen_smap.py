import os
import numpy as np
from sigpy.mri import app
from multiprocessing.dummy import Pool

def genmap(path_t):
    save_str = os.path.basename(path_t)[1:-4]
    mask = np.load('../mask/poisson_320_320_8.npy')
    slice_i = np.load(path_t)
    slice_k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slice_i, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
    slice_kd = slice_k * mask
    slice_m = app.EspiritCalib(slice_kd, show_pbar=False, calib_width=24, kernel_width=6, crop=0, thresh=0.02).run()
    if np.sum(np.isnan(slice_m)):
        print('NaN : ', save_str)
    print(save_str)
    np.save(directory_map + '/map' + save_str + '.npy', slice_m)

# path of the data
directory_full = '/media/disk3/recondata/brain_8coil_val_sm_cor/'
# path for saving the maps
directory_map = '/media/disk3/recondata/brainmulti_esmapc0_posi20cw24k6_volumenorm_complex_val_npy_cor'

if not os.path.exists(directory_map):
    os.makedirs(directory_map)

fh = os.listdir(directory_full)
fh.sort()
fn_t = [os.path.join(directory_full, k) for k in fh]

pool = Pool(4)
pool.map(genmap, fn_t)
