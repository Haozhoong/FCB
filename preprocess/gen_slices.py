import h5py
import os
import numpy as np

def normalization(x, max):
    x = x  / max
    return x

def img_crop(data, shape):
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]

dir = '/media/shz/ZhQ/knee/multicoil_val/'
h5list = os.listdir(dir)
h5list = [h5name for h5name in h5list if '.h5' in h5name]
h5list.sort()

# dir to save the data
directory_full = '/home/shz/recondata/kneemulti_target_norm_complex_npy_val'
if not os.path.exists(directory_full):  #判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(directory_full)
    
cropsize = (320, 320)
volume_numbers = 100 # number of volumes to process
modality = 'CORPD_FBK' # need to be consistent with the modality of the data
volume_idx = 0
total_slices = 0

for h5name in h5list:
    f = h5py.File(os.path.join(dir, h5name))
    slice_idx = 0
    if dict(f.attrs)['acquisition'] == modality:
        num_slices = f.get('kspace').shape[0]
        slice_range = range(int(num_slices*0.2), int(num_slices))
        num_coil = f['kspace'].shape[1]
        volume_full = np.zeros((num_coil, cropsize[0], cropsize[1], len(slice_range)), dtype=np.complex64)
        maps = np.zeros((num_coil, cropsize[0], cropsize[1], len(slice_range)), dtype=np.complex64)
        
        for idx, i in enumerate(slice_range):
            slice_i = np.array(f['kspace'][i])
            slice_i = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(slice_i, axes=(-2, -1))))
            slice_i = img_crop(slice_i, cropsize)
            map = slice_i / np.sqrt(np.sum(abs(slice_i) ** 2, axis=0, keepdims=True))
            volume_full[:,:,:,idx] = slice_i
            maps[:,:,:,idx] = map
        
        volume_full_rss = np.sum(volume_full * np.conj(maps), axis=-2)
        max_full = abs(volume_full_rss).max()
        volume_full = normalization(volume_full, max_full)
        
        for i in range(len(slice_range)):
            fui = volume_full[:, :, :, i]
            map = maps[:, :, :, i]
            fui_rss = abs(np.sum(fui * np.conj(map), axis=0))

            save_str = "%03d" % volume_idx + '_' + "%03d" % slice_idx
            print(save_str)
            slice_idx += 1
            total_slices += 1
            np.save(directory_full + '/s' + save_str + '.npy', fui)

        volume_idx += 1
    if volume_idx >= volume_numbers:
        break