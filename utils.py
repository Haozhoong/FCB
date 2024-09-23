import numpy as np
import torch
import random
import yaml
from types import SimpleNamespace

from model.UNet import UNet
from model.FUNet import FUNet
from model.MoDL import MoDL
from model.FMoDL import FMoDL
from model.VSNet import VSNet
from model.FVSNet import FVSNet
from model.E2EVar import VarNet
from model.FE2EVar import FVarNet

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_args(config_path):
    with open(config_path, 'r') as f:
        args_all = list(yaml.safe_load_all(f))
        
    args_data = args_all[0]
    args_model = args_all[1]
    args_training = args_all[2]
    args_other = args_all[3]
    args = SimpleNamespace(**args_data, **args_model, **args_training, **args_other)

    # new add componment
    if not hasattr(args, 'seed'):
        args.seed = 42
    
    if not hasattr(args, 'shift'):
        args.shift = True
        
    if not hasattr(args, 'return_shift'):
        args.return_shift = True

    args.schedule_on = hasattr(args, 'scheduler')
    
    if not hasattr(args, 'last_decay'):
        args.last_decay = -1
        
    if not hasattr(args, 'gradclip'):
        args.gradclip = None
        
    if not hasattr(args, 'epochstep'):
        args.epochstep = False
        
    if args.schedule_on and args.scheduler == 'CosineAnnealingLR':
        args.epochstep = True
        
    if not hasattr(args, 'norm'):
        args.norm = 1
        
    # for LUnet
    if not hasattr(args, 'kernel_size'):
        args.kernel_size = 3
    if not hasattr(args, 'lcb_l'):
        args.lcb_l = 0
    if not hasattr(args, 'lcb_forward'):
        args.lcb_forward = False
        
    # type transform
    args.LR = float(args.LR)
    args.weight_decay = float(args.weight_decay)
    if hasattr(args, 'LR_max'):
        args.LR_max = float(args.LR_max)
    if hasattr(args, 'LR_repara'):
        args.LR_repara = float(args.LR_repara)
    if hasattr(args, 'weight_decay_repara'):
        args.weight_decay_repara = float(args.weight_decay_repara)
    if hasattr(args, 'momentum_repara'):
        args.momentum_repara = float(args.momentum_repara)
        
    args.mask_path = './mask/{}_{}_{}_{}.npy'.format(args.mask_type, args.img_size, args.img_size, args.sampling_factor)

    # data_path, Need to be modified
    if args.data_type == 'brain':
        args.directory_full_train = '/data/smart/shz/dataset/fastmri/brain_npy/brain_8coil_sm/'
        args.directory_full_val = '/data/smart/shz/dataset/fastmri/brain_npy/brain_8coil_val_sm_cor/'
        args.directory_m_train = '/data/smart/shz/dataset/fastmri/brain_npy/brainmulti_esmapc0_posi20cw24k6_slicenorm_complex_train_npy_new'
        args.directory_m_val = '/data/smart/shz/dataset/fastmri/brain_npy/brainmulti_esmapc0_posi20cw24k6_slicenorm_complex_val_npy_cor'
            
    elif args.data_type == 'knee':
        args.directory_full_train =  '/data/smart/shz/dataset/fastmri/knee_npy/kneemulti_target_slicenorm_complex_npy'
        args.directory_full_val =  '/data/smart/shz/dataset/fastmri/knee_npy/kneemulti_target_slicenorm_complex_val_npy'
        args.directory_m_train = '/data/smart/shz/dataset/fastmri/knee_npy/kneemulti_esmapc0_posi20cw24k6_slicenorm_complex_npy_new'
        args.directory_m_val = '/data/smart/shz/dataset/fastmri/knee_npy/kneemulti_esmapc0_posi20cw24k6_slicenorm_complex_val_npy'

    return args, args_data, args_model, args_training, args_other

def get_model(args):
    if args.model_name == 'UNet':
        model = UNet(in_ch=args.n_coils, out_ch=1, channels=args.n_channels,
                     dwconv=args.dwconv, shorcut=args.shorcut,
                     circ_pad=args.circ_pad, res_study=False,
                     kernel_size=args.kernel_size, lcb=args.lcb_l,
                     forward=args.lcb_forward)
        
    elif args.model_name == 'MoDL':
        model = MoDL(n_layers=args.n_layers, k_iters=args.k_iters)
        
    elif args.model_name == 'FUNet':
        model = FUNet(in_ch=args.n_coils, out_ch=1, num_rows=args.img_size,
                      num_cols=args.img_size, channels=args.n_channels,
                      res_study=False, fcb=args.fcb_l, forward=args.fcb_forward)
        
    elif args.model_name == 'FMoDL':
        model = FMoDL(n_layers=args.n_layers, k_iters=args.k_iters, num_rows=args.img_size,
                      num_cols=args.img_size, fcb_l=args.fcb_l)
        
    elif args.model_name == 'VSNet':
        model = VSNet(alfa=args.alfa, beta=args.beta,
                      cascades=args.cascades, dwconv=args.dwconv, shift=args.shift)
        
    elif args.model_name == 'FVSNet':
        model = FVSNet(alfa=args.alfa, beta=args.beta,
                       cascades=args.cascades, num_rows=args.img_size,
                       num_cols=args.img_size, fcb_l=args.fcb_l, shift=args.shift)
        
    elif args.model_name == 'VarNet':
        model = VarNet(num_cascades=args.num_cascades, sens_chans=args.sens_chans,
                       chans=args.chans, mask_center=args.mask_center)
        
    elif args.model_name == 'FVarNet':
        model = FVarNet(num_cascades=args.num_cascades, sens_chans=args.sens_chans,
                        chans=args.chans, mask_center=args.mask_center)
        
    else:
        print('Unknown model!')
        raise(ValueError)
        
    return model
    
def get_savepath(args, args_model):
    # model save_setting
    if len(args.save_sub_dir) == 0:
        save_sub_dir = args.model_name
        for key in args_model:
            if key != 'model_name':
                save_sub_dir = save_sub_dir + '__' + key + '_' + str(args_model[key])
    else:
        save_sub_dir = args.save_sub_dir

    if len(args.save_label) == 0:
        save_label = str(args.data_type) + '_' +\
            str(args.mask_type) + '_' + str(args.sampling_factor) +\
            ('_' + 'N' + str(args.noise_level) if args.add_noise else '') +\
            '__' + str(args.optimizer) +\
            '__' + 'LR' + '_' + str(args.LR) +\
            '__' + 'wd' + '_' + str(args.weight_decay)
            
        if args.optimizer == 'RMSprop' or args.optimizer == 'SGD':
            save_label += '__' + 'momentum' + '_' + str(args.momentum)
            
        if args.schedule_on:
            save_label += '__' + 'schedule' + '_' + str(args.scheduler)
            
        if args.gradclip:
            save_label += '__' + 'gradclip' + '_' + str(args.gradclip)
        
        save_label += '__' + 'loss' + '_' + str(args.loss)
        
        if args.seed != 42:
            save_label += '__s' + str(args.seed)
            
    return save_sub_dir, save_label
            
def get_repara_savepath(args, args_model):
    #print(args_model)
    if hasattr(args, 'fcb_l'):
        del args_model['fcb_l']
    if hasattr(args, 'fcb_forward'):
        del args_model['fcb_forward']
        
    if 'kernel_size' in args_model.keys():
        del args_model['kernel_size']

    if 'lcb_l' in args_model.keys():
        del args_model['lcb_l']
    if 'lcb_forward' in args_model.keys():
        del args_model['lcb_forward']
    
    args.LR = args.LR_repara
    args.weight_decay = args.weight_decay_repara
    args.seed = 42
    if hasattr(args, 'momentum_repara'):
        args.momentum = args.momentum_repara
    # model save_setting
    save_sub_dir, save_label = get_savepath(args, args_model)
    
    return save_sub_dir, save_label

def save_ckpt(path, epoch, model, best_score, optimizer, scheduler=None):
    """ save current model
    """
    if scheduler is not None:
        torch.save({
            "cur_epoch": epoch,
            "model_state": model.state_dict(),
            "best_score": best_score,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, path)
    else:
        torch.save({
            "cur_epoch": epoch,
            "model_state": model.state_dict(),
            "best_score": best_score,
            "optimizer": optimizer.state_dict()
        }, path)
        
    print("Model saved as %s" % path)    


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def c2r(complex_img, axis=0):
    """
    :input shape: 1 x row x col (complex64)
    :output shape: 2 x row x col (float32)
    """
    if isinstance(complex_img, np.ndarray):
        real_img = np.concatenate((complex_img.real, complex_img.imag), axis=axis)
    elif isinstance(complex_img, torch.Tensor):
        real_img = torch.cat((complex_img.real, complex_img.imag), dim=axis)
    else:
        raise NotImplementedError
    return real_img

def r2c(images, axis=1):
    """
    :input shape: 2c x row x col (float32)
    :output shape: 1c x row x col (complex64)
    """
    C = int(images.shape[axis]/2)
    images = torch.complex(torch.index_select(images, axis, torch.tensor(range(C), device=images.device)),
                           torch.index_select(images, axis, torch.tensor(range(C, images.shape[axis]), device=images.device)))
    return images

def freeze_training(model):
    for param in model.parameters():
        param.requires_grad = False

def set_training(model):
    for param in model.parameters():
        param.requires_grad = True