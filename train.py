import numpy as np
import torch
import torch.nn as nn
import time
import os
import argparse
import shutil
import copy
import collections
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from dataset import MyDataset
from utils import *

def train(model, train_loader, optimizer, scheduler, loss_func, epoch, args):
    running_loss = 0.
    for batch_idx, batch in enumerate(train_loader):
        if args.model_name in ['UNet', 'FUNet']:
            data = torch.abs(batch['input_multi']).to(device=args.device)
            target = batch['target_single'].to(device=args.device)
            output = model(data)
            
        elif args.model_name in ['MoDL', 'FMoDL']:
            zf = c2r(batch['input_single'], axis=1).to(device=args.device)
            mask = batch['mask'].to(device=args.device)[:1]
            target = batch['target_single'].to(device=args.device)
            map = batch['sensmap'].to(device=args.device)
            
            output = model(zf, map, mask)
            output = torch.norm(output, dim=1, keepdim=True)

        elif args.model_name in ['VSNet', 'FVSNet']:
            zf = c2r(batch['input_single'], axis=1).to(device=args.device)
            mask = batch['mask'].to(device=args.device)[:1]
            measure = batch['measure'].to(device=args.device)
            target = batch['target_single'].to(device=args.device)
            map = batch['sensmap'].to(device=args.device)
            
            output = model(zf, measure, mask, map)
            output = torch.norm(output, dim=1, keepdim=True)
                
        elif args.model_name in ['VarNet', 'FVarNet']:
            target = batch['target_single'].to(device=args.device)
            mask = batch['mask'].to(device=args.device)[:1]
            measure = batch['measure'].to(device=args.device)
            
            measure = torch.stack((measure.real, measure.imag), dim=-1)
            mask = mask.unsqueeze(-1).bool()
            output = model(measure, mask, args.num_low_frequencies)
            output = output.unsqueeze(1)
        else:
            raise(ValueError)

        loss = loss_func(output, target)
        running_loss += loss.item()
        if batch_idx % args.vis_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(target), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data))
            
        optimizer.zero_grad()
        loss.backward()
        if args.gradclip:
            nn.utils.clip_grad_norm_(model.parameters(),
                                     max_norm=args.gradclip, norm_type=2)
        optimizer.step()
        if scheduler is not None and not args.epochstep:
            if args.last_decay < 0 or \
                args.last_decay < batch_idx + epoch*len(train_loader):
                scheduler.step()
                
    # scheduler by epoch
    if scheduler is not None and args.epochstep:
        scheduler.step()
    
    running_loss = running_loss / len(train_loader)
    print('Train set: Average loss: {:.4f}'.format(running_loss))
    
    return running_loss
    
    
def val(model, val_loader, loss_func, args):
    val_loss = 0.
    criterion_output = {"nmse": 0., "psnr": 0., "ssim": 0.}
    for batch_idx, batch in enumerate(val_loader):
        with torch.no_grad():
            if args.model_name in ['UNet', 'FUNet']:
                data = abs(batch['input_multi']).to(device=args.device)
                target = batch['target_single'].to(device=args.device)
                output = model(data)
                
            elif args.model_name in ['MoDL', 'FMoDL']:
                zf = c2r(batch['input_single'], axis=1).to(device=args.device)
                mask = batch['mask'].to(device=args.device)[:1]
                target = batch['target_single'].to(device=args.device)
                map = batch['sensmap'].to(device=args.device)

                output = model(zf, map, mask)
                output = torch.norm(output, dim=1, keepdim=True)
                
            elif args.model_name in ['VSNet', 'FVSNet']:
                zf = c2r(batch['input_single'], axis=1).to(device=args.device)
                mask = batch['mask'].to(device=args.device)[:1]
                measure = batch['measure'].to(device=args.device)
                target = batch['target_single'].to(device=args.device)
                map = batch['sensmap'].to(device=args.device)
                
                output = model(zf, measure, mask, map)
                output = torch.norm(output, dim=1, keepdim=True)
                    
            elif args.model_name in ['VarNet', 'FVarNet']:
                target = batch['target_single'].to(device=args.device)
                mask = batch['mask'].to(device=args.device)[:1]
                measure = batch['measure'].to(device=args.device)
                
                measure = torch.stack((measure.real, measure.imag), dim=-1)
                mask = mask.unsqueeze(-1).bool()
                output = model(measure, mask, args.num_low_frequencies)
                output = output.unsqueeze(1)
                    
            else:
                raise(ValueError)
            
            loss = loss_func(output, target)
            val_loss += loss.item()
            
            full_i = target[0, 0].cpu().numpy()
            out_i = output[0, 0].cpu().numpy()
            
            
        mse = np.mean((out_i - full_i) ** 2)
        norms = np.mean(full_i ** 2)
        criterion_output["nmse"] += mse / norms
        criterion_output["psnr"] += peak_signal_noise_ratio(out_i, full_i, data_range=full_i.max())
        criterion_output["ssim"] += structural_similarity(out_i, full_i, win_size=11, data_range=full_i.max())
    
    val_loss = val_loss / len(val_loader)
    print('Test set: Average loss: {:.4f}'.format(val_loss))
    for key in criterion_output.keys():
        criterion_output[key] = criterion_output[key] / len(val_loader)
    
    return criterion_output, val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training command parser', add_help=False)
    parser.add_argument('-c', '--config_path', default='../config/unet.yaml',
                        type=str, help='config path')
    cmd_par =  parser.parse_args()
    
    config_path = cmd_par.config_path
    args, args_data, args_model, args_training, _ = get_args(config_path)
    
    random_seed = args.seed
    set_random_seed(random_seed, True)
    
    train_dataset = MyDataset(directory_full=args.directory_full_train,
                              directory_m=args.directory_m_train,
                              mask_path=args.mask_path,
                              label_index=args.label_index_train,
                              noise_=args.add_noise,
                              noise_level=args.noise_level,
                              shift=args.shift,
                              return_shift=args.return_shift)
    
    val_dataset = MyDataset(directory_full=args.directory_full_val,
                            directory_m=args.directory_m_val,
                            mask_path=args.mask_path,
                            label_index=args.label_index_val,
                            noise_=args.add_noise,
                            noise_level=args.noise_level,
                            shift=args.shift,
                            return_shift=args.return_shift)
    
    
    g = torch.Generator()
    g.manual_seed(random_seed)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=16,
                                               worker_init_fn=seed_worker,
                                               generator=g,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=1,
                                             num_workers=16,
                                             worker_init_fn=seed_worker,
                                             generator=g,
                                             shuffle=False)
    
    
    model = get_model(args).to(args.device)
    if args.model_name == 'MoDL' and 0 < args.pre_train_epoches:
        model.k_iters = 1
    
    if args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.LR, alpha=0.99,
                                        eps=1e-08, weight_decay=args.weight_decay,
                                        momentum=args.momentum, centered=False)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, betas=(0.9, 0.999),
                                        eps=1e-08, weight_decay=args.weight_decay,
                                        amsgrad=False)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.LR, momentum=args.momentum,
                                    dampening=0, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR, betas=(0.9, 0.999),
                                         eps=1e-08, weight_decay=args.weight_decay,
                                         amsgrad=False)
    else:
        raise(ValueError)
    
    # finding saved model path for reparameterization
    if args.repara:
        args_repara, args_model_repara = copy.deepcopy(args), copy.deepcopy(args_model)
        if args_repara.model_name[0] == 'F':
            args_repara.model_name = args_repara.model_name[1:]
        if args_repara.kernel_size > 3:
            args_repara.kernel_size = 3
            args_repara.lcb_l = 0
            args_repara.lcb_forward = False
            
        model_repara = get_model(args_repara).to(args.device)
        if not hasattr(args, 'repara_path'):
            save_sub_dir, save_label = get_repara_savepath(args_repara, args_model_repara)
            save_path = os.path.join(args_repara.save_dir, save_sub_dir, save_label)
            args.repara_path = save_path
            
        print('RePara from : ', args.repara_path+'_alllast_.pth')
        model_repara.load_state_dict(torch.load(args.repara_path+'_alllast_.pth'))
        model.repara(model_repara)
        del model_repara
        
    if args.schedule_on:
        if args.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=args.step_size,
                                                        gamma = args.gamma)
        elif args.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=args.milestones,
                                                             gamma = args.gamma)
        elif args.scheduler == 'ExponentialLR':
            gamma = np.power(1e-1, args.last_decay)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                               gamma=gamma,
                                                               last_epoch=-1,)
        elif args.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=args.epoches)
        elif args.scheduler == 'OneCycleStepLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.LR_max,
                                                            total_steps=args.epoches, pct_start=4.0/args.epoches,
                                                            anneal_strategy='linear',
                                                            cycle_momentum=False,
                                                            base_momentum=0., max_momentum=0.,
                                                            div_factor=0.1*args.epoches, final_div_factor=9)
                        
    else:
        scheduler = None
    
    if args.loss == 'L1':
        loss_func = nn.L1Loss()
    elif args.loss == 'MSE':
        loss_func = nn.MSELoss()
    else:
        raise(ValueError)
        
    save_sub_dir, save_label = get_savepath(args, args_model)
    
    save_path = os.path.join(args.save_dir, save_sub_dir, save_label)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    print('Export path: ', save_path)
    shutil.copyfile(config_path, save_path + '.yaml')
    
    start_epoch = 0
    best_score = 1000000000000.
    if args.resume_training:
        print('Loading: ', save_path + '_last_.pth')
        checkpoint = torch.load(save_path + '_last_.pth')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['cur_epoch'] + 1
        best_score = checkpoint['best_score']
        
    for epoch in range(start_epoch, args.epoches):
        ### pretrian setting iters=10
        if args.model_name == 'MoDL' and epoch == args.pre_train_epoches:
            model.k_iters = args.k_iters
        
        if args.model_name == 'KIKINet' and epoch % args.epoches_seg == 0:
            optimizer.state = collections.defaultdict(dict)
        
        epoch_start = time.time()
        model.train()
        loss_train = train(model, train_loader, optimizer, scheduler, loss_func, epoch, args)
        model.eval()
        criterion, loss_val = val(model, val_loader, loss_func, args)
        nmse, psnr, ssim = criterion["nmse"], criterion["psnr"], criterion["ssim"]
        
        f = open(save_path + '_metric.txt', 'a')
        f.write(f'epoch:{epoch}, nmse:{nmse}, psnr:{psnr}, ssim:{ssim}, trainloss:{loss_train}, valloss:{loss_val}')
        f.write('\n')
        f.close()
        if nmse < best_score:  # save best model
            best_score = nmse
            save_ckpt(save_path + '_best_.pth', epoch, model, best_score, optimizer, scheduler)
        print(f"time cost: {time.time()-epoch_start} s")
        save_ckpt(save_path + '_last_.pth', epoch, model, best_score, optimizer, scheduler)
        print(f"time cost: {time.time()-epoch_start} s")

    torch.save(model.state_dict(), save_path + '_alllast_.pth')
    print('over')