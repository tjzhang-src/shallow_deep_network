# train_networks.py
# For training CNNs and SDNs via IC-only and SDN-training strategies
# It trains and save the resulting models to an output directory specified in the main function

import copy
import argparse
import torch
import time
import os
import random
import numpy as np

import aux_funcs  as af
import network_architectures as arcs

from architectures.CNNs.VGG import VGG



from typing import Optional

def train(models_path, untrained_models, sdn=False, ic_only_sdn=False, device='cpu', scheduler_type: str = 'multistep',
          resume: bool = False, resume_epochs: Optional[int] = None, resume_lr: Optional[float] = None):
    print('Training models...')

    for base_model in untrained_models:
        # Load last checkpoint if resume is requested; otherwise load untrained (epoch=0)
        if resume:
            try:
                trained_model, model_params = arcs.load_model(models_path, base_model, -1)
                print(f"[resume] Loaded last checkpoint for {base_model}")
            except Exception:
                trained_model, model_params = arcs.load_model(models_path, base_model, 0)
                print(f"[resume] No last checkpoint for {base_model}, starting from untrained")
        else:
            trained_model, model_params = arcs.load_model(models_path, base_model, 0)
        dataset = af.get_dataset(model_params['task'])

        learning_rate = model_params['learning_rate'] if resume_lr is None else resume_lr
        momentum = model_params['momentum']
        weight_decay = model_params['weight_decay']
        milestones = model_params['milestones']
        gammas = model_params['gammas']
        num_epochs = resume_epochs if resume_epochs is not None else model_params['epochs']

        model_params['optimizer'] = 'SGD'

        if ic_only_sdn:  # IC-only training, freeze the original weights
            learning_rate = model_params['ic_only']['learning_rate']
            num_epochs = model_params['ic_only']['epochs']
            milestones = model_params['ic_only']['milestones']
            gammas = model_params['ic_only']['gammas']

            model_params['optimizer'] = 'Adam'
            
            # mark model as IC-only mode (dynamic flag used by training function)
            try:
                setattr(trained_model, 'ic_only', True)
            except Exception:
                pass


        optimization_params = (learning_rate, weight_decay, momentum)
        lr_schedule_params = (milestones, gammas)

        if sdn:
            if ic_only_sdn:
                optimizer, scheduler = af.get_sdn_ic_only_optimizer(trained_model, optimization_params, lr_schedule_params)
                trained_model_name = base_model+'_ic_only'

            else:
                if scheduler_type == 'cosine':
                    optimizer, scheduler = af.get_full_optimizer_cosine(trained_model, optimization_params, num_epochs)
                else:
                    optimizer, scheduler = af.get_full_optimizer(trained_model, optimization_params, lr_schedule_params)
                trained_model_name = base_model+'_sdn_training'

        else:
                if scheduler_type == 'cosine':
                    optimizer, scheduler = af.get_full_optimizer_cosine(trained_model, optimization_params, num_epochs)
                else:
                    optimizer, scheduler = af.get_full_optimizer(trained_model, optimization_params, lr_schedule_params)
                trained_model_name = base_model

        print('Training: {}...'.format(trained_model_name))
        if resume:
            print(f"[resume] epochs this run: {num_epochs}, lr: {learning_rate}")
        trained_model.to(device)
        metrics = trained_model.train_func(trained_model, dataset, num_epochs, optimizer, scheduler, device=device)
        # Merge metrics when resuming; otherwise overwrite
        def _merge(key):
            if resume and isinstance(model_params.get(key), list):
                model_params[key] = model_params.get(key, []) + metrics.get(key, [])
            else:
                model_params[key] = metrics.get(key, [])
        for k in ['train_top1_acc','test_top1_acc','train_top5_acc','test_top5_acc','epoch_times','lrs']:
            _merge(k)
        # Update epochs to cumulative length if resuming
        if resume:
            try:
                model_params['epochs'] = len(model_params.get('train_top1_acc', []))
            except Exception:
                pass
        total_training_time = sum(model_params['epoch_times'])
        model_params['total_time'] = total_training_time
        print('Training took {} seconds...'.format(total_training_time))
        arcs.save_model(trained_model, model_params, models_path, trained_model_name, epoch=-1)

def train_sdns(models_path, networks, ic_only=False, device='cpu', scheduler_type: str = 'multistep',
               resume: bool = False, resume_epochs: Optional[int] = None, resume_lr: Optional[float] = None):
    if ic_only: # if we only train the ICs, we load a pre-trained CNN
        load_epoch = -1
    else: # if we train both ICs and the orig network, we load an untrained CNN
        load_epoch = 0

    if resume:
        # Directly resume training existing SDNs
        train(models_path, networks, sdn=True, ic_only_sdn=ic_only, device=device, scheduler_type=scheduler_type,
              resume=True, resume_epochs=resume_epochs, resume_lr=resume_lr)
    else:
        for sdn_name in networks:
            cnn_to_tune = sdn_name.replace('sdn', 'cnn')
            sdn_params = arcs.load_params(models_path, sdn_name)
            sdn_params = arcs.get_net_params(sdn_params['network_type'], sdn_params['task'])
            sdn_model, _ = af.cnn_to_sdn(models_path, cnn_to_tune, sdn_params, load_epoch) # load the CNN and convert it to a SDN
            arcs.save_model(sdn_model, sdn_params, models_path, sdn_name, epoch=0) # save the resulting SDN
        train(models_path, networks, sdn=True, ic_only_sdn=ic_only, device=device, scheduler_type=scheduler_type)


def train_models(models_path, device='cpu', tasks = ['tinyimagenet','cifar10', 'cifar100'], scheduler_type: str = 'multistep',
                 resume_cnn: bool = False, resume_sdn: bool = False, resume_epochs: Optional[int] = None, resume_lr: Optional[float] = None):
    

    cnns = []
    sdns = []

    for task in tasks:
        af.extend_lists(cnns, sdns, arcs.create_vgg16bn(models_path, task, save_type='cd'))
        af.extend_lists(cnns, sdns, arcs.create_resnet56(models_path, task, save_type='cd'))
        af.extend_lists(cnns, sdns, arcs.create_wideresnet32_4(models_path, task, save_type='cd'))
        af.extend_lists(cnns, sdns, arcs.create_mobilenet(models_path, task, save_type='cd'))

    train(models_path, cnns, sdn=False, device=device, scheduler_type=scheduler_type,
        resume=resume_cnn, resume_epochs=resume_epochs, resume_lr=resume_lr)
    #train_sdns(models_path, sdns, ic_only=True, device=device, scheduler_type=scheduler_type) # train SDNs with IC-only strategy
    train_sdns(models_path, sdns, ic_only=False, device=device, scheduler_type=scheduler_type,
           resume=resume_sdn, resume_epochs=resume_epochs, resume_lr=resume_lr) # train SDNs with SDN-training strategy


# for backdoored models, load a backdoored CNN and convert it to an SDN via IC-only strategy
def sdn_ic_only_backdoored(device):
    params = arcs.create_vgg16bn(None, 'cifar10', None, True)

    path = 'backdoored_models'
    backdoored_cnn_name = 'VGG16_cifar10_backdoored'
    save_sdn_name = 'VGG16_cifar10_backdoored_SDN'

    # Use the class VGG
    backdoored_cnn = VGG(params)
    backdoored_cnn.load_state_dict(torch.load('{}/{}'.format(path, backdoored_cnn_name), map_location='cpu'), strict=False)

    # convert backdoored cnn into a sdn
    backdoored_sdn, sdn_params = af.cnn_to_sdn(None, backdoored_cnn, params, preloaded=backdoored_cnn) # load the CNN and convert it to a sdn
    arcs.save_model(backdoored_sdn, sdn_params, path, save_sdn_name, epoch=0) # save the resulting sdn

    networks = [save_sdn_name]

    train(path, networks, sdn=True, ic_only_sdn=True, device=device)

    
def main():
    parser = argparse.ArgumentParser(description='Train CNNs/SDNs with optional IC-only/SDN-training strategies')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides default in aux_funcs)')
    parser.add_argument('--device', type=str, default=None, help="Torch device string, e.g., 'cpu', 'cuda', 'cuda:0'")
    parser.add_argument('--tasks', type=str, nargs='+', default=None, help="Tasks to train, e.g., --tasks cifar10 cifar100 tinyimagenet")
    parser.add_argument('--scheduler', type=str, choices=['multistep','cosine'], default='cosine', help='LR scheduler type')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor for CE loss (0 disables)')
    parser.add_argument('--resume_cnn', action='store_true', help='Resume CNN training from last checkpoints if available')
    parser.add_argument('--resume_sdn', action='store_true', help='Resume SDN training from last checkpoints if available')
    parser.add_argument('--resume_epochs', type=int, default=None, help='Number of epochs to run when resuming (optional)')
    parser.add_argument('--resume_lr', type=float, default=None, help='Learning rate override when resuming (optional)')
    args = parser.parse_args()

    # Seed handling
    if args.seed is not None:
        af.set_random_seeds(args.seed)
        random_seed = args.seed
    else:
        random_seed = af.get_random_seed()
        af.set_random_seeds()
    print('Random Seed: {}'.format(random_seed))

    # Device handling
    if args.device is not None:
        device = args.device
    else:
        device = af.get_pytorch_device()

    # Tasks handling
    tasks = args.tasks if args.tasks is not None else ['tinyimagenet'] #,'cifar10', 'cifar100'

    # Loss configuration
    af.set_label_smoothing(args.label_smoothing)

    models_path = 'networks/{}'.format(af.get_random_seed())
    af.create_path(models_path)
    af.set_logger('outputs/train_models'.format(af.get_random_seed()))

    train_models(models_path, device, tasks, scheduler_type=args.scheduler,
                 resume_cnn=args.resume_cnn, resume_sdn=args.resume_sdn,
                 resume_epochs=args.resume_epochs, resume_lr=args.resume_lr)  # e.g., ['tinyimagenet','cifar10', 'cifar100']
    #sdn_ic_only_backdoored(device)

if __name__ == '__main__':
    main()
