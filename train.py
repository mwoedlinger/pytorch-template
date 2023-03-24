__all__ = ['experiment', 'train', 'evaluation']
__author__ = "Matthias Wödlinger"

from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
from spock import SpockBuilder

import src.models as models
import src.optimizers as optimizers
import src.schedulers as schedulers
from src.utils import prepare_run, prepare_wandb, save_model, load_model
from src.dataset import Dataset
from src.transforms import RandomCrop, CenterCrop
from src.config import *
from src.metrics import calc_mse, calc_psnr, calc_bpp
from src import *

def experiment(cfg, cfg_runtime):
    """
    Training initialization and loop.
    """
    cfg_data = cfg.DataConfig
    cfg_experiments = cfg.ExperimentConfig
    cfg_train = cfg.TrainingConfig
    cfg_model = cfg.ModelConfig

    debug = cfg_experiments.debug
    epochs = cfg_train.epochs
    eval_steps = cfg_train.eval_steps
    if debug:
        debug = True
        epochs = 1
        eval_steps = 10
    if cfg_experiments.testing:
        epochs = 0
    device = torch.device(cfg_experiments.device)

    train_transforms = [RandomCrop(256)]
    eval_transforms = [CenterCrop(512)]
    test_transforms = []

    # Init dataloaders
    train_set = Dataset(name=cfg_data.train_name, path=cfg_data.train_path, transforms=train_transforms, debug=debug)
    train_loader = DataLoader(train_set, batch_size=cfg_train.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    train_logger = Logger(cfg_data.train_name)
                              
    eval_set = Dataset(name=cfg_data.eval_name, path=cfg_data.eval_path, transforms=eval_transforms, debug=debug)
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=False)
    eval_logger = Logger(cfg_data.eval_name, maxlen_img=1)
                              
    test_set = Dataset(name=cfg_data.test_name, path=cfg_data.test_path, transforms=test_transforms, debug=debug)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=False)
    test_logger = Logger(cfg_data.test_name, maxlen_img=30)
    

    # initialize model
    model = getattr(models, cfg_model.name)(**cfg_model.kwargs)
    model = model.to(device)

    # init optimizer and scheduler
    optimizer = getattr(optimizers, cfg_train.optimizer_name)(model.parameters(), lr=cfg_train.lr, **cfg_train.optimizer_kwargs)
    scheduler = getattr(schedulers, cfg_train.lr_scheduler_name)(optimizer, **cfg_train.lr_scheduler_kwargs)

    # train model
    step = train(train_loader, eval_loader, train_logger, eval_logger, model, optimizer, scheduler, epochs, eval_steps, device, cfg, cfg_runtime)

    # test model after training
    print(f'\n## TESTING ON {test_logger.prefix} ##')
    with torch.no_grad():
        evaluation(test_loader, model, test_logger, device, cfg, cfg_runtime)
        test_results = test_logger.scal.copy()
        test_logger.log(step)
        
    return test_results

def train(train_loader, eval_loader, train_logger, eval_logger, model, optimizer, scheduler, epochs, eval_steps, device, cfg, cfg_runtime):
    """
    Training loop function.
    """
    cfg_train = cfg.TrainingConfig
    cfg_experiments = cfg.ExperimentConfig

    if cfg_experiments.resume:
        print(f'Resume from {cfg_runtime.resume_path}')
        start_epoch = load_model(cfg_runtime.resume_path, model, optimizer, scheduler)
    else:
        start_epoch = 0

    # TRAINING LOOP
    step = start_epoch * len(train_loader.dataset)
    for epoch in range(start_epoch, epochs):
        print(f'\ntrain epoch {epoch}/{epochs}')

        for img in tqdm(train_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            img = img.to(device) 

            B = img.size(0)
            _train_step(img, model, optimizer, train_logger, step, device, cfg)
            if step % eval_steps < B:
                with torch.no_grad():
                    evaluation(eval_loader, model, eval_logger, device, cfg, cfg_runtime)
                    eval_logger.log(step)
                save_model(cfg_runtime.exp_path, model, optimizer, scheduler, epoch)
            if step % cfg_train.lr_scheduler_drop < B and step > 0:
                save_model(cfg_runtime.exp_path, model, optimizer, scheduler, epoch, suffix=f'_{get_lr(optimizer)}')
                scheduler.step()

            step += B
        train_logger.log(step)

    return step

def _train_step(img, model, optimizer, train_logger, step, device, cfg):
    """
    A single training step
    """
    cfg_experiments = cfg.ExperimentConfig
    cfg_train = cfg.TrainingConfig
    model.train()

    optimizer.zero_grad()

    output = model(img)
    pred, rate, latents = output.pred, output.rate, output.latents

    # Compute Metrics
    mse = calc_mse(img, pred)
    psnr = calc_psnr(mse, eps=cfg_experiments.eps)

    # Computer BPP
    bpp_y = calc_bpp(rate.y, img)
    bpp_z = calc_bpp(rate.z, img)
    bpp = bpp_y + bpp_z

    loss = (bpp + cfg_train.lmda * mse) / (1 + cfg_train.lmda)

    # Backward - optimize
    loss.mean().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.50)
    optimizer.step()

    # Log scalars
    train_logger.scalars(loss=loss, bpp=bpp, bpp_y=bpp_y, bpp_z=bpp_z, mse=mse, psnr=psnr, lr=get_lr(optimizer))


def evaluation(eval_loader, model, eval_logger, device, cfg, cfg_runtime):
    """
    Evalution loop function.
    """
    for img in tqdm(eval_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        img = img.to(device)
        _evaluation_step(img, model, eval_logger, cfg)
        
    log = [f'    ## {eval_logger.prefix} averages:']
    for name, vals in eval_logger.scal.items():
        log.append(f'    {name:10}: {np.mean(vals):.4}')
    log_str = '\n' + '\n'.join(log)

    with open(Path(cfg_runtime.exp_path) / 'log.txt', 'a') as f:
        f.write(log_str)
        print(log_str)

def _evaluation_step(img, model, eval_logger, cfg):
    """
    A single evaluation step
    """
    cfg_experiments = cfg.ExperimentConfig
    cfg_train = cfg.TrainingConfig
    model.eval()

    output = model(img)
    pred, rate, latents = output.pred, output.rate, output.latents

    # Compute Metrics
    mse = calc_mse(img, pred)
    psnr = calc_psnr(mse, eps=cfg_experiments.eps)

    # Computer BPP
    bpp_y = calc_bpp(rate.y, img)
    bpp_z = calc_bpp(rate.z, img)
    bpp = bpp_y + bpp_z

    loss = (bpp + cfg_train.lmda * mse) / (1 + cfg_train.lmda)

    # Log scalars
    eval_logger.scalars(loss=loss, bpp=bpp, bpp_y=bpp_y, bpp_z=bpp_z, mse=mse, psnr=psnr)
    eval_logger.image(torch.cat([pred[0], img[0]], dim=-1), caption=f'psnr={psnr:.4}, bpp={bpp:.4}')


if __name__ == "__main__":
    description = ''
    cfg = SpockBuilder(WandbConfig, TrainingConfig, ModelConfig, DataConfig, ExperimentConfig, desc=description).generate()

    cfg_runtime = prepare_run(config=cfg, root=Path(__file__).parent)
    experiment(cfg, cfg_runtime)
