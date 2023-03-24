import torch
import wandb
import numpy as np
from os import system, getcwd
from sys import exit
from time import time
from pathlib import Path
from shutil import copytree, copy
import os
import yaml
import json
from spock import SpockBuilder
from .config import *

__all__ = ['save_model', 'load_model', 'get_lr', 'prepare_wandb', 'prepare_run']

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

class Dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)


def ToDict(dictionary: dict):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = ToDict(value)

    return Dict(dictionary)

def tictoc():
    """
    Returns time in seconds since the last time the function was called.
    For the initial call 0 is returned.
    """
    if not hasattr(tictoc, 'tic'):
        tictoc.tic = time()

    toc = time()
    dt = toc - tictoc.tic
    tictoc.tic = toc

    return dt

def save_model(path, model, optimizer, scheduler, epoch, suffix=''):
    state_dict_model = model.state_dict()
    state_dict_training = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    print(f'save model in {path}')
    model_name = f'model{suffix}.pt'
    training_state_name = f'training_state{suffix}.pt'

    torch.save(state_dict_model, str(Path(path) / model_name))
    torch.save(state_dict_training, str(Path(path) / training_state_name))


def load_model(path, model, optimizer=None, scheduler=None, device=torch.device('cpu')):
    checkpoint_model = torch.load(str(Path(path) / 'model.pt'), map_location=device)
    checkpoint_training = torch.load(str(Path(path) / 'training_state.pt'), map_location=device)

    model.load_state_dict(checkpoint_model)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_training['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint_training['scheduler'])

    return checkpoint_training['epoch']

def copy_files(exp_path, root='.', src='src', data='data'):
    dst = Path(exp_path) / 'src'
    dst.mkdir()

    copytree(src, dst / src)
    copytree(data, dst / data)

    print(data)
    print(dst / data)

    remaining_files = [f for f in Path(root).iterdir() if f.suffix == '.py']

    for file in remaining_files:
        file_dst = dst / file.name
        copy(file, file_dst)

def save_config(config, path: Path):
    yaml.dump(config, path)

def load_runtime_config(path: Path):
    return ToDict(json.load(path))

def save_runtime_config(config, path: Path):
    with open(path, 'w') as f:
        json.dump(config, f)

def pprint(c, indent=0):
    if indent == 0:
        print('-'*45)

    if c.keys():
        offset = max([len(k) for k in c.keys()] + [15]) + 2

    for k, v in c.items():
        spaces = offset - len(str(k)) - 1
        if isinstance(v, Dict) or isinstance(v, dict):
            print(' '*indent + str(k) + ':')
            pprint(v, indent + offset)
        elif isinstance(v, list):
            print(' '*indent + str(k) + ':' + ' '*spaces + '[')
            for l in v:
                print(' '*(indent + offset + 1) + str(l))
            print(' '*(indent + offset) + ']')
        else:
            print(' '*indent + str(k) + ':' + ' '*spaces + str(v))
    if indent == 0:
        print('-'*45)

def prepare_wandb(config, root: Path):
    cfg_experiment = config.ExperimentConfig
    cfg_wandb = config.WandbConfig

    if cfg_experiment.resume:
        cfg_runtime = load_runtime_config(config, root / 'cfg_runtime.json', True)
    elif cfg_experiment.resume_with_new_exp:
        cfg_runtime = load_runtime_config(config, root / 'cfg_runtime.json', False)
        cfg_runtime['run_id'] = wandb.util.generate_id()
    else:
        cfg_runtime = Dict({'run_id': wandb.util.generate_id()})

    # Set wandb logging and cache dir
    wandb_cache_dir = Path(cfg_experiment.log_dir) / Path('wandb') / Path('.cache')
    wandb_dir = Path(cfg_experiment.log_dir) / Path('wandb') / cfg_wandb.project / cfg_wandb.group / cfg_runtime.run_id
    if not Path(wandb_dir).is_dir():
        Path(wandb_dir).mkdir(parents=True)

    assert Path(wandb_cache_dir).is_dir(), f'wandb_cache_dir {wandb_cache_dir} does not exist/is no dir.'
    assert Path(wandb_dir).is_dir(), f'wandb_dir {wandb_dir} does not exist/is no dir.'

    os.environ['WANDB_CACHE_DIR'] = str(wandb_cache_dir)
    os.environ['WANDB_DIR'] = str(wandb_dir)

    return cfg_runtime

def init_wandb(config, cfg_runtime, root):
    cfg_wandb = config.WandbConfig
    wandb.init(group=cfg_wandb.group, project=cfg_wandb.project, entity=cfg_wandb.entity, 
               tags=cfg_wandb.tags, config={**dict(config), **cfg_runtime}, id=cfg_runtime.run_id, resume="allow")
    wandb.run.log_code(root)
    wandb.run.name = cfg_runtime.run_id

def create_exp_path(log_dir, run_id, root):
    exp_path = str(Path(log_dir) / Path('experiments') / str(run_id))
    Path(exp_path).mkdir(exist_ok=False, parents=True)
    copy_files(exp_path, root)

    return exp_path

def prepare_run(config, root: Path):
    cfg_experiment = config.ExperimentConfig
    cfg_runtime = prepare_wandb(config, root)

    assert Path(cfg_experiment.log_dir).is_dir(), f'log_dir {cfg_experiment.log_dir} does not exist/is no dir.'
    assert not (cfg_experiment.resume and cfg_experiment.resume_with_new_exp), f'resume and resume_with_new_exp cannot be turned on at the same time.'

    if cfg_experiment.resume or cfg_experiment.resume_with_new_exp:
        cfg_runtime.resume_path = root

    if not cfg_experiment.resume:
        cfg_runtime.exp_path = create_exp_path(cfg_experiment.log_dir, cfg_runtime.run_id, root)
        
    # fix random seeds for reproducibility
    torch.manual_seed(cfg_experiment.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(cfg_experiment.seed)

    print('cfg_runtime:')
    pprint(dict(cfg_runtime))
    print('\n')

    save_runtime_config(cfg_runtime, Path(cfg_runtime.exp_path) / 'cfg_runtime.json')
    SpockBuilder(WandbConfig, TrainingConfig, ModelConfig, DataConfig, ExperimentConfig, desc='').save(user_specified_path=Path(cfg_runtime.exp_path), file_name='config.yaml')

    init_wandb(config, cfg_runtime, root)

    return cfg_runtime