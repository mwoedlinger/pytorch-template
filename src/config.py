from spock import spock
from typing import List, Dict, Optional

@spock
class WandbConfig:
    entity: str
    project: str
    group: str
    tags: List[str]

@spock
class ExperimentConfig:
    device: str
    seed: Optional[int]=0
    log_dir: Optional[str]='/data/mwoedlinger/logging/'
    resume: Optional[bool]=False
    resume_with_new_exp: Optional[bool]=False
    testing: Optional[bool]=False
    debug: Optional[bool]=False
    eps: Optional[bool]=1e-9

    run_id: Optional[str]=None
    config_path: Optional[str]
    resume_path: Optional[str]=None
    exp_path: Optional[str]=None

@spock
class TrainingConfig:
    epochs: int
    batch_size: int
    lr: float
    eval_steps: Optional[int]=20000
    lmda: float

    optimizer_name: Optional[str]='Adam'
    optimizer_kwargs: Optional[Dict[str, float]]={} # Without lr

    lr_scheduler_name: Optional[str]=None
    lr_scheduler_kwargs: Optional[Dict[str, float]]={}
    lr_scheduler_drop: int

@spock
class ModelConfig:
    name: str
    kwargs: Dict[str, int]

@spock
class DataConfig:
    train_path: str
    train_name: str
    
    eval_path: str
    eval_name: str

    test_path: str
    test_name: str
    
