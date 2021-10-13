from typing import Optional
from typing import Tuple

import attr


@attr.s(auto_attribs=True)
class SlotAttentionParams:
    lr: float = 0.00005
    batch_size: int = 8
    val_batch_size: int = 8
    resolution: Tuple[int, int] = (64, 64)
    num_slots: int = 2
    num_iterations: int = 3
    data_root: str = "/kuacc/users/ashah20/object-centric-representation-benchmark/ocrb/data/datasets/"
    #data_root: str = "/datasets/COCO/"
    gpus: int = 1
    max_epochs: int = 100
    num_sanity_val_steps: int = 1
    scheduler_gamma: float = 0.5
    weight_decay: float = 0.0
    num_train_images: Optional[int] = 54
    num_val_images: Optional[int] = 24
    empty_cache: bool = True
    is_logger_enabled: bool = True
    is_verbose: bool = True
    num_workers: int = 4
    n_samples: int = 1
    warmup_steps_pct: float = 0.02
    decay_steps_pct: float = 0.2
    n_steps: int = 10
    dataset_class: str = "vmds"
    T: int = 5
