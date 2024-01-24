
from dataclasses import dataclass, fields


class DataClassUnpack:
    classFieldCache = {}

    @classmethod
    def instantiate(cls, initing_class, arg_dict):
        if initing_class not in cls.classFieldCache:
            cls.classFieldCache[initing_class] = {f.name for f in fields(initing_class) if f.init}

        fieldSet = cls.classFieldCache[initing_class]
        filtered_arg_dict = {k : v for k, v in arg_dict.items() if k in fieldSet}
        return initing_class(**filtered_arg_dict)


@dataclass
class CFG:
    train_file: str
    test_file: str
    syn_train_path: str
    syn_domains: list
    aug_files: list
    plm: str
    num_epochs: int
    plm_lr: float
    head_lr: float
    weight_decay: float
    dropout: float
    alpha: float
    grad_clip: float
    scheduler: str
    warmup_ratio: float
    num_early_stop: int
    dialog_batch_size: int
    dialog_max_length: int
    syn_batch_size: int
    syn_max_length: int
    num_labels: int
    hidden_size: int
    print_every_ratio: float
    cuda: bool
    fp16: bool
    eval_strategy: str
    mode: str