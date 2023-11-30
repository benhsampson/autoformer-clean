from dataclasses import dataclass


@dataclass
class ModelConfig:
    seq_len: int
    label_len: int
    pred_len: int
