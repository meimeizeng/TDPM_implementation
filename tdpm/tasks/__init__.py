from .base import Batch, BaseTask
from .deblur import DeblurTask
from .inpaint import InpaintTask
from .sr import SuperResolutionTask

TASKS = {
    "sr": SuperResolutionTask,
    "inpaint": InpaintTask,
    "deblur": DeblurTask,
}


def build_task(cfg, device):
    if cfg.task not in TASKS:
        raise ValueError(f"Unknown task '{cfg.task}'. Available: {sorted(TASKS)}")
    return TASKS[cfg.task](cfg, device)
