from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler

def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.NUM_EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.SCHEDULER.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(config.SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)

    scheduler = None
    if config.SCHEDULER.NAME == 'cosine':
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=(num_steps - warmup_steps) if config.SCHEDULER.WARMUP_PREFIX else num_steps,
            lr_min=config.SCHEDULER.MIN_LR,
            warmup_lr_init=config.SCHEDULER.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=config.SCHEDULER.WARMUP_PREFIX,
        )
    elif config.TRAIN.SCHEDULER.NAME == 'step':
        scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.SCHEDULER.DECAY_RATE,
            warmup_lr_init=config.SCHEDULER.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return scheduler
