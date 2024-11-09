import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(optimizer, config):
    scheduler_type = config['type'].lower()

    if scheduler_type == 'step':
        step_size = config['step_size']
        gamma = config['gamma']
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        T_max = config['T_max']
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == 'cosine_restart':
        T_0 = config['T_0']
        T_mult = config.get('T_mult', 1)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)
    else:
        raise NotImplementedError(f"Scheduler type {scheduler_type} not supported.")

    return scheduler
