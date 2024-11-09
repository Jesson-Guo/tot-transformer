import torch.optim as optim

def get_optimizer(model, state_evaluator, config):
    optimizer_type = config['type'].lower()
    lr = config['lr']
    weight_decay = config['weight_decay']
    momentum = config.get('momentum', 0.9)

    # Combine parameters from both models
    params = list(model.parameters()) + list(state_evaluator.parameters())

    if optimizer_type == 'adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"Optimizer type {optimizer_type} not supported.")

    return optimizer
