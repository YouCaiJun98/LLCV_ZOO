def adjust_learning_rate(optimizer, epoch):
    """Sets multi-step LR scheduler."""
    if epoch <= 100:
        lr = 1e-4
    elif epoch <= 180:
        lr = 5e-5
    else:
        lr = 1e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
