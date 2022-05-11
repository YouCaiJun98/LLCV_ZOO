from copy import deepcopy
def adjust_learning_rate(optimizer, epoch, scheduler):
    """Sets multi-step LR scheduler."""
    scheduler = deepcopy(scheduler)
    assert -1 in scheduler.keys(), "Default lr is not given!"
    default_lr = scheduler.pop(-1)
    milestones = [key for key in scheduler.keys()]
    lr_list = [lr for lr in scheduler.values()]
    for i, milestone in enumerate(milestones):
        if epoch < milestone:
            lr = lr_list[i]
            break
    if epoch >= milestones[-1]:
        lr = default_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
