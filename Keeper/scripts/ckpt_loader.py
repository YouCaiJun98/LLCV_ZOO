import torch
import models.unet_denoise as LSID
import utils
import datasets
#from trainer import Trainer, Validator

torch.cuda.set_device(3)
device = torch.device('cuda')
model = LSID.lsid(inchannel=4, outchannel=4)
path = 'checkpoints/Sony_baseline_SID-train-20220113-174826/best_ssim.pth.tar'
checkpoint = torch.load(path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()

torch.save(model.state_dict(), 'pure_params.pth.tar')
'''
dataset_class = datasets.__dict__['Sony']
dt = dataset_class('./datasets/Sony', './datasets/Sony/Sony_test_list.txt',
                   split='test',gt_png=False, use_camera_wb=False, upper=-1)
test_loader = torch.utils.data.DataLoader(dt, batch_size=1, shuffle=False,
                                          num_workers=4, pin_memory=True)

criterion = torch.nn.L1Loss().cuda()
log_file = 'test.log'
validator = Validator(cmd='test', cuda=torch.cuda.is_available() , model=model,
                      criterion=criterion, val_loader=test_loader, log_file=log_file,
                      result_dir=None, use_camera_wb=False, print_freq=1)
validator.validate()
'''
