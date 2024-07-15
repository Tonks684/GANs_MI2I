import sys
import os
from pathlib import Path

model_name = f'dlmbl_vscyto' 
dataroot = Path('../../')
output_dir = Path('../../results/')
python_file = Path('../../pix2pixHD/train_tesaro.py')
# Path for chosen training file see source code for explanation of files to choose from.
GAN_config = {}
GAN_config['--dataroot'] = dataroot
GAN_config['--data_type'] = '16'
GAN_config['--batchSize'] = '4'
GAN_config['--checkpoints_dir'] = os.path.join(output_dir,f'{model_name}') 
GAN_config['--label_nc'] = '0'
GAN_config['--name'] = f'{model_name}'
GAN_config['--no_instance'] = ''
GAN_config['--resize_or_crop'] = 'none'
GAN_config['--input_nc'] = '1'
GAN_config['--output_nc'] = '1'
GAN_config['--seed'] = '42'

GAN_config['--no_vgg_loss'] = ''
GAN_config['--nThreads'] = '1'
# GAN_config['--gpu_ids'] = '0'
GAN_config['--loadSize'] = '256'
GAN_config['--ndf'] = '32'
GAN_config['--norm'] = 'instance'
GAN_config['--use_dropout'] = ''
GAN_config['--fp16'] = '' 

## Used only is retraining from epoch
# GAN_config['--continue_train'] = ''
# GAN_config['--which_epoch'] = 'latest'

command = f"python {python_file}"
for key, value in GAN_config.items():
    command += f" {key} {value}"
print(command)

