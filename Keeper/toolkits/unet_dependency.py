__all__ = ['module_name', 'module_dependency', 'downsampled_layer',
           'scale_1', 'scale_1_2', 'scale_1_4', 'scale_1_8', 'scale_1_16',]


module_name = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2',
               'conv4_1', 'conv4_2', 'conv5_1', 'conv5_2', 'up6',     'conv6_1',
               'conv6_2', 'up7',     'conv7_1', 'conv7_2', 'up8',     'conv8_1',
               'conv8_2', 'up9',     'conv9_1', 'conv9_2', 'conv10']

scale_1    = ['conv1_1', 'conv1_2', 'up9', 'conv9_1', 'conv9_2', 'conv10']
scale_1_2  = ['conv2_1', 'conv2_2', 'up8', 'conv8_1', 'conv8_2']
scale_1_4  = ['conv3_1', 'conv3_2', 'up7', 'conv7_1', 'conv7_2']
scale_1_8  = ['conv4_1', 'conv4_2', 'up6', 'conv6_1', 'conv6_2']
scale_1_16 = ['conv5_1', 'conv5_2']

module_dependency = {
    'conv1_1': ['conv1_1', 'input'],
    'conv1_2': ['conv1_1', 'conv1_2'],
    'conv2_1': ['conv1_2', 'conv1_2_downsampled', 'conv2_1'],
    'conv2_2': ['conv1_2', 'conv2_1', 'conv2_2'],
    'conv3_1': ['conv1_2', 'conv2_2', 'conv2_2_downsampled', 'conv3_1'],
    'conv3_2': ['conv1_2', 'conv2_2', 'conv3_1', 'conv3_2'],
    'conv4_1': ['conv1_2', 'conv2_2', 'conv3_2', 'conv3_2_downsampled', 'conv4_1'],
    'conv4_2': ['conv1_2', 'conv2_2', 'conv3_2', 'conv4_1', 'conv4_2'],
    'conv5_1': ['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv4_2_downsampled', 'conv5_1'],
    'conv5_2': ['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_1', 'conv5_2'],
    'up6':     ['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2', 'up6'],
    'conv6_1': ['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'up6',     'conv6_1'],
    'conv6_2': ['conv1_2', 'conv2_2', 'conv3_2', 'conv6_1', 'conv6_2'],
    'up7':     ['conv1_2', 'conv2_2', 'conv3_2', 'conv6_2', 'up7'],
    'conv7_1': ['conv1_2', 'conv2_2', 'conv3_2', 'up7',     'conv7_1'],
    'conv7_2': ['conv1_2', 'conv2_2', 'conv7_1', 'conv7_2'],
    'up8':     ['conv1_2', 'conv2_2', 'conv7_2', 'up8'],
    'conv8_1': ['conv1_2', 'conv2_2', 'up8',     'conv8_1'],
    'conv8_2': ['conv1_2', 'conv8_1', 'conv8_2'],
    'up9':     ['conv1_2', 'conv8_2', 'up9'],
    'conv9_1': ['conv1_2', 'up9',     'conv9_1'],
    'conv9_2': ['conv9_1', 'conv9_2'],
    'conv10':  ['conv9_2', 'conv10'],
}

downsampled_layer = ['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']


