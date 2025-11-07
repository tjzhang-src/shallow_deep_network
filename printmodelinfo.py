import torch
from network_architectures import create_vgg16bn
from architectures.SDNs.VGG_SDN import VGG_SDN
from profiler import print_sdn_model_and_convfc_flops

# 构建 CIFAR-10 的 VGG16 SDN 未训练模型
params = create_vgg16bn(None, 'cifar10', None, get_params=True)
params['architecture'] = 'sdn'
params['base_model'] = 'tmp_vgg16_sdn'
params['network_type'] = 'vgg16'
model = VGG_SDN(params)

# 打印模型结构与各出口累计 ops（只计卷积和全连接）
print_sdn_model_and_convfc_flops(model, input_size=params['input_size'], device=torch.device('cpu'))