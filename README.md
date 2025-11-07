比较好的pcgrad cw 攻击
python attack.py --entropy_thresholds='0.20,0.20,0.30,0.3,0.35,0.35' --cw --pcgrad='one-sided'
--pgd_update_mode='l2_dir' --skip_contains=["cnn","cifar10_","imagenet", "training"]

针对cifar10这些不好攻击的模型，只能调大迭代次数，提高权重
python3 attack.py --entropy_thresholds="0.7,0.7,0.8,0.8,0.85,0.85" --lambda_exits="1,1,1,1,1,1,1" --pgd_steps=100 --lambda_earlyexits=100 --lambda_ce=1 --skip_contains=["cnn","cifar100","imagenet", "training"]

## Per-exit conv+fc FLOPs (ops) 统计

按用户指定公式统计每个出口从输入到该出口累计的运算量（只计 Conv2d 和 Linear）：
- Conv2d: H_out × W_out × K × K × C_in × C_out
- Linear: in_features × out_features（按 batch 线性累加，默认 batch=1）

用法示例（CPU 上运行一次前向统计，不需要训练权重）：

```python
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
clear

# 打印模型结构与各出口累计 ops（只计卷积和全连接）
print_sdn_model_and_convfc_flops(model, input_size=params['input_size'], device=torch.device('cpu'))
```

如果只需要拿到各出口累计 ops 的字典，可直接调用：

```python
from profiler import profile_sdn_convfc
exit_ops = profile_sdn_convfc(model, input_size=params['input_size'], device=torch.device('cpu'))
# exit_ops: {exit_id(0-based): ops(int)}
```

## 以“块”为单位打印 SDN 结构体（更直观）

新增块状结构打印，按 Init / Blocks / End 三个部分展示：

```python
from profiler import print_sdn_block_structure

print_sdn_block_structure(model)
```

示例输出（节选）：

```
+--------------------+
| Init |
+--------------------+
	- Conv2d(3->16, k=3x3, s=1, p=1)
	- BN(16)
	- ReLU

Blocks
======
+-----------------------------------+
| Block 0: BasicBlockWOutput |
+-----------------------------------+
	[sub-0] Sequential: Conv-BN-ReLU-Conv-BN
	[sub-1] Sequential([])  # shortcut
	[sub-2] ReLU
	Exit 0: InternalClassifier -> Linear(...->num_classes)

+------------------+
| End |
+------------------+
	- AvgPool2d(k=8, s=8)
	- Flatten
	- Linear(64->num_classes)
```

	## 直接导出模型结构图（SVG）

	新增纯 Python 的 SVG 导出工具，生成更直观的结构图文件：

	- 脚本：`printmodelsvg.py`
	- 输出：`outputs/diagrams/model_structure.svg`

	用法：

	```bash
	# 使用与你当前环境一致的 Python 解释器（之前能运行 printmodelinfo.py 的那个）
	python printmodelsvg.py
	```

	如果你在某个 shell 里没有装 PyTorch，可使用另外一个已装的环境运行；或安装 CPU 版本：

	```bash
	# 可选：安装 CPU 版 torch（若当前环境缺少）
	python3 -m pip install --upgrade pip
	python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

	# 然后运行
	python3 printmodelsvg.py
	```

	生成的 SVG 可以直接在浏览器或 VS Code 里打开预览，也便于嵌入文档/汇报。