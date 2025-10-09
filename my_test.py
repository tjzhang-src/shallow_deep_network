import aux_funcs as af
import network_architectures as arcs

models_path = 'networks/<你的seed>'
model_name = '<你的已训练模型名>'   # 例如：ResNet56_cifar10_sdn_training 或 VGG16_cifar10

device = af.get_pytorch_device()
model, params = arcs.load_model(models_path, model_name, -1)  # -1 加载最新
dataset = af.get_dataset(params['task'])
model.to(device)

metrics = model.test_func(model, dataset, device=device)  # 调用 mf.sdn_test 或 mf.cnn_test
print(metrics)  # 一般包含 test_top1_acc / test_top5_acc 等