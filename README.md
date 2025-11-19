比较好的pcgrad cw 攻击
python attack.py --entropy_thresholds='0.20,0.20,0.30,0.3,0.35,0.35' --cw --pcgrad='one-sided'
--pgd_update_mode='l2_dir' --skip_contains=["cnn","cifar10_","imagenet", "training"]

针对cifar10这些不好攻击的模型，只能调大迭代次数，提高权重
python3 attack.py --entropy_thresholds="0.7,0.7,0.8,0.8,0.85,0.85" --lambda_exits="1,1,1,1,1,1,1" --pgd_steps=100 --lambda_earlyexits=100 --lambda_ce=1 --skip_contains=["cnn","cifar100","imagenet", "training"]
