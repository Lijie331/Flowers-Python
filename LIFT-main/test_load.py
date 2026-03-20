import os, sys, json, torch

# 设置路径
clip_cache = r'D:\1B.毕业设计\CLIP_cache'
os.environ['CLIP_CACHE_DIR'] = clip_cache
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 加载类别
sys.path.insert(0, r'D:\1B.毕业设计\Code - 副本\LIFT-main')
from datasets.oxford_flowers import Oxford_Flowers
CLASSNAMES = Oxford_Flowers.classnames
print('类别数:', len(CLASSNAMES))

# 加载模型
from models import PeftModelFromCLIP
from clip import clip

class Config:
    backbone = 'CLIP-RN50'
    resolution = 224
    classifier = 'CosineClassifier'
    scale = 1.0
    bias = 'none'
    init_style = 'uniform'
    full_tuning = False
    bias_tuning = False
    bn_tuning = False
    ln_tuning = False
    vpt_shallow = False
    vpt_deep = False
    adapter = False
    adaptformer = False
    lora = False
    lora_mlp = False
    ssf_attn = False
    ssf_mlp = False
    ssf_ln = False
    mask = False
    partial = None
    vpt_len = None
    adapter_dim = 64
    mask_ratio = 0.0
    mask_seed = 42

config = Config()

# 加载CLIP - 使用jit.load
print('Loading CLIP...')
clip_model_file = os.path.join(clip_cache, 'RN50.pt')
print(f'Model file: {clip_model_file}')
print(f'File exists: {os.path.exists(clip_model_file)}')

# 使用jit.load加载
loaded = torch.jit.load(clip_model_file, map_location='cpu')
state_dict = loaded.state_dict()
clip_model = clip.build_model(state_dict)
clip_model.float()
print('CLIP模型加载成功')

# 构建LIFT模型
print('构建LIFT模型...')
classifier = PeftModelFromCLIP(config, clip_model, len(CLASSNAMES))
print('head weight shape:', classifier.head.weight.shape)

# 加载checkpoint
CHECKPOINT_PATH = r'D:\1B.毕业设计\Code - 副本\LIFT-main\output\oxford_flowers_clip_rn50_num_epochs_50\checkpoint.pth.tar'
print('Loading checkpoint...')
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
head_dict = checkpoint['head']
new_head_dict = {k[7:] if k.startswith('module.') else k: v for k, v in head_dict.items()}

# 加载head权重
classifier.head.load_state_dict(new_head_dict, strict=False)
print('模型加载成功！')
print('head weight shape after loading:', classifier.head.weight.shape)
print()
print('测试完成! 现在可以用 app.py 启动服务了')
