"""
Flask后端服务 - 花卉识别API
启动命令: python app.py
"""

import os
import sys
import io
import base64
import json

# 解决OpenMP运行时冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置CLIP缓存目录
os.environ['CLIP_CACHE_DIR'] = r'D:\1B.毕业设计\CLIP_cache'

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
from torchvision import transforms

from clip import clip

# 添加项目根目录到路径
sys.path.insert(0, r'D:\1B.毕业设计\Code - 副本\LIFT-main')

app = Flask(__name__)
CORS(app)


# ============== 花卉信息配置 ==============
# 花卉信息JSON文件路径
FLOWER_CLASSES_FILE = r'D:\1B.毕业设计\数据集\flower_classes.json'

# 花卉类别名称列表和对照表（从JSON文件加载）
CLASSNAMES = []  # 英文名称列表
CLASSNAMES_CN = {}  # 英文->中文对照表
CLASS_INFO = {}  # 完整花卉信息字典 {id: {name_en, name_cn}}

def load_flower_classes():
    """从JSON文件加载花卉类别信息"""
    global CLASSNAMES, CLASSNAMES_CN, CLASS_INFO
    
    try:
        with open(FLOWER_CLASSES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 按ID顺序加载
        sorted_classes = sorted(data['classes'], key=lambda x: x['id'])
        
        CLASSNAMES = [item['name_en'] for item in sorted_classes]
        CLASSNAMES_CN = {item['name_en']: item['name_cn'] for item in sorted_classes}
        CLASS_INFO = {item['id']: {'name_en': item['name_en'], 'name_cn': item['name_cn']} for item in sorted_classes}
        
        print(f"[INFO] 成功加载 {len(CLASSNAMES)} 个花卉类别")
        return True
    except Exception as e:
        print(f"[ERROR] 加载花卉类别失败: {e}")
        return False

# 启动时加载花卉信息
load_flower_classes()
# ==========================================


# 路径配置
CHECKPOINT_PATH = r'D:\1B.毕业设计\Code - 副本\LIFT-main\output\oxford_flowers_clip_rn50_num_epochs_50\checkpoint.pth.tar'
RESOLUTION = 224
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]

# 全局模型
classifier = None
device = None


def get_config_dict():
    """返回配置字典"""
    return {
        'backbone': "CLIP-RN50",
        'resolution': 224,
        'prompt': "default",
        'scale': 1.0,
        'learnable_scale': False,
        'bias': "none",
        'init_style': "uniform",
        'full_tuning': False,
        'classifier': "CosineClassifier",
        'bias_tuning': False,
        'bn_tuning': False,
        'ln_tuning': False,
        'vpt_shallow': False,
        'vpt_deep': False,
        'adapter': False,
        'adaptformer': False,
        'lora': False,
        'lora_mlp': False,
        'ssf_attn': False,
        'ssf_mlp': False,
        'ssf_ln': False,
        'mask': False,
        'partial': None,
        'vpt_len': None,
        'adapter_dim': 64,
        'mask_ratio': 0.0,
        'mask_seed': 42,
    }


class ModelConfig:
    """配置类 - 同时支持属性访问和字典解包"""
    def __init__(self):
        self.backbone = "CLIP-RN50"
        self.resolution = 224
        self.prompt = "default"
        self.scale = 1.0
        self.learnable_scale = False
        self.bias = "none"
        self.init_style = "uniform"
        self.full_tuning = False
        self.classifier = "CosineClassifier"
        self.bias_tuning = False
        self.bn_tuning = False
        self.ln_tuning = False
        self.vpt_shallow = False
        self.vpt_deep = False
        self.adapter = False
        self.adaptformer = False
        self.lora = False
        self.lora_mlp = False
        self.ssf_attn = False
        self.ssf_mlp = False
        self.ssf_ln = False
        self.mask = False
        self.partial = None
        self.vpt_len = None
        self.adapter_dim = 64
        self.mask_ratio = 0.0
        self.mask_seed = 42
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def keys(self):
        return [k for k in dir(self) if not k.startswith('_') and not callable(getattr(self, k))]


def load_clip_to_cpu(backbone_name):
    """加载CLIP模型到CPU - 使用备用方式避免路径问题"""
    backbone_name = backbone_name.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    
    # 直接指定模型路径，避免_download的路径问题
    clip_cache_dir = r'D:\1B.毕业设计\CLIP_cache'
    model_path = os.path.join(clip_cache_dir, f"{backbone_name.replace('/', '-')}.pt")
    
    # 清理路径中的换行符
    model_path = model_path.replace('\n', '').replace('\r', '')
    
    print(f"Loading CLIP from: {model_path}")
    
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = model.state_dict()
        model = clip.build_model(state_dict)
    except Exception as e:
        print(f"JIT load failed: {e}")
        try:
            loaded = torch.load(model_path, map_location="cpu", weights_only=False)
            if hasattr(loaded, 'state_dict'):
                state_dict = loaded.state_dict()
            elif isinstance(loaded, dict):
                state_dict = loaded
            else:
                state_dict = {k: v.cpu() for k, v in loaded.named_parameters()}
            model = clip.build_model(state_dict)
        except Exception as e2:
            print(f"Fallback load also failed: {e2}")
            # 最后尝试用clip._download
            downloaded_path = clip._download(url).replace('\n', '').replace('\r', '')
            print(f"Trying downloaded path: {downloaded_path}")
            model = torch.jit.load(downloaded_path, map_location="cpu").eval()
            state_dict = model.state_dict()
            model = clip.build_model(state_dict)
    
    model.float()
    return model


def load_model():
    """加载模型"""
    global classifier, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 使用硬编码配置
    config = ModelConfig()
    print(f"Using backbone: {config.backbone}")
    
    # 加载CLIP模型
    print(f"Loading CLIP model: {config.backbone}")
    clip_model = load_clip_to_cpu(config.backbone)
    
    # 导入模型构建函数
    from models import PeftModelFromCLIP
    
    # 构建模型
    print("Building LIFT model...")
    classifier = PeftModelFromCLIP(config, clip_model, len(CLASSNAMES))
    classifier.to(device)
    classifier.eval()
    
    # 加载checkpoint
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    
    tuner_dict = checkpoint["tuner"]
    head_dict = checkpoint["head"]
    
    # 移除 'module.' 前缀
    new_tuner_dict = {k[7:] if k.startswith("module.") else k: v for k, v in tuner_dict.items()}
    new_head_dict = {k[7:] if k.startswith("module.") else k: v for k, v in head_dict.items()}
    
    classifier.tuner.load_state_dict(new_tuner_dict, strict=False)
    classifier.head.load_state_dict(new_head_dict, strict=False)
    
    print("Model loaded successfully!")
    return classifier


def get_transform():
    """获取图像预处理变换"""
    return transforms.Compose([
        transforms.Resize(RESOLUTION * 8 // 7),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'message': 'Flower recognition service is running',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })


@app.route('/api/classify', methods=['POST'])
def classify_flower():
    """花卉识别接口"""
    global classifier
    
    if classifier is None:
        try:
            load_model()
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Failed to load model: {str(e)}'
            }), 500
    
    classifier.eval()
    transform = get_transform()
    
    try:
        # 处理JSON请求（base64图片）
        if request.is_json:
            data = request.get_json()
            top_k = data.get('top_k', 5)
            image_data = data.get('image', '')
            
            if not image_data:
                return jsonify({
                    'success': False,
                    'error': 'Missing image data'
                }), 400
            
            # 解码base64图片
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 处理form-data请求（文件上传）
        elif 'image' in request.files:
            file = request.files['image']
            top_k = int(request.form.get('top_k', 5))
            image = Image.open(file).convert('RGB')
        
        else:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # 预处理图像
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            output = classifier(img_tensor)
            probs = torch.softmax(output, dim=1)
            
            # 获取top_k预测
            top_probs, top_indices = torch.topk(probs, min(top_k, len(CLASSNAMES)), dim=1)
        
        # 构建结果
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_id = int(idx.item())
            english_name = CLASSNAMES[class_id]
            chinese_name = CLASSNAMES_CN.get(english_name, english_name)
            results.append({
                'class_id': class_id,
                'name': english_name,
                'name_cn': chinese_name,
                'display_name': f"{chinese_name} ({english_name})",
                'confidence': round(prob.item() * 100, 2)
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'top_result': results[0] if results else None
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """获取所有花卉类别"""
    return jsonify({
        'success': True,
        'classes': [{'id': i, 'name_en': info['name_en'], 'name_cn': info['name_cn']} 
                    for i, info in enumerate(CLASSNAMES)],
        'total': len(CLASSNAMES)
    })


@app.route('/api/flower-info/<int:class_id>', methods=['GET'])
def get_flower_info(class_id):
    """获取指定花卉的详细信息"""
    if class_id in CLASS_INFO:
        return jsonify({
            'success': True,
            'data': {
                'id': class_id,
                'name_en': CLASS_INFO[class_id]['name_en'],
                'name_cn': CLASS_INFO[class_id]['name_cn']
            }
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Flower class not found'
        }), 404


if __name__ == '__main__':
    print("=" * 50)
    print("Flower Recognition API Server")
    print("=" * 50)
    print("Endpoints:")
    print("  GET  /api/health         - Health check")
    print("  POST /api/classify       - Classify flower image")
    print("  GET  /api/classes        - List all flower classes")
    print("  GET  /api/flower-info/<id> - Get flower details")
    print("")
    print(f"Flower classes loaded from: {FLOWER_CLASSES_FILE}")
    print("")
    print("Starting server on http://127.0.0.1:5000")
    print("=" * 50)
    
    # 预加载模型
    load_model()
    
    # 启动服务
    app.run(host='127.0.0.1', port=5000, debug=False)
