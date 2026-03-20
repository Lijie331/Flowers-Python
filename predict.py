"""
LIFT模型预测脚本
用于对单张图像进行花卉识别
"""

import os
import sys

# 将LIFT-main目录添加到Python路径
LIFT_DIR = r"D:\1B.毕业设计\Code - 副本\LIFT-main"
if LIFT_DIR not in sys.path:
    sys.path.insert(0, LIFT_DIR)

# 设置环境变量
os.environ['CLIP_CACHE_DIR'] = r'D:\1B.毕业设计\CLIP_cache'

import torch
from torchvision import transforms
from PIL import Image
# 直接从clip子模块导入，避免与其他clip包冲突
from clip.clip import _MODELS, _download, build_model


def load_clip_model(backbone_name="RN50"):
    """加载CLIP模型"""
    print(f"加载CLIP {backbone_name}...")
    backbone_name = backbone_name.lstrip("CLIP-")
    url = _MODELS[backbone_name]
    model_path = _download(url)
    
    # 加载模型 - 使用trainer.py中的方法
    try:
        loaded = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = loaded.state_dict()
        model = build_model(state_dict)
    except Exception as e:
        print(f"JIT load failed: {e}")
        loaded = torch.load(model_path, map_location="cpu")
        if hasattr(loaded, 'state_dict'):
            state_dict = loaded.state_dict()
        elif isinstance(loaded, dict):
            state_dict = loaded
        else:
            state_dict = {k: v.cpu() for k, v in loaded.named_parameters()}
        model = build_model(state_dict)
    
    return model


def load_trained_model(model, checkpoint_path):
    """加载训练好的权重"""
    print(f"加载训练权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 尝试加载state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 加载权重
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Warning: {e}")
        # 尝试部分加载
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    return model


def preprocess_image(image_path, resolution=224):
    """预处理图像"""
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    
    transform = transforms.Compose([
        transforms.Resize(resolution * 8 // 7),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def predict(model, image_tensor, class_names, top_k=5):
    """进行预测"""
    with torch.no_grad():
        # 获取图像特征
        if hasattr(model, 'encode_image'):
            image_features = model.encode_image(image_tensor)
        else:
            # 对于ResNet类型的模型
            image_features = model(image_tensor)
        
        # 如果有classifier，使用它
        if hasattr(model, 'classifier') and model.classifier is not None:
            logits = model.classifier(image_features)
        else:
            # 否则使用线性层
            logits = image_features @ model.visual.output_proj.T if hasattr(model.visual, 'output_proj') else image_features
        
        probs = logits.softmax(dim=-1)
        top_probs, top_indices = probs[0].topk(top_k)
    
    results = []
    for i in range(top_k):
        class_id = top_indices[i].item()
        # 限制类别ID在有效范围内
        class_id = min(class_id, len(class_names) - 1)
        results.append({
            'class_id': int(class_id),
            'class_name': class_names[class_id],
            'confidence': top_probs[i].item()
        })
    
    return results


def main():
    # 配置 - 修改这里来指定您的图片路径
    checkpoint_path = r"D:\1B.毕业设计\Code - 副本\LIFT-main\output\oxford_flowers_clip_rn50\checkpoint.pth.tar"
    image_path = r"D:\1B.毕业设计\数据集\jpg\image_00001.jpg"  # 测试图像
    
    # 102个花卉类别名称
    class_names = [
        "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
        "english marigold", "tiger lily", "moon orchid", "bird of paradise",
        "monkshood", "globe thistle", "snapdragon", "coltsfoot",
        "king protea", "spear thistle", "yellow iris", "globe flower",
        "purple coneflower", "peruvian lily", "balloon flower", "giant white arum lily",
        "fire lily", "pincushion mushroom", "fritillary", "red ginger",
        "sunflower", "lily", "calendula", "marsh orchid",
        "artichoke", "hibiscus", "lotus lotus", "foxtail lily",
        "clematis", "hibiscus", "larkspur", "carnation",
        "garden phlox", "love-in-the-mist", "cosmos", "alpine sea holly",
        "ruby-lipped cattleya", "cape flower", "siam tulip", "lenten rose",
        "barbeton daisy", "daffodil", "magnolia", "cyclamen",
        "watercress", "monkshood", "arts shawl", "kingfisher",
        "corn poppy", "prince of wales feathers", "gypsophila", "ardtemis",
        "busy lizzie", "bromelia", "magnolia", "mexican petunia",
        "bougainvillea", "camellia", "mallow", "mexican hat",
        "geranium", "pentas", "bee balm", "balloon flower",
        "oxeye daisy", "black-eyed susan", "cobaea", "blanket flower",
        "trumpet creeper", "blackberry lily", "common tulip", "wild rose",
        "thorn apple", "morning glory", "passion flower", "lotus",
        "toad lily", "anemone", "frangipani", "plumeria",
        "hippeastrum", "blue poppy", "celandine", "tree poppy",
        "azalea", "flowering cherry", "indian strawberry", "frangipani",
        "magenta spider lily", "gaillardia", "yarrow", "colchicum",
        "mexican sunflower", "oxeye daisy", "gardenias", "marigold",
        "petunia", "california poppy", "canna lily", "osteospermum",
        "california poppy", "snapdragon", "camellia", "impatiens",
        "begonia", "lantana", "verbena", "wedelia",
        "yellow sunflower", "hirsute viola", "canna", "petunia"
    ]
    
    # 加载模型
    model = load_clip_model("RN50")
    model = load_trained_model(model, checkpoint_path)
    model.eval()
    model.float()  # 转换为float32
    
    # 预处理图像
    print(f"\n处理图像: {image_path}")
    image_tensor = preprocess_image(image_path)
    
    # 预测
    print("识别中...")
    results = predict(model, image_tensor, class_names, top_k=5)
    
    # 输出结果
    print("\n" + "="*50)
    print("识别结果:")
    print("="*50)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['class_name']}")
        print(f"   置信度: {result['confidence']*100:.2f}%")
    print("="*50)


if __name__ == "__main__":
    main()
