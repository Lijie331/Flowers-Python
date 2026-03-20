"""
LIFT模型推理脚本 - 用于花卉识别
使用训练好的checkpoint进行图片分类
"""

import os
import sys
import io
import json
import base64

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# 设置CLIP缓存目录
os.environ['CLIP_CACHE_DIR'] = r'D:\1B.毕业设计\CLIP_cache'

from clip import clip
from timm.models.vision_transformer import vit_base_patch16_224

# 添加项目根目录到路径
sys.path.insert(0, r'D:\1B.毕业设计\Code - 副本\LIFT-main')
from models import PeftModelFromCLIP, Peft_ViT, ViT_Tuner
from models.classifiers import CosineClassifier
from datasets.oxford_flowers import Oxford_Flowers


def load_clip_to_cpu(backbone_name):
    """加载CLIP模型到CPU"""
    backbone_name = backbone_name.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = model.state_dict()
        model = clip.build_model(state_dict)
    except Exception as e:
        print(f"JIT load failed: {e}")
        loaded = torch.load(model_path, map_location="cpu")
        if hasattr(loaded, 'state_dict'):
            state_dict = loaded.state_dict()
        elif isinstance(loaded, dict):
            state_dict = loaded
        else:
            state_dict = {k: v.cpu() for k, v in loaded.named_parameters()}
        model = clip.build_model(state_dict)
    
    model.float()
    return model


class FlowerClassifier:
    """花卉分类器"""
    
    def __init__(self, checkpoint_path, backbone_name="CLIP-ViT-B/16", resolution=224):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resolution = resolution
        
        # 花卉类别名称 (102类)
        self.classnames = [
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
        
        print(f"Loading CLIP model: {backbone_name}")
        clip_model = load_clip_to_cpu(backbone_name)
        
        print("Building LIFT model...")
        self.model = PeftModelFromCLIP(
            self._create_config(backbone_name),
            clip_model,
            len(self.classnames)
        )
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        self.load_checkpoint(checkpoint_path)
        
        # 图像预处理
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = transforms.Compose([
            transforms.Resize(self.resolution * 8 // 7),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        
        print("Model loaded successfully!")
    
    def _create_config(self, backbone_name):
        """创建模型配置"""
        class Config:
            backbone = backbone_name
            prompt = "default"
            resolution = self.resolution
            scale = 1.0
            learnable_scale = False
            bias = "none"
            init_style = "uniform"
        return Config()
    
    def load_checkpoint(self, checkpoint_path):
        """加载训练好的权重"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        tuner_dict = checkpoint["tuner"]
        head_dict = checkpoint["head"]
        
        # 移除 'module.' 前缀（如果有多GPU训练）
        new_tuner_dict = {}
        for k, v in tuner_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_tuner_dict[k] = v
        tuner_dict = new_tuner_dict
        
        new_head_dict = {}
        for k, v in head_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_head_dict[k] = v
        head_dict = new_head_dict
        
        self.model.tuner.load_state_dict(tuner_dict, strict=False)
        self.model.head.load_state_dict(head_dict, strict=False)
        
        print("Checkpoint loaded!")
    
    def preprocess_image(self, image):
        """预处理图像"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        return self.transform(image)
    
    @torch.no_grad()
    def predict(self, image, top_k=5):
        """
        预测图像类别
        
        Args:
            image: PIL.Image, 文件路径 或 bytes
            top_k: 返回前k个预测结果
        
        Returns:
            List of dicts: [{'class_id': 0, 'name': 'rose', 'confidence': 0.95}, ...]
        """
        img_tensor = self.preprocess_image(image)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        output = self.model(img_tensor)
        probs = torch.softmax(output, dim=1)
        
        # 获取top_k预测
        top_probs, top_indices = torch.topk(probs, min(top_k, len(self.classnames)), dim=1)
        
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            results.append({
                'class_id': idx.item(),
                'name': self.classnames[idx.item()],
                'confidence': round(prob.item() *100, 2)  # 转为百分比
            })
        
        return results
    
    def predict_base64(self, base64_image, top_k=5):
        """从base64编码的图片进行预测"""
        # 解码base64
        image_data = base64.b64decode(base64_image)
        return self.predict(image_data, top_k)


# 全局分类器实例
_classifier = None


def get_classifier():
    """获取或创建分类器实例"""
    global _classifier
    if _classifier is None:
        checkpoint_path = r'D:\1B.毕业设计\Code - 副本\LIFT-main\output\model\checkpoint.pth.tar'
        _classifier = FlowerClassifier(checkpoint_path)
    return _classifier


if __name__ == "__main__":
    # 测试代码
    classifier = get_classifier()
    
    # 测试预测
    test_image_path = r"D:\1B.毕业设计\数据集\jpg\image_00001.jpg"
    if os.path.exists(test_image_path):
        results = classifier.predict(test_image_path)
        print("\n预测结果:")
        for r in results:
            print(f"  {r['class_id']}: {r['name']} - {r['confidence']}%")
