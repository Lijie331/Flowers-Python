# LIFT花卉识别系统 - 部署指南

## 📁 文件说明

```
D:\1B.毕业设计\Code - 副本\LIFT-main\
├── inference.py      # 独立推理脚本（可直接测试用）
├── app.py            # Flask后端服务（生产环境使用）
└── output/model/     # 训练好的模型权重
    └── checkpoint.pth.tar
```

## 🚀 启动步骤

### 方式一：使用Flask后端服务（推荐）

#### 1. 安装依赖

```bash
# 进入LIFT-main目录
cd "D:\1B.毕业设计\Code - 副本\LIFT-main"

# 安装Flask和跨域支持
pip install flask flask-cors
```

#### 2. 启动后端服务

```bash
python app.py
```

服务启动后会在 `http://127.0.0.1:5000` 运行。

### 方式二：测试推理脚本

```bash
python inference.py
```

## 🌐 使用前端界面

#### 1. 启动Vue前端

```bash
# 进入前端目录
cd "D:\1B.毕业设计\Graduation-Project-Flowers\Graduation-Project-Flowers"

# 安装依赖（如果尚未安装）
npm install

# 启动开发服务器
npm run dev
```

#### 2. 访问识别页面

打开浏览器访问 `http://localhost:5173`，进入"识别"页面。

#### 3. 上传图片并识别

1. 点击上传区域或拖拽图片
2. 点击"开始识别"按钮
3. 等待识别结果

## 📡 API接口说明

### 1. 健康检查
```
GET http://127.0.0.1:5000/api/health
```

响应：
```json
{
  "status": "ok",
  "message": "Flower recognition service is running",
  "device": "cuda"
}
```

### 2. 图片识别
```
POST http://127.0.0.1:5000/api/classify
Content-Type: multipart/form-data
```

参数：
- `image`: 图片文件
- `top_k`: 返回前k个预测结果（可选，默认5）

响应：
```json
{
  "success": true,
  "results": [
    {"class_id": 0, "name": "pink primrose", "confidence": 95.5},
    {"class_id": 5, "name": "english marigold", "confidence": 2.3},
    ...
  ],
  "top_result": {
    "class_id": 0,
    "name": "pink primrose",
    "confidence": 95.5
  }
}
```

### 3. 获取所有类别
```
GET http://127.0.0.1:5000/api/classes
```

## ⚠️ 常见问题

### 1. 无法连接到后端服务
确保Flask服务已启动，并且运行在 `http://127.0.0.1:5000`

### 2. CUDA内存不足
如果遇到CUDA内存错误，可以将模型移到CPU运行。修改 `app.py` 中的设备选择逻辑。

### 3. CLIP模型下载失败
代码会自动下载CLIP模型。如果下载失败，请检查网络连接。

## 📊 花卉类别（102类）

| ID | 英文名称 | ID | 英文名称 |
|----|---------|----|---------|
| 0 | pink primrose | 1 | hard-leaved pocket orchid |
| 2 | canterbury bells | 3 | sweet pea |
| 4 | english marigold | 5 | tiger lily |
| 6 | moon orchid | 7 | bird of paradise |
| 8 | monkshood | 9 | globe thistle |
| ... | ... | ... | ... |

(完整列表见 app.py 中的 CLASSNAMES 变量)

## 🔧 配置修改

### 修改API地址
如果后端服务不在本机运行，修改前端文件中的 `API_BASE_URL`：
```javascript
const API_BASE_URL = 'http://你的服务器IP:5000'
```

### 修改模型路径
在 `app.py` 中修改：
```python
CHECKPOINT_PATH = r'D:\你的路径\checkpoint.pth.tar'
```
