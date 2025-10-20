# 安装指南

## 系统要求

- Python 3.8 或更高版本
- pip 包管理器

## 快速安装

### 从源码安装（推荐）

```bash
# 克隆仓库
git clone https://github.com/your-username/hybrid-retrieval-evaluation.git
cd hybrid-retrieval-evaluation

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .
```

### 开发环境安装

如果您想参与开发，安装额外的开发依赖：

```bash
pip install -e ".[dev]"
```

## 验证安装

安装完成后，运行测试确保一切正常：

```bash
python -m pytest tests/ -v
```

## 基本使用

安装完成后，您可以：

1. 运行基础示例：
```bash
python examples/basic_usage.py
```

2. 运行简单评估示例：
```bash
python examples/simple_evaluation.py
```

## 环境配置（可选）

如果需要使用外部API（如OpenAI），请配置环境变量：

```bash
# 复制环境变量模板
cp experiment_code/.env.example experiment_code/.env

# 编辑 .env 文件，填入您的API密钥
```

## 故障排除

### 常见问题

1. **导入错误**：确保使用虚拟环境并正确安装了依赖
2. **测试失败**：检查Python版本是否符合要求（≥3.8）
3. **权限问题**：确保有写入日志文件的权限

### 获取帮助

如有问题，请：
1. 查看项目文档
2. 在GitHub Issues中搜索类似问题
3. 创建新的Issue描述您的问题

## 卸载

```bash
pip uninstall hybrid-retrieval-evaluation
```