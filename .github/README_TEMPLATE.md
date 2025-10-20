# 混合检索评估工具 - GitHub仓库配置

## 仓库设置步骤

### 1. 创建新仓库

1. 登录GitHub
2. 点击右上角的 "+" → "New repository"
3. 填写仓库信息：
   - **Repository name**: `hybrid-retrieval-evaluation`
   - **Description**: `Hybrid retrieval evaluation metrics extension for RAGAS framework`
   - **Public**: 选择公开仓库（推荐）
   - **Initialize**: 不要勾选任何初始化选项

### 2. 推送代码

在本地项目根目录执行：

```bash
# 初始化git仓库（如果尚未初始化）
git init

# 添加远程仓库
git remote add origin https://github.com/your-username/hybrid-retrieval-evaluation.git

# 添加所有文件
git add .

# 提交初始版本
git commit -m "Initial commit: Hybrid retrieval evaluation tool with information theory metrics"

# 推送到main分支
git branch -M main
git push -u origin main
```

### 3. 配置仓库设置

#### 基本设置
1. 进入仓库的 "Settings" → "General"
2. 设置默认分支为 `main`
3. 启用 "Automatically delete head branches"
4. 禁用 "Allow merge commits"（推荐只使用Squash merge）

#### 分支保护规则
1. 进入 "Settings" → "Branches"
2. 点击 "Add rule"
3. 配置规则：
   - **Branch name pattern**: `main`
   - **Protect matching branches**: 勾选
   - **Require a pull request before merging**: 勾选
   - **Require status checks to pass before merging**: 勾选
   - **Require branches to be up to date before merging**: 勾选
   - **Status checks**: 添加 `test` 和 `lint`

### 4. 配置Secrets

1. 进入 "Settings" → "Secrets and variables" → "Actions"
2. 添加以下secrets：
   - `PYPI_API_TOKEN`: PyPI发布令牌（如果需要发布到PyPI）
   - `CODECOV_TOKEN`: Codecov令牌（可选，用于覆盖率报告）

### 5. 启用功能

#### Issues
- 启用Issues功能
- 配置Issue模板（已包含在 `.github/ISSUE_TEMPLATE/`）

#### Discussions（可选）
- 启用Discussions功能，用于社区讨论

#### Wiki（可选）
- 启用Wiki功能，用于文档维护

### 6. 配置Topics

在仓库主页添加相关topics：
- `hybrid-retrieval`
- `ragas`
- `information-theory`
- `evaluation-metrics`
- `dense-retrieval`
- `sparse-retrieval`
- `python`

### 7. 创建初始Release

1. 点击 "Create a new release"
2. 选择标签：v1.0.0
3. 标题：v1.0.0 - Initial Release
4. 描述：
   ```
   混合检索评估工具初始版本发布！

   ## 功能特性
   - 基于信息论的混合检索评估指标
   - 自适应权重优化
   - 三重置信度验证机制
   - RAGAS框架扩展
   - 完整的测试覆盖（61个测试用例）

   ## 安装
   ```bash
   pip install hybrid-retrieval-evaluation
   ```

   ## 快速开始
   查看 [快速开始指南](https://github.com/your-username/hybrid-retrieval-evaluation/blob/main/QUICK_START.md)
   ```

## 社区建设

### 贡献指南
- 参考 [CONTRIBUTING.md](../CONTRIBUTING.md)
- 遵循代码规范
- 提交前运行测试

### 行为准则
- 参考 [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)
- 保持友好和专业的交流

### 联系方式
- Issue: 使用GitHub Issues
- 邮件：1312750677@qq.com
- 学术讨论：欢迎提交PR和Issue

## 推广建议

1. **学术社区**
   - 在相关论文中引用此工具
   - 在学术会议上介绍
   - 提交到相关论文代码平台

2. **技术社区**
   - 在知乎、CSDN等平台分享
   - 参与相关技术讨论
   - 撰写技术博客

3. **社交媒体**
   - Twitter/X: @your_handle
   - LinkedIn: 分享项目进展
   - 微信公众号：技术文章分享

## 持续维护

### 定期任务
- [ ] 回复Issues和PRs
- [ ] 更新依赖包
- [ ] 发布新版本
- [ ] 更新文档

### 版本规划
- v1.1.0: 添加更多评估指标
- v1.2.0: 优化性能和稳定性
- v2.0.0: 支持更多检索框架

### 监控指标
- Star数量增长
- Fork数量
- Issue响应时间
- 下载量统计

## 许可证

MIT License - 详见 [LICENSE](../LICENSE) 文件