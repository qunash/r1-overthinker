# DeepSeek R1 Overthinker

一个基于 DeepSeek R1 模型的增强思考应用。通过检测和替换模型的思考过程，强制模型进行更深入的推理。更方便的自定义下载模型和路径，更方便修改/自定义其中组件（可实现RAG等文件上传）

An enhanced thinking application based on the DeepSeek R1 model. It forces deeper reasoning by detecting and replacing the model's thinking process. Easier to customize model downloads and paths, and more convenient to modify/customize components (supporting features like RAG file upload).

## 项目结构 | Project Structure

```
src/
├── app.py              # 主应用程序入口 | Main application entry
├── utils.py            # 工具模块 | Utility module
└── load_model.py       # 模型下载，snapshot_download是目前hugging face下载最方便的方式 | Model download, snapshot_download is currently the fastest way to download from Hugging Face
```

## 功能特点 | Features

1. **简化的项目结构 | Simplified Structure**
   - 核心功能集成在app.py中 | Core functionality integrated in app.py
   - 独立的模型下载工具 | Standalone model download utility
   - 易于扩展和自定义 | Easy to extend and customize

2. **增强的思考过程 | Enhanced Thinking Process**
   - 检测并扩展模型思考 | Detect and extend model thinking
   - 可自定义思考提示词 | Customizable thinking prompts
   - 灵活的参数控制 | Flexible parameter control

## 参数说明 | Parameters

- **Minimum Thinking Tokens**: 最小思考token数 | Minimum tokens for thinking process
- **Maximum Output Tokens**: 最大输出token数 | Maximum tokens for output
- **Maximum Reasoning Extensions**: 最大推理扩展次数 | Maximum number of reasoning extensions
- **Temperature**: 生成温度，官方建议为0.6 | Generation temperature, officially recommended as 0.6
- **Top-p**: 采样阈值 | Sampling threshold
- **Repetition Penalty**: 重复惩罚系数 | Repetition penalty coefficient

## 使用方法 | Usage

1. **安装依赖 | Install Dependencies**
   ```bash
   pip install gradio torch transformers vllm huggingface_hub
   ```

2. **运行应用 | Run Application**
   ```bash
   python app.py
   ```

## 注意事项 | Notes

1. **模型说明 | Model Information**:
   - 默认使用 unsloth/DeepSeek-R1-Distill-Qwen-32B 模型 | Using unsloth/DeepSeek-R1-Distill-Qwen-32B model by default
   - 不支持GGUF格式模型 | GGUF format models are not supported
   - 首次运行会自动下载模型 | Model will be downloaded automatically on first run

2. **GPU要求 | GPU Requirements**:
   - 目前仅支持单GPU | Currently only supports single GPU
   - 默认使用GPU 0 | Uses GPU 0 by default
   - 可在utils.py中修改GPU设置 | GPU settings can be modified in utils.py

3. **GPU内存使用参考 | GPU Memory Usage Reference**:
   - 3,000 tokens → ~8 GB VRAM
   - 22,000 tokens → ~12 GB VRAM
   - 41,000 tokens → ~16 GB VRAM
   - 78,000 tokens → ~24 GB VRAM
   - 154,000 tokens → ~40 GB VRAM

4. **内存优化建议 | Memory Optimization Tips**:
   - 降低上下文长度 | Reduce context length
   - 减少最大输出token数 | Reduce maximum output tokens
   - 使用更小的模型 | Use smaller models

## 自定义与扩展 | Customization & Extension

- 支持自定义模型下载路径 | Custom model download paths
- 可集成RAG等功能 | Integrable with RAG functionality
- 可自定义UI组件 | Customizable UI components
- 灵活的参数配置 | Flexible parameter configuration
