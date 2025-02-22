# DeepSeek R1 Overthinker

一个基于 DeepSeek R1 模型的增强思考应用。通过检测和替换模型的思考过程，强制模型进行更深入的推理。更方便的自定义下载模型和路径，考虑到unsloth目前还不支持multiple GPU，因此将unsloth置换为了vllm，更方便多卡部署和管理

An enhanced thinking application based on the DeepSeek R1 model. It forces deeper reasoning by detecting and replacing the model's thinking process.

## 项目结构 | Project Structure

```
src/
├── app.py              # 主应用程序入口 | Main application entry
├── utils.py            # 工具模块 | Utility module
└── load_model.py       # 模型下载工具 | Model download utility
```

## 快速开始 | Quick Start

1. **安装依赖 | Install Dependencies**
   ```bash
   pip install gradio torch transformers vllm huggingface_hub
   ```

2. **运行应用 | Run Application**
   ```bash
   # 基本用法 | Basic usage
   python app.py

   # 使用本地模型 | Use local model
   python app.py --model_path /path/to/model

   # 指定GPU | Specify GPU
   python app.py --gpu_device "0,1"
   ```

## 功能特点 | Features

1. **增强的思考过程 | Enhanced Thinking Process**
   - 检测并扩展模型思考 | Detect and extend model thinking
   - 可自定义思考提示词 | Customizable thinking prompts
   - 灵活的参数控制 | Flexible parameter control

2. **可配置参数 | Configurable Parameters**
   - Minimum Thinking Tokens: 最小思考token数
   - Maximum Output Tokens: 最大输出token数
   - Temperature: 生成温度（推荐0.6）
   - Top-p & Repetition Penalty: 采样控制

## GPU配置说明 | GPU Configuration

1. **单/多GPU使用 | Single/Multi-GPU Usage**
   ```bash
   # 单GPU | Single GPU
   python app.py --gpu_device 0

   # 多GPU | Multi GPU
   python app.py --gpu_device "1,7"  # 使用GPU 1和7
   python app.py --gpu_device "0,1,2,3"  # 使用连续的GPU
   ```

2. **GPU内存优化 | GPU Memory Optimization**
   - 实际可用显存 ≈ (单卡显存 × GPU数量)
   - 内存参考值 | Memory Reference:
     ```
     3,000 tokens  →  ~8 GB VRAM
     22,000 tokens → ~12 GB VRAM
     41,000 tokens → ~16 GB VRAM
     78,000 tokens → ~24 GB VRAM
     ```
   - 优化建议 | Optimization Tips:
     1. 减小context length
     2. 减小batch size (max_num_seqs)
     3. 调整GPU内存使用率 (gpu_memory_utilization)
     4. 设置环境变量: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## 注意事项 | Notes

1. **模型说明 | Model Information**
   - 默认使用DeepSeek-R1-Distill-Qwen-32B
   - 不支持GGUF格式
   - 首次运行自动下载模型

2. **多GPU注意事项 | Multi-GPU Notes**
   - GPU编号用逗号分隔，不含空格
   - 自动启用张量并行(Tensor Parallelism)
   - 建议使用相同型号GPU

3. **上下文长度控制 | Context Length Control**
   - 通过UI设置Maximum context length
   - 实际长度 = min(设置值, 模型原始窗口)
   - 批处理优化：默认限制为min(context_length, 8192)

## 自定义扩展 | Customization

- 支持自定义模型路径 | Custom model paths
- 可集成RAG等功能 | RAG integration
- 可自定义UI组件 | UI customization
- 灵活的参数配置 | Parameter configuration
