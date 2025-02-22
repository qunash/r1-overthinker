"""
主应用程序入口

此模块实现了基于 Gradio 的 Web 界面，包括:
1. 模型选择和加载界面
2. 参数调整界面
3. 聊天界面
"""

import os
import gradio as gr
from gradio import ChatMessage
from gradio_log import Log
import uvicorn.config
import argparse
import multiprocessing
from utils import (
    ModelManager, TextGenerator,
    DEFAULT_GENERATION_PARAMS,
    DEFAULT_REPLACEMENT_TOKENS,
    LATEX_DELIMITERS,
    MODEL_CONFIG,
    UI_CONFIG
)
import torch

# 设置vLLM的多进程方法
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="R1 Overthinker Application")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to local model directory")
    parser.add_argument("--gpu_device", type=str, default="0",
                       help="GPU device to use (default: 0)")
    return parser.parse_args()

def load_selected_model(model_name: str, context_length: int):
    """加载选定的模型。
    
    Args:
        model_name: 要加载的模型名称
        context_length: 上下文长度
        
    Returns:
        tuple: (model_info, next_tab, load_button_state, model_dropdown_state, context_length_state)
    """
    if not model_name:
        return "请选择一个模型", 0, gr.update(interactive=True), \
               gr.update(interactive=True), gr.update(interactive=True)
    
    model_manager = ModelManager()
    model_manager.set_context_length(context_length)
    message, success = model_manager.load_model(model_name)
    
    if success:
        return message, 1, gr.update(interactive=False), \
               gr.update(interactive=False), gr.update(interactive=False)
    else:
        return message, 0, gr.update(interactive=True), \
               gr.update(interactive=True), gr.update(interactive=True)

def update_global_params(min_tokens: int, max_tokens: int, max_swaps: int, 
                        replacements: str, temperature: float, top_p: float, 
                        rep_penalty: float):
    """更新全局生成参数。
    
    Args:
        min_tokens: 最小思考token数
        max_tokens: 最大输出token数
        max_swaps: 最大替换次数
        replacements: 替换token列表
        temperature: 温度参数
        top_p: top-p采样参数
        rep_penalty: 重复惩罚系数
    """
    text_generator = TextGenerator(ModelManager())
    text_generator.update_params(
        min_thinking_tokens=min_tokens,
        max_output_tokens=max_tokens,
        max_swaps=max_swaps,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=rep_penalty
    )
    text_generator.set_replacement_tokens(replacements)

def main():
    """Main entry point."""
    args = parse_args()
    
    # Update config with command line arguments
    if args.model_path:
        MODEL_CONFIG["model_path"] = args.model_path
    MODEL_CONFIG["gpu_device"] = args.gpu_device
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 设置CUDA可见设备
    os.environ["CUDA_VISIBLE_DEVICES"] = MODEL_CONFIG["gpu_device"]
    
    # 设置PyTorch内存分配器
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Initialize model manager and generator
    model_manager = ModelManager()
    text_generator = TextGenerator(model_manager)
    
    def get_model_list():
        """Get model list based on configuration."""
        if MODEL_CONFIG["model_path"]:
            return [MODEL_CONFIG["model_path"]]  # 使用本地模型
        return model_manager.get_available_models()  # 使用在线模型
    
    def create_demo():
        """Create and configure the Gradio interface."""
        with gr.Blocks() as demo:
            gr.Markdown("""
                # 🤔 DeepSeek R1 Overthinker
                
                Using this app you can force DeepSeek R1 models to think more deeply, for as long as you wish. It works like this:
                - Detects when the model tries to conclude thoughts too early (`</think>` token)
                - Replaces those with prompts that encourage additional reasoning
                - Continues until a minimum threshold of thinking is reached
                
                You decide how long the model should think. The result is more thorough and well-reasoned responses (hopefully).
            """)
            
            current_tab = gr.State(value=0)
            
            with gr.Tabs() as tabs:
                with gr.Tab("1. Choose Model"):
                    model_dropdown = gr.Dropdown(
                        choices=get_model_list(),
                        label="Select Model",
                        interactive=True
                    )
                    
                    context_length = gr.Number(
                        value=MODEL_CONFIG["default_context_length"],
                        label="Maximum context length (in tokens)",
                        precision=0,
                        minimum=MODEL_CONFIG["min_context_length"],
                        info="""Higher values require more VRAM. Reference values from Llama 3.1 testing:
                        3,000 tokens → ~8 GB VRAM
                        22,000 tokens → ~12 GB VRAM
                        41,000 tokens → ~16 GB VRAM
                        78,000 tokens → ~24 GB VRAM
                        154,000 tokens → ~40 GB VRAM
                        
                        Actual VRAM usage depends on the model. Start with a lower value if you experience out-of-memory errors."""
                    )
                    
                    load_button = gr.Button("Load Selected Model")
                    model_info = gr.Markdown(model_manager.get_current_model_info())
                    
                    loading_log = Log(
                        model_manager.log_file,
                        dark=True,
                        xterm_font_size=14,
                        label="Loading Progress",
                        every=0.5
                    )
                    
                    load_button.click(
                        fn=load_selected_model,
                        inputs=[model_dropdown, context_length],
                        outputs=[
                            model_info,           # Markdown
                            current_tab,          # State
                            load_button,          # Button
                            model_dropdown,       # Dropdown
                            context_length        # Number
                        ],
                        queue=True
                    ).success(
                        fn=lambda tab: gr.Tabs(selected=tab),
                        inputs=[current_tab],
                        outputs=[tabs]
                    )
                    
                with gr.Tab("2. Chat", id=1):
                    with gr.Row():
                        with gr.Column(scale=3):
                            chat_interface = gr.ChatInterface(
                                text_generator.generate,
                                chatbot=gr.Chatbot(
                                    bubble_full_width=True,
                                    show_copy_button=True,
                                    latex_delimiters=LATEX_DELIMITERS,
                                    type="messages",
                                    editable='all',
                                    min_height=UI_CONFIG["chatbot_min_height"],
                                ),
                                fill_height=True,
                                type="messages"
                            )
                            
                        with gr.Column(scale=1):
                            min_tokens = gr.Slider(
                                minimum=0,
                                maximum=model_manager.max_seq_length-512,
                                value=DEFAULT_GENERATION_PARAMS["min_thinking_tokens"],
                                step=128,
                                label="Minimum Thinking Tokens"
                            )
                            max_tokens = gr.Slider(
                                minimum=256,
                                maximum=model_manager.max_seq_length,
                                value=DEFAULT_GENERATION_PARAMS["max_output_tokens"],
                                step=256,
                                label="Maximum Output Tokens"
                            )
                            max_swaps = gr.Slider(
                                minimum=-1,
                                maximum=20,
                                value=DEFAULT_GENERATION_PARAMS["max_swaps"],
                                step=1,
                                label="Maximum Reasoning Extensions",
                                info="Limit how many times to extend the model's reasoning (-1 for unlimited)."
                            )
                            replacements = gr.Textbox(
                                value=DEFAULT_REPLACEMENT_TOKENS,
                                label="Replacement Tokens (one per line)",
                                lines=5,
                                max_lines=5
                            )
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=DEFAULT_GENERATION_PARAMS["temperature"],
                                step=0.1,
                                label="Temperature"
                            )
                            top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=DEFAULT_GENERATION_PARAMS["top_p"],
                                step=0.05,
                                label="Top-p"
                            )
                            rep_penalty = gr.Slider(
                                minimum=1.0,
                                maximum=2.0,
                                value=DEFAULT_GENERATION_PARAMS["repetition_penalty"],
                                step=0.1,
                                label="Repetition Penalty"
                            )
                            
                            for param in [min_tokens, max_tokens, max_swaps, replacements,
                                        temperature, top_p, rep_penalty]:
                                param.change(
                                    fn=update_global_params,
                                    inputs=[min_tokens, max_tokens, max_swaps, replacements,temperature, top_p, rep_penalty],
                                    outputs=[]
                                )
                                
            gr.HTML("""
            <div style="border-top: 1px solid #303030;">
                <br>
                <p>App by: <a href="https://twitter.com/hahahahohohe">
                <img src="https://img.shields.io/twitter/follow/hahahahohohe?label=%40anzorq&style=social" alt="Twitter Follow"></a></p><br>
                <p>Enjoying this app? Please consider <a href="https://www.buymeacoffee.com/anzorq">supporting me</a></p>
                <a href="https://www.buymeacoffee.com/anzorq" target="_blank">
                <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" 
                style="height: 45px !important;width: 162px !important;" ></a><br><br>
                <a href="https://github.com/qunash/r1-overthinker" target="_blank">
                <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/qunash/r1-overthinker?style=social"></a>
            </div>
            """)
            
        return demo
    
    # Disable uvicorn colored logging
    uvicorn.config.LOGGING_CONFIG["formatters"]["default"]["use_colors"] = False
    uvicorn.config.LOGGING_CONFIG["formatters"]["access"]["use_colors"] = False
    
    demo = create_demo()
    demo.launch(debug=True, share=True)

if __name__ == "__main__":
    # 确保在主进程中运行
    multiprocessing.freeze_support()
    main() 