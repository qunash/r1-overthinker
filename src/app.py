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
from utils import (
    ModelManager, TextGenerator,
    DEFAULT_GENERATION_PARAMS,
    DEFAULT_REPLACEMENT_TOKENS,
    LATEX_DELIMITERS,
    MODEL_CONFIG,
    UI_CONFIG
)

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = MODEL_CONFIG["gpu_device"]

# 初始化模型管理器和生成器
model_manager = ModelManager()
text_generator = TextGenerator(model_manager)

def load_selected_model(model_name: str, context_length: int):
    """包装函数，处理模型加载和进度更新
    
    Args:
        model_name: 要加载的模型名称
        context_length: 上下文长度
        
    Yields:
        更新界面组件的状态列表
    """
    # 首先禁用按钮和输入
    yield [
        "Loading model...",                   # model_info
        0,                                    # current_tab
        gr.update(interactive=False),         # load_button
        gr.update(interactive=False),         # model_dropdown
        gr.update(interactive=False)          # context_length
    ]

    # 设置上下文长度
    model_manager.set_context_length(context_length)
    message, success = model_manager.load_model(model_name)

    # 返回列表，按顺序对应输出组件
    yield [
        message,                          # model_info
        1 if success else 0,              # current_tab
        gr.update(interactive=True),      # load_button
        gr.update(interactive=True),      # model_dropdown
        gr.update(interactive=True)       # context_length
    ]

def update_global_params(
    min_tokens: int,
    max_tokens: int,
    max_swaps: int,
    replacements: str,
    temp: float,
    top_p: float,
    rep_pen: float
) -> None:
    """更新生成器参数
    
    Args:
        min_tokens: 最小思考token数
        max_tokens: 最大输出token数
        max_swaps: 最大思考扩展次数
        replacements: 思考提示词
        temp: 生成温度
        top_p: 采样阈值
        rep_pen: 重复惩罚系数
    """
    text_generator.update_params(
        min_thinking_tokens=min_tokens,
        max_output_tokens=max_tokens,
        max_swaps=max_swaps,
        temperature=temp,
        top_p=top_p,
        repetition_penalty=rep_pen
    )
    text_generator.set_replacement_tokens(replacements)

def create_demo() -> gr.Blocks:
    """创建Gradio界面
    
    Returns:
        Gradio Blocks实例
    """
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
            # 模型选择标签页
            with gr.Tab("1. Choose Model"):
                model_dropdown = gr.Dropdown(
                    choices=model_manager.get_available_models(),
                    label="Select DeepSeek R1 Model",
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

                # 创建带自动刷新的日志组件
                loading_log = Log(
                    model_manager.log_file,
                    dark=True,
                    xterm_font_size=14,
                    label="Loading Progress",
                    every=0.5
                )

                # 更新模型加载时的模型信息
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

            # 聊天界面标签页
            with gr.Tab("2. Chat", id=1):
                with gr.Row():
                    # 聊天区域
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

                    # 参数调整区域
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

                        # 连接所有参数组件到更新函数
                        for param in [min_tokens, max_tokens, max_swaps, replacements, temperature, top_p, rep_penalty]:
                            param.change(
                                fn=update_global_params,
                                inputs=[min_tokens, max_tokens, max_swaps, replacements, temperature, top_p, rep_penalty],
                                outputs=[]
                            )

        gr.HTML("""
        <div style="border-top: 1px solid #303030;">
          <br>
          <p>App by: <a href="https://twitter.com/hahahahohohe"><img src="https://img.shields.io/twitter/follow/hahahahohohe?label=%40anzorq&style=social" alt="Twitter Follow"></a></p><br>
          <p>Enjoying this app? Please consider <a href="https://www.buymeacoffee.com/anzorq">supporting me</a></p>
          <a href="https://www.buymeacoffee.com/anzorq" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 45px !important;width: 162px !important;" ></a><br><br>
          <a href="https://github.com/qunash/r1-overthinker" target="_blank"><img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/qunash/r1-overthinker?style=social"></a>
        </div>
        """)

    return demo

if __name__ == "__main__":
    # 禁用 uvicorn 的彩色日志
    uvicorn.config.LOGGING_CONFIG["formatters"]["default"]["use_colors"] = False
    uvicorn.config.LOGGING_CONFIG["formatters"]["access"]["use_colors"] = False
    
    demo = create_demo()
    demo.launch(debug=True, share=True) 