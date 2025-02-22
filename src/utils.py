"""Utility module for R1 Overthinker, including configuration, model management and text generation."""

import os
import torch
import gc
import sys
import logging
import random
from typing import List, Dict, Generator, Tuple, Optional
from huggingface_hub import HfApi
from gradio import ChatMessage
from vllm import LLM, SamplingParams

###########################
# 配置
###########################

DEFAULT_GENERATION_PARAMS = {
    "min_thinking_tokens": 1024,
    "max_output_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "max_swaps": -1
}

DEFAULT_REPLACEMENT_TOKENS = """Hold on - what if we challenge our initial assumptions?
Let me try a completely different approach to this problem
What would be a strong counter-argument to my current reasoning?
Let's validate each step of my previous logic
Could there be edge cases we haven't considered?
What if we work backwards from the expected result?
Are there any hidden constraints we're missing?
Let's try to disprove our current answer
Let me break this down into smaller sub-problems
Is there a more elegant solution we're overlooking?
What patterns emerge if we look at extreme cases?
Could we solve this using an entirely different domain?"""

LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\begin{equation}", "right": "\\end{equation}", "display": True},
    {"left": "\\begin{align}", "right": "\\end{align}", "display": True},
    {"left": "\\begin{alignat}", "right": "\\end{alignat}", "display": True},
    {"left": "\\begin{gather}", "right": "\\end{gather}", "display": True},
    {"left": "\\begin{CD}", "right": "\\end{CD}", "display": True},
    {"left": "\\boxed{", "right": "}", "display": False},
    {"left": "\\frac{", "right": "}", "display": False},
    {"left": "\\sqrt{", "right": "}", "display": False}
]

MODEL_CONFIG = {
    "default_context_length": 21848,
    "min_context_length": 1024,
    "gpu_device": "0,1",  # GPU配置
    "model_path": None  # 将通过命令行参数设置
}

UI_CONFIG = {
    "chatbot_min_height": 550,
    "max_batch_tokens": 20
}

###########################
# 模型管理
###########################

class ModelManager:
    """Model manager class for loading, unloading and managing model state."""
    
    def __init__(self):
        """Initialize model manager."""
        self.current_model_name: Optional[str] = None
        self.model: Optional[LLM] = None
        self.tokenizer = None  # vllm内部包含tokenizer
        self.max_seq_length = MODEL_CONFIG["default_context_length"]
        self.log_file = "loading.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        open(self.log_file, 'w').close()
        
    def log_print(self, text: str) -> None:
        """Log message and print to console."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(text + "\n")
        print(text)
        
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        if MODEL_CONFIG["model_path"]:
            # 如果指定了本地模型路径，直接返回
            return [MODEL_CONFIG["model_path"]]
            
        # 否则搜索在线模型
        try:
            api = HfApi()
            models = api.list_models(
                search="DeepSeek-R1",
                author="deepseek-ai",  # 使用官方模型
                sort="downloads",
                direction=-1
            )
            model_options = sorted([model.id for model in models])
            if not model_options:
                self.log_print("No DeepSeek-R1 models found")
            return model_options
        except Exception as e:
            self.log_print(f"Error fetching models: {str(e)}")
            return []
            
    def unload_current_model(self) -> None:
        """Unload current model and clean up memory."""
        if self.model is not None:
            try:
                del self.model
                self.model = None
                self.current_model_name = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                self.log_print(f"Error during model unloading: {str(e)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
    def load_model(self, model_path: str) -> Tuple[str, bool]:
        """Load specified model.
        
        Args:
            model_path: Path to the model
            
        Returns:
            Tuple of (message, success)
        """
        if model_path == self.current_model_name:
            return "Model already loaded", True
            
        try:
            self.log_print(f"Loading model from: {model_path}")
            self.unload_current_model()
            
            # 设置GPU设备
            gpu_ids = MODEL_CONFIG["gpu_device"].replace(" ", "").split(",")
            gpu_count = len(gpu_ids)
            
            # 设置CUDA可见设备
            os.environ["CUDA_VISIBLE_DEVICES"] = MODEL_CONFIG["gpu_device"]
            self.log_print(f"Setting CUDA_VISIBLE_DEVICES={MODEL_CONFIG['gpu_device']}")
            self.log_print(f"Using {gpu_count} GPU(s): {gpu_ids}")
            
            # 确保max_num_batched_tokens >= max_model_len
            max_num_batched_tokens = max(self.max_seq_length, 8192)
            self.log_print(f"Setting max_num_batched_tokens to {max_num_batched_tokens}")
            
            # 使用vllm加载模型
            self.model = LLM(
                model=model_path,  # 可以是本地路径或模型名称
                tensor_parallel_size=gpu_count,  # 设置为实际的GPU数量
                max_num_seqs=32,  # 批处理大小
                max_model_len=self.max_seq_length,  # 设置最大上下文长度
                max_num_batched_tokens=max_num_batched_tokens,  # 确保大于等于max_model_len
                trust_remote_code=True,
                enforce_eager=True,  # 强制即时执行，避免编译开销
                #gpu_memory_utilization=0.8,  # 控制每个GPU的内存使用率
                skip_tokenizer_init=True  # 添加此参数
            )
            
            self.current_model_name = model_path
            self.log_print(f"Successfully loaded model from {model_path} with context length {self.max_seq_length}")
            return f"Successfully loaded model", True
            
        except Exception as e:
            self.log_print(f"Error loading model: {str(e)}")
            return f"Error loading model: {str(e)}", False
            
    def get_current_model_info(self) -> str:
        """Get current model information."""
        return "No model currently loaded" if self.current_model_name is None else f"Currently loaded model: {self.current_model_name}"
        
    def force_gpu_cleanup(self) -> None:
        """Force GPU memory cleanup."""
        self.unload_current_model()
        if torch.cuda.is_available():
            initial = torch.cuda.memory_allocated()
            gc.collect()
            torch.cuda.empty_cache()
            final = torch.cuda.memory_allocated()
            self.log_print(f"GPU Memory freed: {(initial - final) / 1024**2:.2f} MB")
            
    def set_context_length(self, context_length: int) -> None:
        """设置上下文长度。
        
        Args:
            context_length: 新的上下文长度
        """
        if context_length < MODEL_CONFIG["min_context_length"]:
            self.log_print(f"Warning: context length {context_length} is less than minimum {MODEL_CONFIG['min_context_length']}")
            context_length = MODEL_CONFIG["min_context_length"]
        self.max_seq_length = context_length

###########################
# 文本生成
###########################

class TextGenerator:
    """Text generator class for managing text generation."""
    
    def __init__(self, model_manager: ModelManager):
        """Initialize text generator.
        
        Args:
            model_manager: ModelManager实例
        """
        self.model_manager = model_manager
        self.params = DEFAULT_GENERATION_PARAMS.copy()
        self.replacement_tokens = DEFAULT_REPLACEMENT_TOKENS
        
    def update_params(self, **kwargs) -> None:
        """更新生成参数。
        
        Args:
            **kwargs: 要更新的参数键值对
        """
        self.params.update(kwargs)
        
    def set_replacement_tokens(self, tokens: str) -> None:
        """设置替换token列表。
        
        Args:
            tokens: 换行符分隔的token列表
        """
        self.replacement_tokens = tokens
        
    def generate(self, message: str, history: List[Dict]) -> Generator[List[ChatMessage], None, None]:
        """Generate response with extended thinking process.
        
        Args:
            message: User input message
            history: Chat history
            
        Yields:
            List of ChatMessage objects containing thinking process and final response
        """
        if self.model_manager.model is None:
            yield [ChatMessage(role="assistant", content="Please load a model first in the 'Choose Model' tab")]
            return
            
        try:
            # Convert history to messages format
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
            messages.append({"role": "user", "content": message})
            
            # 构建prompt
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"Human: {msg['content']}\n\nAssistant: "
                else:
                    prompt += f"{msg['content']}\n\n"
            
            # 设置采样参数
            sampling_params = SamplingParams(
                temperature=self.params["temperature"],
                top_p=self.params["top_p"],
                repetition_penalty=self.params["repetition_penalty"],
                max_tokens=self.params["max_output_tokens"]
            )
            
            thinking_msg = None
            final_msg = None
            thinking_content = ""
            final_content = ""
            n_thinking_tokens = 0
            swap_count = 0
            is_thinking = False
            
            # 使用vllm生成
            outputs = self.model_manager.model.generate(prompt, sampling_params)
            
            for output in outputs:
                text = output.outputs[0].text
                
                # 检测思考标记
                if "<think>" in text:
                    is_thinking = True
                    thinking_msg = ChatMessage(
                        role="assistant",
                        content="",
                        metadata={"title": "🤔 Thinking process"}
                    )
                    continue
                elif "</think>" in text:
                    is_thinking = False
                    final_msg = ChatMessage(role="assistant", content="")
                    
                    # 检查是否需要扩展思考
                    if n_thinking_tokens < self.params["min_thinking_tokens"] and \
                       (self.params["max_swaps"] == -1 or swap_count < self.params["max_swaps"]):
                        replacement = "\n" + random.choice([t.strip() for t in self.replacement_tokens.split('\n') if t.strip()])
                        thinking_content += replacement
                        n_thinking_tokens += len(replacement.split())
                        swap_count += 1
                        
                        if thinking_msg:
                            thinking_msg.content = thinking_content
                            thinking_msg.metadata = {
                                "title": f"🤔 Thinking process (Extensions: {swap_count}, Tokens: {n_thinking_tokens})"
                            }
                            yield [thinking_msg]
                        continue
                
                if is_thinking:
                    thinking_content += text
                    n_thinking_tokens += len(text.split())
                    if thinking_msg:
                        thinking_msg.content = thinking_content
                        thinking_msg.metadata = {
                            "title": f"🤔 Thinking process (Extensions: {swap_count}, Tokens: {n_thinking_tokens})"
                        }
                        yield [thinking_msg]
                else:
                    final_content += text
                    if final_msg:
                        messages = []
                        if thinking_msg:
                            messages.append(thinking_msg)
                        final_msg.content = final_content.strip()
                        messages.append(final_msg)
                        yield messages
            
            # 生成完成后的最终输出
            messages = []
            if thinking_msg:
                thinking_msg.content = thinking_content
                messages.append(thinking_msg)
            if final_msg:
                final_msg.content = final_content.strip()
                messages.append(final_msg)
            yield messages
            
        except Exception as e:
            yield [ChatMessage(role="assistant", content=f"Error during generation: {str(e)}")] 