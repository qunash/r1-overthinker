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
    "gpu_device": "0"  # Unsloth只支持使用一个GPU
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
        """Get list of available DeepSeek R1 models."""
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
                    
    def load_model(self, model_name: str) -> Tuple[str, bool]:
        """Load specified model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple of (message, success)
        """
        if model_name == self.current_model_name:
            return "Model already loaded", True
            
        try:
            self.log_print(f"Loading model: {model_name}")
            self.unload_current_model()
            
            if not os.path.exists(f"models/{model_name}"):
                self.log_print("Downloading model files...")
                from load_model import download_model
                if not download_model(model_name, f"models/{model_name}"):
                    return "Failed to download model files", False
                    
            # 使用vllm加载模型
            self.model = LLM(
                model=model_name,
                download_dir=f"models/{model_name}",
                tensor_parallel_size=1,  # 单GPU
                max_num_seqs=32,  # 批处理大小
                max_num_batched_tokens=self.max_seq_length,
                trust_remote_code=True
            )
            
            self.current_model_name = model_name
            self.log_print(f"Successfully loaded {model_name}")
            return f"Successfully loaded {model_name}", True
            
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
            
    def set_context_length(self, length: int) -> bool:
        """Set maximum sequence length."""
        try:
            length = int(length)
            if length < MODEL_CONFIG["min_context_length"]:
                length = MODEL_CONFIG["min_context_length"]
            self.max_seq_length = length
            return True
        except:
            return False

###########################
# 文本生成
###########################

class TextGenerator:
    """Text generator class implementing thinking process detection and replacement."""
    
    def __init__(self, model_manager: ModelManager):
        """Initialize text generator with model manager."""
        self.model_manager = model_manager
        self.params = DEFAULT_GENERATION_PARAMS.copy()
        self.replacement_tokens = DEFAULT_REPLACEMENT_TOKENS
        
    def update_params(self, **kwargs) -> None:
        """Update generation parameters."""
        self.params.update(kwargs)
        
    def set_replacement_tokens(self, tokens: str) -> None:
        """Set replacement tokens for thinking process."""
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