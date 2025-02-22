import os
from vllm import LLM, SamplingParams
import torch

def test_model():
    try:
        # 设置CUDA设备
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        # 设置PyTorch内存分配器
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        print("Step 1: Importing vLLM successful")
        
        # 初始化模型
        print("Step 2: Initializing model...")
        model = LLM(
            model="/data1/hao/models/DeepSeek-R1-Distill-Qwen-32B",
            tensor_parallel_size=4,  # 使用4张GPU
            trust_remote_code=True  
        )
        
        print("Step 3: Model loaded successfully!")
        
        # 简单的推理测试
        print("\nStep 4: Testing inference...")
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=100,
            top_p=0.95,
        )
        
        prompts = ["Hello, who are you?"]
        outputs = model.generate(prompts, sampling_params)
        
        # 打印结果
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\nPrompt: {prompt}")
            print(f"Generated text: {generated_text}")
            
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    test_model()
