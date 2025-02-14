"""Model downloading script for R1 Overthinker."""

import os
from huggingface_hub import snapshot_download

def download_model(model_name: str, local_dir: str = None):
    """Download model files from Hugging Face.
    
    Args:
        model_name: Name of the model to download
        local_dir: Local directory to save model files (defaults to model_name)
    """
    # Enable hf_transfer for faster downloads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    if local_dir is None:
        local_dir = f"models/{model_name}"
        
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir="",
            force_download=False,  # Avoid re-downloading
            resume_download=True   # Support download resumption
        )
        print(f"Successfully downloaded {model_name} to {local_dir}")
        return True
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False
        
if __name__ == "__main__":
    # Example usage
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    download_model(model_name)
