[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qunash/r1-overthinker/blob/master/r1_overthinker.ipynb)
# **DeepSeek R1 Overthinker**
Using this app you can force [DeepSeek R1](https://api-docs.deepseek.com/news/news250120) models to think more deeply by extending their reasoning process. It uses [unsloth](https://github.com/unsloth/unsloth) optimized models for better performance and unlimited context length (only limited by available VRAM).

The app works by detecting when the model tries to conclude thoughts too early and replacing those with prompts that encourage additional reasoning, continuing until a minimum threshold of thinking set by you is reached.

<br>
<br>

App by [anzorq](https://twitter.com/hahahahohohe). If you like it, please consider supporting me:

[<a href="https://www.buymeacoffee.com/anzorq" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" height="45px" width="162px" alt="Buy Me A Coffee"></a>](https://www.buymeacoffee.com/anzorq)

---
<img width="891" alt="image" src="https://github.com/user-attachments/assets/05d1c32d-de56-446a-b8b7-e7e51fa32b18" />

## Features
- 🤔 Force models to think longer and more thoroughly
- 🔄 Customizable reasoning extensions and thinking thresholds
- 🎯 Fine-grained control over model parameters (temperature, top-p, etc.)
- 💭 Visible thinking process with token count tracking
- 📝 LaTeX support for mathematical expressions
- 🖥️ Optimized for various VRAM configurations
- ♾️ Unlimited context length (VRAM-dependent)
- 🔄 Choose from multiple model sizes (1.5B to 70B parameters)

## Available Models
You can choose from any of the [unsloth-optimized distilled DeepSeek R1 models](https://huggingface.co/models?search=unsloth%20r1):

### Qwen-based Models
- 1.5B parameters (Qwen): [unsloth/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B)
- 7B parameters (Qwen): [unsloth/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B)
- 14B parameters (Qwen): [unsloth/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B)
- 32B parameters (Qwen): [unsloth/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B)

### LLaMA-based Models
- 8B parameters (LLaMA): [unsloth/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B)
- 70B parameters (LLaMA): [unsloth/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-70B)

Choose the model size based on your available VRAM and performance requirements. Larger models generally provide better quality responses but require more VRAM. Qwen and LLaMA architectures may perform differently on various tasks.

> **Note**: You can run models up to 14B parameters on a free Google Colab T4 GPU.

## Related Work
### s1: Simple test-time scaling
The paper "s1: Simple test-time scaling" is an independent work by Niklas Muennighoff et al. that tests and validates the approach used in this repository. The key contributions of the paper include:
- Developing budget forcing to control test-time compute by forcefully terminating the model’s thinking process or lengthening it by appending “Wait” multiple times to the model’s generation.
- Curating a small dataset s1K of 1,000 questions paired with reasoning traces.
- Achieving strong reasoning performance and test-time scaling with the Qwen2.5-32B-Instruct language model.

For more details, see the [paper's repository](https://github.com/simplescaling/s1).

## Credits
- Original idea and implementation - [vgel's gist](https://gist.github.com/vgel/8a2497dc45b1ded33287fa7bb6cc1adc)
- DeepSeek LLM - https://github.com/deepseek-ai/DeepSeek-LLM
- unsloth - https://github.com/unsloth/unsloth
- Gradio - https://github.com/gradio-app/gradio

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

<br>

[![Visitors](https://api.visitorbadge.io/api/visitors?path=qunash%2Fr1-overthinker&labelColor=%23d9e3f0&countColor=%23263759)](https://visitorbadge.io/status?path=qunash%2Fr1-overthinker) 
