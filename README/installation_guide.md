# Installation Guide

## Installation Steps

### 1. Upgrade pip and setuptools
```bash
pip install --upgrade pip setuptools
```

### 2. Install dependencies

#### Option A: Install from requirements.txt (Recommended first attempt)
```bash
pip install -r requirements.txt
```

#### Option B: If requirements.txt fails due to dependency conflicts
If you encounter dependency resolution issues, try installing these packages separately in this order:

```bash
# Install AutoRAG with GPU support
python -m pip install AutoRAG[gpu]

# Install vLLM
python -m pip install vllm

# Install spacy with transformers
python -m pip install spacy[transformers]

# Install remaining packages
pip install smac ConfigSpace ragas wandb huggingface-hub optuna plotly
pip install FlagEmbedding onnxruntime optimum[openvino,nncf]
pip install optuna-integration[wandb]
pip install python-dotenv matplotlib pandas numpy scipy kaleido Pillow
```

### 3. Download Spacy Language Models
After installing spacy, download the required language models if compressor includes spacy:

```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_trf
```

## Environment Configuration

### 1. Set API Keys and Tokens

#### OpenAI API Key (Required for LLM operations)
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

#### Hugging Face Token (Required for vllm models and vectorDB)
```bash
export HF_TOKEN="your-huggingface-token-here"
# or
export HUGGINGFACE_HUB_TOKEN="your-huggingface-token-here"
```

#### Email Credentials (For experiment notifications)
Create a `.env` file in your project root:
```env
EMAIL_SENDER=your.email@gmail.com
EMAIL_PASSWORD=your-app-password-here
EMAIL_RECIPIENTS=recipient1@email.com,recipient2@email.com
```

**Note**: For Gmail, use an App Password, not your regular password. See Email Notifier README for details.

### 2. Configure WandB (Weights & Biases)

Login to your WandB account:
```bash
wandb login
```

You'll be prompted to enter your API key, which you can find at: https://wandb.ai/authorize

## Troubleshooting

### Common Issues

1. **Dependency conflicts**: If you encounter version conflicts, create a fresh virtual environment:
```bash
python -m venv autorag_env
source autorag_env/bin/activate  # On Windows: autorag_env\Scripts\activate
```

2. **CUDA/GPU errors**: Ensure you have the correct CUDA version installed for your system. Check with:
```bash
nvidia-smi
```

3. **Memory issues with vLLM**: vLLM requires significant GPU memory. Ensure you have at least 16GB of GPU memory for most models.

4. **Spacy model download fails**: If spacy model downloads fail, try installing them individually with admin privileges or in a virtual environment.

5. **WandB login issues**: If `wandb login` fails, you can set the API key directly:
```bash
export WANDB_API_KEY="your-wandb-api-key"
```

## Important Notes

- Ensure all API keys are kept secure and never committed to version control
- GPU support requires appropriate CUDA drivers and compatible hardware