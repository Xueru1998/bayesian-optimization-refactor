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
# Install AutoRAG
python -m pip install AutoRAG

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
After installing spacy, download the required language models:

```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_trf
```

## Environment Configuration

### 1. SAP API Credentials (Required for Model Access)

Create a `.env` file in your project root with the following credentials:

```env
# SAP credentials for embedding and generator models
SAP_CLIENT_ID='your-sap-client-id'
SAP_CLIENT_SECRET='your-sap-client-secret'
SAP_AUTH_URL='your-sap-auth-url'

# SAP credentials for reranker models
SAP_RERANKER_AUTH_URL='your-sap-reranker-auth-url'
SAP_RERANKER_CLIENT_ID='your-sap-reranker-client-id'
SAP_RERANKER_CLIENT_SECRET='your-sap-reranker-client-secret'
```

**Note**: The system will automatically handle token refresh for all SAP API calls.

### 2. Model Deployment URLs

When configuring your models in the YAML configuration, you must include the deployed model URLs:

- **Embedding Models**: Include the full deployment URL in the `embedding_model` field
- **Generator Models**: Include the full deployment URL in the `api_url` field
- **Reranker Models**: Include the full deployment URL in the `api-url` field

### 3. Additional API Keys (Optional)

#### OpenAI API Key (If using OpenAI models)
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

#### Hugging Face Token (If using HF models)
```bash
export HF_TOKEN="your-huggingface-token-here"
# or
export HUGGINGFACE_HUB_TOKEN="your-huggingface-token-here"
```

### 4. Email Credentials (For experiment notifications)
Add to your `.env` file:
```env
EMAIL_SENDER=your.email@gmail.com
EMAIL_PASSWORD=your-app-password-here
EMAIL_RECIPIENTS=recipient1@email.com,recipient2@email.com
```

**Note**: For Gmail, use an App Password, not your regular password. See Email Notifier README for details.

### 5. Configure WandB (Weights & Biases)

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

2. **SAP API Authentication errors**: 
   - Verify all SAP credentials are correctly set in the `.env` file
   - Ensure the auth URLs are accessible from your network
   - Check that client ID and secret pairs match their respective auth URLs

3. **Model deployment URL errors**:
   - Verify the deployment URLs are complete and include the deployment ID
   - Check that the URL format matches the model type (embeddings vs. chat completions vs. rerank)

4. **WandB login issues**: If `wandb login` fails, you can set the API key directly:
```bash
export WANDB_API_KEY="your-wandb-api-key"
```
