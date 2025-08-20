import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from datetime import datetime, timedelta
import threading

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    
    def __init__(self, project_dir: str, **kwargs):
        self.project_dir = project_dir
        self.kwargs = kwargs
    
    @abstractmethod
    def _pure(self, prompts: List[str], **kwargs) -> tuple:
        pass
    
    def pure(self, previous_result: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if "prompts" not in previous_result.columns:
            raise ValueError("DataFrame must have 'prompts' column")
        
        prompts = previous_result["prompts"].tolist()
        generated_texts, usages, _ = self._pure(prompts, **kwargs)
        
        result_df = previous_result.copy()
        result_df["generated_texts"] = generated_texts
        return result_df
    
    async def astream(self, prompt: str, **kwargs):
        raise NotImplementedError("Streaming not supported for this generator")
    
    def structured_output(self, prompts: List[str], output_class):
        raise NotImplementedError("Structured output not supported for this generator")


class OpenAILLM(BaseGenerator):
    
    def __init__(self, project_dir: str, llm: str = "gpt-4o-mini", batch: int = 8, **kwargs):
        super().__init__(project_dir, **kwargs)
        self.model = llm
        self.batch = batch
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def _pure(self, prompts: List[str], **kwargs) -> tuple:
        results = []
        usages = []
        
        params = {
            "model": self.model,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 500),
            "top_p": kwargs.get("top_p", 1.0),
        }
        
        for i in range(0, len(prompts), self.batch):
            batch_prompts = prompts[i:i + self.batch]
            
            for prompt in batch_prompts:
                try:
                    response = self.client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        **params
                    )
                    results.append(response.choices[0].message.content)
                    
                    if hasattr(response, 'usage'):
                        usages.append({
                            'prompt_tokens': response.usage.prompt_tokens,
                            'completion_tokens': response.usage.completion_tokens,
                            'total_tokens': response.usage.total_tokens
                        })
                    else:
                        usages.append({})
                        
                except Exception as e:
                    logger.error(f"OpenAI API error: {e}")
                    results.append("")
                    usages.append({})
        
        return results, usages, []


class SAPAPI(BaseGenerator):
    _shared_token = None
    _shared_token_expiry = None
    _token_lock = None
    _rate_limit_lock = None
    _last_429_time = None
    _consecutive_429_count = 0
    
    def __init__(self, project_dir: str, llm: str = "mistralai", 
                 model: str = "mistralai-large-instruct",
                 api_url: str = None, bearer_token: str = None, 
                 batch: int = 8, temperature: float = 0.7,
                 max_tokens: int = 500, **kwargs):
        super().__init__(project_dir, **kwargs)
        self.llm = llm
        self.model = model
        self.batch = batch
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.max_retries = 5
        self.initial_wait_time = 60
        self.max_wait_time = 300

        self.api_url = self._resolve_api_url(api_url)
        
        self.auth_url = os.getenv('SAP_AUTH_URL')
        self.client_id = os.getenv('SAP_CLIENT_ID')
        self.client_secret = os.getenv('SAP_CLIENT_SECRET')
        
        if not self.auth_url:
            raise ValueError("SAP_AUTH_URL not found in .env file. Please set SAP_AUTH_URL in your .env file.")
        if not self.client_id:
            raise ValueError("SAP_CLIENT_ID not found in .env file. Please set SAP_CLIENT_ID in your .env file.")
        if not self.client_secret:
            raise ValueError("SAP_CLIENT_SECRET not found in .env file. Please set SAP_CLIENT_SECRET in your .env file.")
        
        logger.debug(f"SAP API URL resolved to: {self.api_url}")
        
        if not self.api_url:
            raise ValueError("SAP API URL is required. It must be provided in the config file")
        
        if SAPAPI._token_lock is None:
            import threading
            SAPAPI._token_lock = threading.Lock()
        
        if SAPAPI._rate_limit_lock is None:
            SAPAPI._rate_limit_lock = threading.Lock()
        
        self._ensure_valid_token()
    
    def _resolve_api_url(self, api_url: str) -> str:
        if not api_url:
            logger.error("SAP API URL is required. It must be provided in the config file.")
            return None
        
        logger.debug(f"Using API URL from config: {api_url}")
        return api_url
    
    def _refresh_token(self):
        with SAPAPI._token_lock:
            if SAPAPI._shared_token and SAPAPI._shared_token_expiry and datetime.now() < SAPAPI._shared_token_expiry:
                logger.debug("Using existing valid token from cache")
                return
            
            try:
                headers = {
                    'Content-Type': 'application/x-www-form-urlencoded'
                }
                
                data = {
                    'grant_type': 'client_credentials',
                    'client_id': self.client_id,
                    'client_secret': self.client_secret
                }
                
                logger.info("Refreshing SAP bearer token...")
                response = requests.post(
                    self.auth_url,
                    headers=headers,
                    data=data,
                    timeout=30
                )
                response.raise_for_status()
                
                token_data = response.json()
                SAPAPI._shared_token = token_data.get('access_token')
                expires_in = token_data.get('expires_in', 43199)
                
                SAPAPI._shared_token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
                
                logger.info(f"Token refreshed successfully. Expires in {expires_in} seconds")
                
            except Exception as e:
                logger.error(f"Failed to refresh token: {e}")
                raise ValueError(f"Failed to obtain SAP bearer token: {e}")
    
    def _ensure_valid_token(self):
        if not SAPAPI._shared_token or (SAPAPI._shared_token_expiry and datetime.now() >= SAPAPI._shared_token_expiry):
            self._refresh_token()
        else:
            logger.debug("Token is still valid, no refresh needed")
    
    def _get_model_type(self) -> str:
        if any(x in self.model.lower() for x in ['gpt-4o-mini', 'gpt-35-turbo', 'gpt-3.5-turbo']):
            return 'openai'
        elif any(x in self.model.lower() for x in ['claude', 'anthropic']):
            return 'anthropic'
        elif any(x in self.model.lower() for x in ['gemini']):
            return 'gemini'
        elif any(x in self.model.lower() for x in ['mistral']):
            return 'mistral'
        else:
            return 'openai'
    
    def _build_request_body(self, prompt: str, generation_params: Dict) -> Dict:
        model_type = self._get_model_type()
        
        if model_type == 'openai' or model_type == 'mistral':
            return {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": generation_params.get("max_tokens", self.max_tokens),
                "temperature": generation_params.get("temperature", self.temperature)
            }
        
        elif model_type == 'anthropic':
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": generation_params.get("max_tokens", self.max_tokens),
                "messages": [{"role": "user", "content": prompt}]
            }
        
        elif model_type == 'gemini':
            return {
                "generation_config": {
                    "maxOutputTokens": generation_params.get("max_tokens", self.max_tokens),
                    "temperature": generation_params.get("temperature", self.temperature)
                },
                "contents": {
                    "role": "user",
                    "parts": {"text": prompt}
                }
            }
        
        else:
            return {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": generation_params.get("max_tokens", self.max_tokens),
                "temperature": generation_params.get("temperature", self.temperature)
            }
    
    def _extract_response_content(self, data: Dict) -> str:
        model_type = self._get_model_type()
        
        try:
            if model_type == 'openai' or model_type == 'mistral':
                if 'choices' in data and len(data['choices']) > 0:
                    return data['choices'][0]['message']['content']
            
            elif model_type == 'anthropic':
                if 'content' in data and len(data['content']) > 0:
                    return data['content'][0]['text']
            
            elif model_type == 'gemini':
                if 'candidates' in data and len(data['candidates']) > 0:
                    candidate = data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if len(parts) > 0 and 'text' in parts[0]:
                            return parts[0]['text']
            
            else:
                if 'choices' in data and len(data['choices']) > 0:
                    return data['choices'][0]['message']['content']
            
            logger.error(f"Could not extract content from response: {data}")
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting response content: {e}")
            return ""
    
    def _extract_usage_info(self, data: Dict) -> Dict:
        model_type = self._get_model_type()
        
        try:
            if model_type == 'openai' or model_type == 'mistral':
                if 'usage' in data:
                    return {
                        'prompt_tokens': data['usage'].get('prompt_tokens', 0),
                        'completion_tokens': data['usage'].get('completion_tokens', 0),
                        'total_tokens': data['usage'].get('total_tokens', 0)
                    }
            
            elif model_type == 'anthropic':
                if 'usage' in data:
                    return {
                        'prompt_tokens': data['usage'].get('input_tokens', 0),
                        'completion_tokens': data['usage'].get('output_tokens', 0),
                        'total_tokens': data['usage'].get('input_tokens', 0) + data['usage'].get('output_tokens', 0)
                    }
            
            elif model_type == 'gemini':
                if 'usageMetadata' in data:
                    usage = data['usageMetadata']
                    prompt_tokens = usage.get('promptTokenCount', 0)
                    completion_tokens = usage.get('candidatesTokenCount', 0)
                    return {
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': usage.get('totalTokenCount', 0)
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error extracting usage info: {e}")
            return {}
    
    def _handle_rate_limit(self, retry_attempt: int):
        with SAPAPI._rate_limit_lock:
            SAPAPI._last_429_time = datetime.now()
            SAPAPI._consecutive_429_count += 1
            
            wait_time = min(
                self.initial_wait_time * (2 ** retry_attempt),
                self.max_wait_time
            )
            
            if SAPAPI._consecutive_429_count > 3:
                wait_time = self.max_wait_time
            
            logger.warning(f"Rate limit hit (429). Waiting {wait_time} seconds before retry. "
                         f"Retry attempt: {retry_attempt + 1}/{self.max_retries}, "
                         f"Consecutive 429s: {SAPAPI._consecutive_429_count}")
            
            time.sleep(wait_time)
    
    def _reset_rate_limit_counter(self):
        with SAPAPI._rate_limit_lock:
            if SAPAPI._last_429_time and (datetime.now() - SAPAPI._last_429_time).seconds > 300:
                SAPAPI._consecutive_429_count = 0
    
    def _make_api_request(self, prompt: str, generation_params: Dict, retry_count: int = 0) -> tuple:
        self._ensure_valid_token()
        self._reset_rate_limit_counter()
        
        headers = {
            "ai-resource-group": "enterpriseAIgroup",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {SAPAPI._shared_token}"
        }
        
        request_body = self._build_request_body(prompt, generation_params)
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=request_body,
                    timeout=60
                )
                
                if response.status_code == 429:
                    self._handle_rate_limit(attempt)
                    continue
                
                if response.status_code == 401 and retry_count < 2:
                    logger.warning("Received 401 Unauthorized. Refreshing token and retrying...")
                    self._refresh_token()
                    return self._make_api_request(prompt, generation_params, retry_count + 1)
                
                response.raise_for_status()
                
                with SAPAPI._rate_limit_lock:
                    SAPAPI._consecutive_429_count = 0
                
                data = response.json()
                content = self._extract_response_content(data)
                usage = self._extract_usage_info(data)
                
                return content, usage
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    self._handle_rate_limit(attempt)
                    continue
                elif e.response.status_code == 401 and retry_count < 2:
                    logger.warning("Received 401 Unauthorized. Refreshing token and retrying...")
                    self._refresh_token()
                    return self._make_api_request(prompt, generation_params, retry_count + 1)
                else:
                    logger.error(f"SAP API HTTP error: {e}")
                    if attempt < self.max_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        logger.info(f"Retrying after {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    return "", {}
                    
            except requests.exceptions.Timeout as e:
                logger.error(f"SAP API timeout error: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    logger.info(f"Retrying after timeout in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return "", {}
                
            except Exception as e:
                logger.error(f"SAP API error: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    logger.info(f"Retrying after error in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return "", {}
        
        logger.error(f"Failed to get response after {self.max_retries} attempts")
        return "", {}
    
    def _pure(self, prompts: List[str], **kwargs) -> tuple:
        generation_params = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }
        
        results = []
        usages = []
        
        for i in range(0, len(prompts), self.batch):
            batch_prompts = prompts[i:i + self.batch]
            
            for prompt in batch_prompts:
                content, usage = self._make_api_request(prompt, generation_params)
                results.append(content)
                usages.append(usage)
                
                if SAPAPI._consecutive_429_count > 0:
                    time.sleep(2)
        
        return results, usages, []


class Vllm(BaseGenerator):
    
    def __init__(self, project_dir: str, llm: str, model: str = None, **kwargs):
        super().__init__(project_dir, **kwargs)
        self.llm = llm
        self.model = model or llm
        
        try:
            from vllm import LLM, SamplingParams
            
            vllm_kwargs = kwargs.copy()
            if 'gpu_memory_utilization' not in vllm_kwargs:
                vllm_kwargs['gpu_memory_utilization'] = 0.8
            
            self.vllm_model = LLM(model=self.model, **vllm_kwargs)
        except ImportError:
            raise ImportError("Please install vllm")
    
    def __del__(self):
        self.cleanup()
    
    def cleanup(self):
        if hasattr(self, 'vllm_model'):
            try:
                from vllm.distributed import destroy_model_parallel
                from vllm.distributed import destroy_distributed_environment
                import torch
                
                if hasattr(self.vllm_model, 'llm_engine'):
                    if hasattr(self.vllm_model.llm_engine, 'model_executor'):
                        self.vllm_model.llm_engine.model_executor.shutdown()
                
                del self.vllm_model
                
                try:
                    destroy_model_parallel()
                except:
                    pass
                
                try:
                    destroy_distributed_environment()
                except:
                    pass
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                
                import gc
                gc.collect()
                
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
                    
            except Exception as e:
                logger.error(f"Error during vLLM cleanup: {e}")
    
    def _pure(self, prompts: List[str], **kwargs) -> tuple:
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 256),
            top_p=kwargs.get("top_p", 1.0),
        )
        
        outputs = self.vllm_model.generate(prompts, sampling_params)
        
        results = []
        usages = []
        
        for output in outputs:
            results.append(output.outputs[0].text)
            usages.append({})
        
        return results, usages, []

class LlamaIndexLLM(BaseGenerator):
    
    def __init__(self, project_dir: str, llm: str = "openai", 
                 model: str = "gpt-4o-mini", batch: int = 8, **kwargs):
        super().__init__(project_dir, **kwargs)
        self.llm_provider = llm
        self.model = model
        self.batch = batch
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 256)
        
        self._init_llm()
    
    def _init_llm(self):
        try:
            from llama_index.core.llms import LLM
            
            if self.llm_provider == "openai":
                from llama_index.llms.openai import OpenAI
                self.llm = OpenAI(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            elif self.llm_provider == "anthropic":
                from llama_index.llms.anthropic import Anthropic
                self.llm = Anthropic(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            elif self.llm_provider == "cohere":
                from llama_index.llms.cohere import Cohere
                self.llm = Cohere(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
                
        except ImportError as e:
            raise ImportError(f"Please install required llama-index packages: {e}")
    
    def _pure(self, prompts: List[str], **kwargs) -> tuple:
        results = []
        usages = []
        
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        if temperature != self.temperature or max_tokens != self.max_tokens:
            self.temperature = temperature
            self.max_tokens = max_tokens
            self._init_llm()
        
        for i in range(0, len(prompts), self.batch):
            batch_prompts = prompts[i:i + self.batch]
            
            for prompt in batch_prompts:
                try:
                    response = self.llm.complete(prompt)
                    results.append(response.text)
                    usages.append({})
                except Exception as e:
                    logger.error(f"LlamaIndex LLM error: {e}")
                    results.append("")
                    usages.append({})
        
        return results, usages, []


def make_generator_callable_param(kwargs: Dict[str, Any]) -> tuple:
    
    generator_module_type = kwargs.pop("generator_module_type", "llama_index_llm")
    
    if generator_module_type == "llama_index_llm":
        generator_class = LlamaIndexLLM
        generator_params = {
            "llm": kwargs.pop("llm", "openai"),
            "model": kwargs.pop("model", "gpt-4o-mini"),
            "batch": kwargs.pop("batch", 8),
            "temperature": kwargs.pop("temperature", 0.7),
            "max_tokens": kwargs.pop("max_tokens", 256)
        }
    
    elif generator_module_type == "sap_api":
        generator_class = SAPAPI
        generator_params = {
            "llm": kwargs.pop("llm", "mistralai"),
            "model": kwargs.pop("model", "mistralai-large-instruct"),
            "api_url": kwargs.pop("api_url", None),
            "bearer_token": None,
            "batch": kwargs.pop("batch", 8),
            "temperature": kwargs.pop("temperature", 0.7),
            "max_tokens": kwargs.pop("max_tokens", 500)
        }
    
    elif generator_module_type == "openai":
        generator_class = OpenAILLM
        generator_params = {
            "llm": kwargs.pop("model", "gpt-4o-mini"),
            "batch": kwargs.pop("batch", 8)
        }
    
    elif generator_module_type == "vllm":
        generator_class = Vllm
        generator_params = {
            "llm": kwargs.pop("llm"),
            "model": kwargs.pop("model", None)
        }
        for key in ["tensor_parallel_size", "gpu_memory_utilization"]:
            if key in kwargs:
                generator_params[key] = kwargs.pop(key)
        
        if "gpu_memory_utilization" not in generator_params:
            generator_params["gpu_memory_utilization"] = 0.8

    else:
        raise ValueError(f"Unknown generator module type: {generator_module_type}")
    
    return generator_class, generator_params


class GeneratorWrapper:
    
    def __init__(self, generator_instance: BaseGenerator):
        self.generator = generator_instance
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        results, _, _ = self.generator._pure(prompts, **kwargs)
        return results
    
    def generate_from_dataframe(self, df: pd.DataFrame, prompt_column: str = 'prompts',
                              output_column: str = 'generated_texts', **kwargs) -> pd.DataFrame:
        if prompt_column not in df.columns:
            raise ValueError(f"Prompt column '{prompt_column}' not found in dataframe")
        
        result_df = self.generator.pure(df, **kwargs)
        return result_df
    
    def pure(self, previous_result: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self.generator.pure(previous_result, **kwargs)
    
    def cleanup(self):
        if hasattr(self.generator, 'cleanup'):
            self.generator.cleanup()


def create_generator(model: str, provider: Optional[str] = None, 
                    project_dir: str = "./", **kwargs) -> GeneratorWrapper:
    
    if not provider:
        provider = _detect_provider(model)
    
    if 'module_type' in kwargs:
        provider = kwargs.pop('module_type')
    if 'batch_size' in kwargs:
        kwargs['batch'] = kwargs.pop('batch_size')
    
    if provider in ['openai', 'openai_llm']:
        generator = OpenAILLM(
            project_dir=project_dir,
            llm=model,
            batch=kwargs.pop('batch', 8),
            **kwargs
        )
    
    elif provider == 'sap_api':
        generator = SAPAPI(
            project_dir=project_dir,
            model=model,
            llm=kwargs.pop('llm', 'mistralai'),
            api_url=kwargs.pop('api_url', None),
            bearer_token=None,
            batch=kwargs.pop('batch', 8),
            temperature=kwargs.pop('temperature', 0.7),
            max_tokens=kwargs.pop('max_tokens', 500),
            **kwargs
        )
    
    elif provider == 'vllm':
        vllm_kwargs = kwargs.copy()
        if 'gpu_memory_utilization' not in vllm_kwargs:
            vllm_kwargs['gpu_memory_utilization'] = 0.8
        
        generator = Vllm(
            project_dir=project_dir,
            llm=model,
            model=kwargs.pop('model', model),
            **vllm_kwargs
        )
    
    elif provider in ['llama_index', 'llama_index_llm']:
        generator = LlamaIndexLLM(
            project_dir=project_dir,
            llm=kwargs.pop('llm', 'openai'),
            model=model,
            batch=kwargs.pop('batch', 8),
            temperature=kwargs.pop('temperature', 0.7),
            max_tokens=kwargs.pop('max_tokens', 256),
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return GeneratorWrapper(generator)


def _detect_provider(model: str) -> str:
    
    model_lower = model.lower()
    
    if any(x in model_lower for x in ['gpt', 'o1-mini', 'o1-preview']):
        return 'openai'
    
    elif any(x in model_lower for x in ['mistralai-large-instruct', 'gpt-4o-mini', 'gpt-35-turbo', 'claude', 'anthropic', 'gemini']):
        return 'sap_api'
    
    elif any(x in model_lower for x in ['mistral', 'llama', 'qwen', 'mixtral']):
        return 'vllm'
    
    else:
        return 'llama_index'