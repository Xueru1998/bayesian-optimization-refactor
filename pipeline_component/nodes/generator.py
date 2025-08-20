import os
import pandas as pd
import numpy as np
import tempfile
from typing import List, Dict, Any, Union, Optional
from abc import ABC, abstractmethod
import logging
import gc
import torch

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
                if hasattr(self.vllm_model, 'llm_engine'):
                    if hasattr(self.vllm_model.llm_engine, 'model_executor'):
                        self.vllm_model.llm_engine.model_executor.shutdown()
                
                del self.vllm_model
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
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


class VllmAPI(BaseGenerator):
    
    def __init__(self, project_dir: str, llm: str, uri: str, batch: int = 8, max_tokens: int = 400, **kwargs):
        super().__init__(project_dir, **kwargs)
        self.llm = llm
        self.uri = uri
        self.batch = batch
        self.max_tokens = max_tokens
        
        import requests
        self.requests = requests
    
    def _pure(self, prompts: List[str], **kwargs) -> tuple:
        results = []
        usages = []
        
        params = {
            "model": self.llm,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", 1.0),
        }
        
        for i in range(0, len(prompts), self.batch):
            batch_prompts = prompts[i:i + self.batch]
            
            for prompt in batch_prompts:
                try:
                    response = self.requests.post(
                        f"{self.uri}/v1/completions",
                        json={
                            "prompt": prompt,
                            **params
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    if "choices" in data and len(data["choices"]) > 0:
                        results.append(data["choices"][0]["text"])
                    else:
                        results.append("")
                    
                    if "usage" in data:
                        usages.append(data["usage"])
                    else:
                        usages.append({})
                        
                except Exception as e:
                    logger.error(f"VLLM API error: {e}")
                    results.append("")
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


class GeneratorModule:
    
    def __init__(self, model_name="openai", model_type="gpt-4o-mini", batch_size=8, **kwargs):
        self.model_name = model_name
        self.model_type = model_type
        self.batch_size = batch_size
        self.generator = None
        self.temp_dir = tempfile.mkdtemp(prefix="autorag_gen_")
        self.extra_kwargs = kwargs  
        self.runtime_temperature = None
        
    def __del__(self):
        self.cleanup()
    
    def cleanup(self):
        if self.generator:
            if self.model_name == "vllm" and hasattr(self.generator, 'cleanup'):
                try:
                    self.generator.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up vLLM: {e}")
            
            del self.generator
            self.generator = None
            
        if hasattr(self, 'temp_dir') and self.temp_dir:
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass

    def _initialize_generator(self):
        if self.generator:
            return self.generator
            
        if self.model_name == "openai":
            self.generator = OpenAILLM(
                project_dir=self.temp_dir,
                llm=self.model_type,
                batch=self.batch_size
            )
        
        elif self.model_name == "vllm":
            vllm_kwargs = {k: v for k, v in self.extra_kwargs.items() if k != 'batch'}
            
            if 'gpu_memory_utilization' not in vllm_kwargs:
                vllm_kwargs['gpu_memory_utilization'] = 0.8
            
            self.generator = Vllm(
                project_dir=self.temp_dir,
                llm=self.model_type,
                model=self.model_type,
                **vllm_kwargs
            )
        
        elif self.model_name == "vllm_api":
            if 'uri' not in self.extra_kwargs:
                raise ValueError("URI is required for vllm_api generator")
            
            self.generator = VllmAPI(
                project_dir=self.temp_dir,
                llm=self.model_type,
                uri=self.extra_kwargs['uri'],
                batch=self.batch_size,
                max_tokens=int(self.extra_kwargs.get('max_tokens', 400))
            )
        
        elif self.model_name == "llama_index":
            llm_provider = self.extra_kwargs.get('llm', 'openai')

            if isinstance(llm_provider, list):
                llm_provider = llm_provider[0] if llm_provider else 'openai'

            temperature = self.runtime_temperature if self.runtime_temperature is not None else self.extra_kwargs.get('temperature', 0.7)
            self.generator = LlamaIndexLLM(
                project_dir=self.temp_dir,
                llm=llm_provider,
                model=self.model_type,
                batch=self.batch_size,
                temperature=temperature,
                max_tokens=self.extra_kwargs.get('max_tokens', 256)
            )
        
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
        return self.generator
    
    
    def generate(self, prompts: List[str], max_tokens: int = None, temperature: float = 0.7) -> List[str]:
        generator = self._initialize_generator()
        
        generation_params = {}

        if self.model_name == "llama_index":
            try:
                generated_texts, _, _ = generator._pure(prompts)
                return generated_texts
            except Exception as e:
                print(f"Error in text generation: {e}")
                import traceback
                traceback.print_exc()
                return ["Error generating text."] * len(prompts)

        if temperature is not None:
            generation_params["temperature"] = float(temperature)
        
        if max_tokens is not None:
            generation_params["max_tokens"] = int(max_tokens)
        elif self.model_name == "vllm":
            generation_params["max_tokens"] = 256
        elif hasattr(self, 'extra_kwargs') and 'max_tokens' in self.extra_kwargs:
            try:
                generation_params["max_tokens"] = int(self.extra_kwargs['max_tokens'])
            except (ValueError, TypeError):
                generation_params["max_tokens"] = 256
            
        try:
            generated_texts, _, _ = generator._pure(prompts, **generation_params)
            return generated_texts
        except Exception as e:
            print(f"Error in text generation: {e}")
            import traceback
            traceback.print_exc()
            return ["Error generating text."] * len(prompts)
    
    def generate_from_dataframe(self, df: pd.DataFrame, prompt_column: str = 'prompts', 
                            output_column: str = 'generated_texts',
                            **generation_params) -> pd.DataFrame:
        if prompt_column not in df.columns:
            raise ValueError(f"Prompt column '{prompt_column}' not found in dataframe")
        
        prompts = df[prompt_column].tolist()

        if 'temperature' in generation_params and self.model_name == "llama_index":
            self.runtime_temperature = generation_params.get('temperature')
            self.generator = None
        
        print(f"Generating responses for {len(prompts)} prompts using {self.model_name}:{self.model_type}...")
        generated_texts = self.generate(prompts, **generation_params)
        
        result_df = df.copy()
        result_df[output_column] = generated_texts
        
        return result_df
    
    def pure(self, previous_result: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self.generator.pure(previous_result, **kwargs)
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        generator = self._initialize_generator()
        
        if not hasattr(generator, 'astream'):
            raise NotImplementedError(f"Asynchronous generation not supported for {self.model_name} model")
        
        full_text = ""
        async for chunk in generator.astream(prompt, **kwargs):
            full_text = chunk
            
        return full_text
    
    def stream(self, prompt: str, callback=None, **kwargs):
        generator = self._initialize_generator()
        
        import asyncio
        
        async def _stream():
            full_text = ""
            async for chunk in generator.astream(prompt, **kwargs):
                if callback:
                    callback(chunk)
                full_text = chunk
            return full_text
            
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_stream())
    
    def structured_output(self, prompts: List[str], output_class):
        generator = self._initialize_generator()
        
        return generator.structured_output(prompts, output_class)


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
    
    elif generator_module_type == "vllm_api":
        generator_class = VllmAPI
        generator_params = {
            "llm": kwargs.pop("llm", "default"),
            "uri": kwargs.pop("uri"),
            "batch": kwargs.pop("batch", 8),
            "max_tokens": kwargs.pop("max_tokens", 400)
        }

    else:
        raise ValueError(f"Unknown generator module type: {generator_module_type}")
    
    return generator_class, generator_params


def create_model_provider_map():
    return {
        "gpt-4o": "openai",
        "gpt-4o-mini": "openai",
        "gpt-4": "openai",
        "gpt-3.5-turbo": "openai",
        "gpt-3.5-turbo-16k": "openai",
        "gpt-4-turbo": "openai",
        
        "mistralai/Mistral-7B-Instruct-v0.2": "vllm",
        "mistralai/Mistral-7B-v0.1": "vllm",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "vllm",
        "meta-llama/Llama-2-7b-hf": "vllm",
        "meta-llama/Llama-2-13b-hf": "vllm",
        "meta-llama/Llama-2-70b-hf": "vllm",
        "meta-llama/Llama-2-7b-chat-hf": "vllm",
        "meta-llama/Llama-2-13b-chat-hf": "vllm",
        "meta-llama/Llama-2-70b-chat-hf": "vllm",
        "Qwen/Qwen-7B": "vllm",
        "Qwen/Qwen-7B-Chat": "vllm",
        "Qwen/Qwen-14B": "vllm",
        "Qwen/Qwen-14B-Chat": "vllm",
        "TheBloke/Mistral-7B-Instruct-v0.2-AWQ": "vllm",
        "TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ": "vllm",
        "TheBloke/Llama-2-7B-AWQ": "vllm",
        "TheBloke/Llama-2-13B-AWQ": "vllm",
        "TheBloke/CodeLlama-7B-AWQ": "vllm",
        "TheBloke/CodeLlama-13B-AWQ": "vllm",
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ": "vllm",
        "TheBloke/Llama-2-7B-AWQ": "vllm",
    }


def get_provider_for_model(model_name: str) -> str:
    model_provider_map = create_model_provider_map()
    
    if model_name in model_provider_map:
        return model_provider_map[model_name]

    for prefix, provider in model_provider_map.items():
        if model_name.startswith(prefix):
            return provider

    print(f"Warning: Unknown model '{model_name}', defaulting to llama_index provider")
    return "llama_index"


def create_generator(model: str = None, module_type: str = None, uri: str = None, 
                    batch_size: int = 8, max_tokens: int = None, **kwargs) -> GeneratorModule:

    if max_tokens is not None:
        try:
            kwargs['max_tokens'] = int(max_tokens)
        except (ValueError, TypeError):
            kwargs['max_tokens'] = 256

    if module_type == "vllm_api":
        if not uri:
            raise ValueError("URI is required for vllm_api generator")
        return GeneratorModule(
            model_name="vllm_api", 
            model_type=model,
            batch_size=batch_size,
            uri=uri,
            **kwargs
        )
    elif module_type == "vllm":
        vllm_kwargs = {k: v for k, v in kwargs.items() if k != 'batch'}
        
        if 'gpu_memory_utilization' not in vllm_kwargs:
            vllm_kwargs['gpu_memory_utilization'] = 0.8
        
        return GeneratorModule(
            model_name="vllm",
            model_type=model,
            batch_size=batch_size,
            **vllm_kwargs
        )
    elif module_type == "openai":
        return GeneratorModule(
            model_name="openai",
            model_type=model,
            batch_size=batch_size,
            **kwargs
        )
    elif module_type == "llama_index":
        if 'llm' not in kwargs:
            kwargs['llm'] = 'openai'
        return GeneratorModule(
            model_name="llama_index",
            model_type=model,
            batch_size=batch_size,
            **kwargs
        )

    if model:
        provider = get_provider_for_model(model)
        if provider == "vllm":
            vllm_kwargs = {k: v for k, v in kwargs.items() if k != 'batch'}
            
            if 'gpu_memory_utilization' not in vllm_kwargs:
                vllm_kwargs['gpu_memory_utilization'] = 0.8
            
            return GeneratorModule(
                model_name=provider,
                model_type=model,
                batch_size=batch_size,
                **vllm_kwargs
            )
        else:
            return GeneratorModule(
                model_name=provider,
                model_type=model,
                batch_size=batch_size,
                **kwargs
            )

    return GeneratorModule(batch_size=batch_size, **kwargs)