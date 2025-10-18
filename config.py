from langchain_openai import ChatOpenAI
import os
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    task_type="CAUSAL_LM"
)

train_config = {
    # train_process_config
    "seed": 42,
    "num_train_epochs": 10,
    # generate_config
    "max_length": 812,
    # output_config
    "output_dir": "./checkpoints",
    "overwrite_output_dir": True,
    "save_strategy": "epoch",  # Save model checkpoint at each epoch
    # log_config
    "report_to": ['tensorboard'],
    "logging_dir": "logs",
    "logging_strategy": "epoch",  # Log model state at each epoch
    # evaluation_config
    "eval_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "loss",  # Default loss type: CrossEntropy
    "greater_is_better": False,
    # dataloader_config
    "dataloader_num_workers": 4,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "gradient_checkpointing": True,
    "gradient_accumulation_steps": 16,
   # "fp16": True,
    # optimizer_config (Default optimizer: AdamW)
    "learning_rate": 1e-4,
    "weight_decay": 0.1,
    # learning_rate scheduler 
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05  
}


llm_api_config = {
    "openai":{
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "base_url": os.environ.get("OPENAI_BASE_URL")
    },
    "qwen": {
        "api_key": os.environ.get("QWEN_API_KEY"),
        "base_url": os.environ.get("QWEN_BASE_URL")
    },
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY"),
        "base_url": os.environ.get("DEEPSEEK_BASE_URL")
    },
    '': {
        "api_key": "Empty",
        "base_url": os.environ.get("VLLM_BASE_URL")
    }
}


def get_llm_client(model,provider=''):
    api_config = llm_api_config[provider.lower()]
    if api_config["api_key"] is None:
        api_config["api_key"] = os.environ["DEFAULT_API_KEY"]
    if api_config["base_url"] is None:
        api_config["base_url"] = os.environ["DEFAULT_BASE_URL"]
    
    return ChatOpenAI(
        model=os.path.join(provider, model),
        temperature=0,
        **api_config
    )
