from transformers import AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from config import train_config, lora_config
from prompt import UserProfileAgentPrompt

base_model_path = "/home/ccm/models/Qwen2.5-0.5B-Instruct"


def load_train_test_ds():
    data_files = {
        "train": "data/train.jsonl",
        "test": "data/test.jsonl"
    }

    split = load_dataset("json",
        data_files=data_files
    )

    return split["train"], split["test"]


def main():
    model = AutoModelForCausalLM.from_pretrained(base_model_path)

    # 1. load dataset
    train_ds, test_ds = load_train_test_ds()
    prompt_template = UserProfileAgentPrompt.load_prompt_template()

    # 2. Preprocess
    def preprocess(sample):
        output = sample["Output"] if sample["Output"] is not None else ''

        return {
            "prompt":[
                {"role": "system", "content": UserProfileAgentPrompt.system_prompt},
                {"role": "user", "content": prompt_template.format(data=sample["Input"])}
            ],
            "completion": [{"role": "assistant", "content": output}]
        }

    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    test_ds = test_ds.map(preprocess, remove_columns=test_ds.column_names)

    # 3. Set up the trainer
    sft_config = SFTConfig(
        **train_config,
        chat_template_path=base_model_path,
        do_train=True
    )

    sft_trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        args=sft_config,
       # peft_config=lora_config  # Have a bug
    ) 

    # 4. Train
    ret = sft_trainer.train()
    print(ret)



if __name__ == '__main__':
    main()
    

    
