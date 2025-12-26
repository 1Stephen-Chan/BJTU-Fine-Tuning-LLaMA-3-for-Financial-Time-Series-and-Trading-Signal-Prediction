import torch
import os
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer

# --- 配置参数 ---
MODEL_ID = 'LLM-Research/Llama-3.2-1B-Instruct'
JSONL_FILE = 'stock_mask_T10.jsonl'
OUTPUT_DIR = "./llama_3_2_stock_ft_t10"

# --- 格式化函数 ---
def formatting_prompts_func(example):
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"你是一个专业的股票分析助手。<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{example['instruction']}\n数据如下:{example['input']}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
        f"{example['output']}<|eot_id|>"
    )

# --- 主训练函数 ---
def run_fine_tuning():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"准备进行 LoRA 微调...")

    # 加载数据集
    train_dataset = load_dataset("json", data_files=JSONL_FILE, split="train")

    # 模型加载 
    model_dir = snapshot_download(MODEL_ID, cache_dir='./')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    print("正在以 BF16 全精度加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16, 
        device_map="auto",         
        trust_remote_code=True
    )

    # LoRA 配置
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # 直接应用 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 训练参数 
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=4,  
        gradient_accumulation_steps=4,   
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        optim="adamw_torch",             
        bf16=True,                      
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )

    # 初始化 SFTTrainer 
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        formatting_func=formatting_prompts_func
    )

    # 开始训练
    print("正在启动训练...")
    trainer.train()

    # 保存
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"成功! 适配器已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_fine_tuning()