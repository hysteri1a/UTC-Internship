import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import pandas as pd
import torch


def setup_argparse():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Script for Causal Language Models")

    # 必需参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the trained model and checkpoints")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the training data JSON file")

    # 模型和数据参数
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length (input + output). Default: 1024")
    parser.add_argument("--use_fast_tokenizer", action="store_true",
                        help="Whether to use fast tokenizer implementation")

    # LoRA参数
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA attention dimension. Default: 8")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling factor. Default: 32")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout probability. Default: 0.05")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj",
                        help="Comma-separated list of modules to apply LoRA to. Default: q_proj,v_proj")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per device. Default: 4")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs. Default: 3")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Initial learning rate. Default: 2e-5")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of gradient accumulation steps. Default: 2")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable mixed precision training")
    parser.add_argument("--logging_dir", type=str, default="./logs",
                        help="Directory for storing training logs. Default: ./logs")

    # 量化参数（修改为8-bit配置）
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load model in 8-bit quantization mode")
    parser.add_argument("--compute_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Compute dtype for quantization. Default: bfloat16")

    return parser.parse_args()


def process_data(example, tokenizer, max_seq_length):
    conv = example["conversation"]
    instruction_text = conv.get("instruction", "")
    human_text = conv["input"]
    assistant_text = conv["output"]

    input_text = f"{tokenizer.bos_token}{instruction_text}\n\nUser:{human_text}\n\nAssistant:"

    # 编码输入部分
    input_tokens = tokenizer(
        input_text,
        truncation=True,
        max_length=max_seq_length // 2,  # 预留空间给输出
        add_special_tokens=False
    )

    # 编码输出部分
    output_tokens = tokenizer(
        assistant_text,
        truncation=True,
        max_length=max_seq_length - len(input_tokens["input_ids"]),
        add_special_tokens=False
    )

    # 合并并添加EOS
    input_ids = input_tokens["input_ids"] + output_tokens["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = input_tokens["attention_mask"] + output_tokens["attention_mask"] + [1]
    labels = [-100] * len(input_tokens["input_ids"]) + output_tokens["input_ids"] + [tokenizer.eos_token_id]

    # 确保不超长
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        labels = labels[:max_seq_length]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main(args):
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 配置量化参数（8-bit配置）
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=args.load_in_8bit,
        llm_int8_threshold=6.0,
        bnb_4bit_compute_dtype=getattr(torch, args.compute_dtype),
    ) if args.load_in_8bit else None

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=args.load_in_8bit,
        device_map="auto",
        quantization_config=quantization_config
    )

    # 配置LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载和预处理数据集
    data = pd.read_json(args.dataset_path)
    train_ds = Dataset.from_pandas(data)
    train_dataset = train_ds.map(
        process_data,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": args.max_seq_length
        },
        remove_columns=train_ds.column_names
    )

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_dir=args.logging_dir,
        save_strategy="epoch",
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100
        ),
    )

    # 开始训练
    trainer.train()

    # 保存最终模型
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    args = setup_argparse()
    main(args)