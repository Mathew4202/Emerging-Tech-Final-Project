from datasets import load_dataset

dataset = load_dataset("json", data_files="training/sample_train.jsonl", split="train")

def format_example(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }

formatted_dataset = dataset.map(format_example)

print(formatted_dataset[0]["text"])