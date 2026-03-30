import json

with open("training/sample_train.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        item = json.loads(line)
        print("Question:", item["instruction"])
        print("Answer:", item["output"])
        print("---")