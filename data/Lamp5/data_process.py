import ijson, json

items = []
with open("train_questions.json", "r", encoding="utf-8") as f:
    parser = ijson.items(f, "item")  # 针对大数组
    for i, obj in enumerate(parser):
        if i >= 10:
            break
        items.append(obj)

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)