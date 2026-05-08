import json

with open("sqa_sample.json", "r") as f, open("sqa_id.txt", "w") as ff:
    for line in f:
        data = json.loads(line)
        ff.write(data["id"] + "\n")

with open("openbook_sample.json", "r") as f, open("openbook_id.txt", "w") as ff:
    for line in f:
        data = json.loads(line)
        ff.write(data["id"] + "\n")
