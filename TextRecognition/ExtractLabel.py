import os
import glob

root_dir = "F:\img\synth\out"

labels = ''
with open(os.path.join(root_dir, "labels.txt"), "r", encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        name, label = line.split(' ')
        labels += label


print(len(labels))

print(sorted(set(labels)))
