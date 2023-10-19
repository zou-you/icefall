from pathlib import Path
from collections import defaultdict


path = '/home/zouyou/workspaces/ASR/newKaldi/icefall/egs/kespeech/ASR/KeSpeech/KeSpeech/Tasks/ASR/test/text'
p = Path(path)

audio = defaultdict()

with open(path, 'r') as fr:
    res = fr.readlines()


{item.strip().split()  for item in open(path, 'r')}

print(1)