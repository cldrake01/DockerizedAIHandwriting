from trdg.generators import GeneratorFromRandom
from tqdm import *
import pickle as p
import os

datas = GeneratorFromRandom(
    length=10,
    count=6000,
    random_blur=True,
    random_skew=True,
    is_handwritten=True
)

if os.path.exists("saves/dataset.p"):
    with open("saves/dataset.p", 'rb') as file:
        (targets, images) = p.load(file)

else:
    targets = []
    images = []
pbar = tqdm(datas)
for target, image in pbar:
    targets.append(target)
    images.append(image)

print("Saving data")
print(len(targets))
print(len(targets))
file = open("saves/dataset.p", "wb")
p.dump((targets, images), file)
