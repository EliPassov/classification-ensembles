import os
from shutil import copyfile


train_folder = '/home/eli/Data/cats_vs_dogs/train'

dest_sub_train = '/home/eli/Data/cats_vs_dogs/sub_train'
dest_sub_eval = '/home/eli/Data/cats_vs_dogs/sub_eval'

mini1 = '/home/eli/Data/cats_vs_dogs/mini_train1'
mini2 = '/home/eli/Data/cats_vs_dogs/mini_train2'
mini3 = '/home/eli/Data/cats_vs_dogs/mini_train3'
mini4 = '/home/eli/Data/cats_vs_dogs/mini_eval'

paths = os.listdir(train_folder)

import random
random.shuffle(paths)

for p in paths[:1000]:
    if not os.path.exists(mini1):
        os.mkdir(mini1)
    copyfile(os.path.join(train_folder, p),os.path.join(mini1,p))

for p in paths[:1000]:
    if not os.path.exists(mini2):
        os.mkdir(mini2)
    copyfile(os.path.join(train_folder, p),os.path.join(mini2,p))

for p in paths[:1000]:
    if not os.path.exists(mini3):
        os.mkdir(mini3)
    copyfile(os.path.join(train_folder, p),os.path.join(mini3,p))

for p in paths[-1000:]:
    if not os.path.exists(mini4):
        os.mkdir(mini4)
    copyfile(os.path.join(train_folder, p),os.path.join(mini4,p))
