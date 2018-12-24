import os
from shutil import copyfile


train_folder = '/home/eli/Data/cats_vs_dogs/train'

dest_sub_train = '/home/eli/Data/cats_vs_dogs/sub_train'
dest_sub_eval = '/home/eli/Data/cats_vs_dogs/sub_eval'

mini1 = '/home/eli/Data/cats_vs_dogs/mini_train1'
mini2 = '/home/eli/Data/cats_vs_dogs/mini_train2'
mini3 = '/home/eli/Data/cats_vs_dogs/mini_train3'

super_mini1 = '/home/eli/Data/cats_vs_dogs/super_mini_train1'
super_mini2 = '/home/eli/Data/cats_vs_dogs/super_mini_train2'
super_mini3 = '/home/eli/Data/cats_vs_dogs/super_mini_train3'

mini4 = '/home/eli/Data/cats_vs_dogs/mini_eval'
super_mini4 = '/home/eli/Data/cats_vs_dogs/super_mini_eval'

paths = os.listdir(train_folder)

import random
random.shuffle(paths)

for p in paths[:1000]:
    if not os.path.exists(mini1):
        os.mkdir(mini1)
    copyfile(os.path.join(train_folder, p),os.path.join(mini1,p))

for p in paths[1000:2000]:
    if not os.path.exists(mini2):
        os.mkdir(mini2)
    copyfile(os.path.join(train_folder, p),os.path.join(mini2,p))

for p in paths[2000:3000]:
    if not os.path.exists(mini3):
        os.mkdir(mini3)
    copyfile(os.path.join(train_folder, p),os.path.join(mini3,p))

for p in paths[3000:3100]:
    if not os.path.exists(super_mini1):
        os.mkdir(super_mini1)
    copyfile(os.path.join(train_folder, p),os.path.join(super_mini1,p))

for p in paths[3100:3200]:
    if not os.path.exists(super_mini2):
        os.mkdir(super_mini2)
    copyfile(os.path.join(train_folder, p),os.path.join(super_mini2,p))

for p in paths[3200:3300]:
    if not os.path.exists(super_mini3):
        os.mkdir(super_mini3)
    copyfile(os.path.join(train_folder, p), os.path.join(super_mini3, p))


for p in paths[-1000:]:
    if not os.path.exists(mini4):
        os.mkdir(mini4)
    copyfile(os.path.join(train_folder, p),os.path.join(mini4,p))


for p in paths[-1100:-1000]:
    if not os.path.exists(super_mini4):
        os.mkdir(super_mini4)
    copyfile(os.path.join(train_folder, p), os.path.join(super_mini4, p))
