assert __name__ == "__main__"

# Performs no training, simply builds a list of sequences.

import sys
import os

folder_in = sys.argv[1]

train_dt = []
for (dirpath, dirnames, filenames) in os.walk(folder_in):
    seq = {}
    cometID = os.path.relpath(dirpath, folder_in)
    seq["ID"] = cometID
    images = []
    for filename in filenames:
        ext = os.path.splitext(filename)[1]
        if ext=='.fts':
            images.append(filename)
    images.sort()
    seq["images"] = images
    if len(images)>0:
        train_dt.append(seq)

print(train_dt)

