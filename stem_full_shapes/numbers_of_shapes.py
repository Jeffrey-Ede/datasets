import numpy as np

SAVE_FILE = "//ads.warwick.ac.uk/shared/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/visualize_data/misc/shapes.npy"

shapes = np.load(SAVE_FILE)

dict = {}
for s in shapes:
    string = str(s)
    if string in dict:
        dict[string] += 1
    else:
        dict[string] = 1

#for i, s in enumerate(dict):
#    if dict[s] == 1:
#        print(i)
#        quit()

sorted = sorted([s for s in dict], key = lambda s: dict[s] + np.product(np.fromstring(s.strip('[]'), dtype=int, sep=' '), axis=0, dtype=np.float32)/(4008*2672))

for s in reversed(sorted):
    print(np.array(s), dict[s])

#for s in dict:
#    print(s, dict[s])

