import split_folders
from configurations import *
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
print(ENHANCED_DIR)
print(SPLIT_DIR)
split_folders.ratio(ENHANCED_DIR, output=SPLIT_DIR, seed=1, ratio=(.8, .2)) # default values