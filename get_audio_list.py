import os

with open("train.lst", "w+") as file:
    path = "data"	
    for (dirname, subdir, subfile) in os.walk(path):
        for f in subfile:
            file.writelines(os.path.join(dirname[5:], f) + "\n")
