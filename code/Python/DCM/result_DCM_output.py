import os,glob
from time import sleep

# Select all 'graphml' files in current directory
myl=sorted(glob.glob('*graphml'))
print (myl)

#Range define the required no. of subsets of nodes
for j in (myl):
    for i in range(4,9,1):
        os.system("python.exe DCM-SubsetNodes.py %s %d > %s-%d-DCM.txt"
                  %(j,i, j.replace(".graphml", ""),i))

