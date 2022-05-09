import os,glob
from time import sleep
myl=sorted(glob.glob('*graphml'))
print (myl)
for j in (myl):
    for i in range(2,4,1):
        os.system("python.exe Safa-GenGraph-Default.py %s %d > %s-%d-default.txt"%(j,i,j.replace(".graphml", ""),i))
