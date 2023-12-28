'''
This script is used to filter the corrupt Image Net images to only the selected classes
'''


import numpy as np
import os
from matplotlib.pyplot import imread
import platform
import shutil
import pandas as pd
import json



'''
#extracting list of folder names in subset ImageNet20
folder_path = './datasets/subImageNet'
subset_wid = []
wid_list = os.listdir(os.path.join(folder_path,'train'))
for _,wid in enumerate(wid_list):
    subset_wid.append(wid)
print (subset_wid)

#deleting files that are not needed

corrupt_path = './datasets/Backup'
corrupt_type = os.listdir(corrupt_path)
for _,ctype in enumerate(corrupt_type):
    severity_scale = os.listdir(os.path.join(corrupt_path,ctype))
    for _,slevel in enumerate(severity_scale):
        print (ctype)
        print (slevel)
        wid_list = os.listdir(os.path.join(corrupt_path,ctype,slevel))
        for _,wid in enumerate(wid_list):
            if wid not in subset_wid:
                shutil.rmtree(os.path.join(corrupt_path,ctype,slevel,wid))
'''




json_file_path = './datasets/word_dict.txt'

with open(json_file_path, 'r') as j:
     wnid_dict = json.loads(j.read())



wnids=[]
words={}
labels={}

for key,value in wnid_dict.items():
    wnids.append(value[0])
    words[value[0]]=value[1]
    
wnids = pd.DataFrame(wnids)

words = pd.DataFrame(words.items())
    
wnids.to_csv('./datasets/wnids.txt',index=False, header = False)
words.to_csv('./datasets/words.txt',index=False, header = False, sep = '\t')



