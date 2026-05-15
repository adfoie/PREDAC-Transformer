import Levenshtein
import sys
import pandas as pd
import os
import argparse
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument('--csv', required=False, metavar='file')
parser.add_argument('--seq', required=False, metavar='file')
parser.add_argument('--outdir', required=True, metavar='outdir')
parser.add_argument('--datatype', required=True,metavar='datatype')#H1N1_HA1
args = parser.parse_args()

def read_fasta(dir):
    seqdict={}
    idlist=[]
    seq=[]
    key=0
    with open(dir, 'r') as infile:
        for line in infile:
            if line[0]=='>':
                if key!=0:
                    seq.append(seqdict[key])
                    key=line[1:].strip()
                    idlist.append(key)
                    seqdict[key]=''
                else:
                    key=line[1:].strip()
                    idlist.append(key)
                    seqdict[key]=''
            elif len(line.strip())>0:
                seqdict[key]+=line.strip()
        seq.append(seqdict[key])
        #print(idlist)
        #print(len(seq))
        dfseq={'Isolate_Id':idlist,'seq':seq}
        dfseq=pd.DataFrame(dfseq)    
    return dfseq


dfseq=read_fasta(args.seq)

def replace_bases(seq):
    # 将序列转换为列表，以便修改
    listx=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    listb=['D','N']
    listz=['E','Q']
    listj=['L','I']
    seq_list = list(seq)
    # 遍历序列中的每个字符
    for i in range(len(seq_list)):
        if seq_list[i] == 'X': 
            seq_list[i] = str(np.random.choice(listx))
        elif seq_list[i] == 'B': 
            seq_list[i] = str(np.random.choice(listb))
        elif seq_list[i] == 'Z':  
            seq_list[i] = str(np.random.choice(listz))
        elif seq_list[i] == 'J':  
            seq_list[i] = str(np.random.choice(listj))
    # 将列表转换回字符串
    return ''.join(seq_list)




dfseq.columns=['Isolate_Id','seq']
dfseq=dfseq.drop_duplicates('Isolate_Id').reset_index(drop=True)
dfseq['seqid']=''
dfseq['seqid']=dfseq['seqid'].index.tolist()
dfseq['seqid']='seq'+dfseq['seqid'].astype(str)
dfseq.to_csv(args.outdir+'/'+args.datatype+'_forhi.csv',index=False)
#getting clear seq
dfseq['seq'] = dfseq['seq'].apply(replace_bases)
with open(args.outdir+'/'+args.datatype+'_forhi.fasta', 'w') as f:
    for i in range(len(dfseq)):
        f.write(">" +dfseq['seqid'].iloc[i] + "\n" + dfseq['seq'].iloc[i] + "\n")  

