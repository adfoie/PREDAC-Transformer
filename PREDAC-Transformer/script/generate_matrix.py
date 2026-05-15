import multiprocessing
import os
import numpy as np
import pandas as pd
import argparse
import random

import torch
import esm



parser = argparse.ArgumentParser()

parser.add_argument('--aaindex_file', required=True, metavar='file')
parser.add_argument('--seq_dir', required=True)
parser.add_argument('--type', required=True,metavar='predict or training or testing')
parser.add_argument('--dir', required=True, metavar='file')
parser.add_argument('--thread', required=True)
parser.add_argument('--esm_dir', required=True)
args = parser.parse_args()


def extract_esm(filename,subtype):
    EMB_PATH = subtype+'_esm_seq' # Path to directory of embeddings for P62593.fasta
    EMB_LAYER = 6
    fn = EMB_PATH+'/'+filename+'.pt'
    embs = torch.load(fn)
    Xs=(embs['representations'][EMB_LAYER])
    Xs = Xs.numpy()
    return Xs
class str_to_num():
    def __init__(self):
        self.seq1 = []
        self.seq2 = []
        self.name1 = []
        self.name2 = []

        #version2
        self.all_data = []

    def read_from_csv(self,csv_data,type='predict'):
        print("starting read data....")
        self.seq1 = csv_data['seq_1']
        self.seq2 = csv_data['seq_2']
        self.name1 = csv_data['new_name_1']
        self.name2 = csv_data['new_name_2']
        if type=='predict':
            pass
        else:
            self.label = csv_data['label']
            assert(len(self.seq1) == len(self.label))
        assert(len(self.seq1) == len(self.seq2))
        print("read data done")



    def generate_seq_test(self,seq1, seq2 ,name1,name2,matchfile,subtype):
        #this is a two-dimensional array
        info=pd.read_csv(subtype+'_HA1_forhi.csv')
        lista=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
        listb=['D','N']
        listz=['E','Q']
        listj=['L','I']

        strains=[]
        for i in range(len(seq1)):
            strain1=[]
            strain2=[]
            
            for j in range(len(seq1[i])):
                seq1_aa=seq1[i][j]
                seq2_aa=seq2[i][j]
                if seq1[i][j]=='X':
                    seq1_aa=random.choice(lista)
                elif seq1[i][j]=='B':
                    seq1_aa=random.choice(listb)
                elif seq1[i][j]=='Z':
                    seq1_aa=random.choice(listz)
                elif seq1[i][j]=='J':
                    seq1_aa=random.choice(listj)
                else:
                    pass
                if seq2[i][j]=='X':
                    seq2_aa=random.choice(lista)
                elif seq2[i][j]=='B':
                    seq2_aa=random.choice(listb)
                elif seq2[i][j]=='Z':
                    seq2_aa=random.choice(listz)
                elif seq2[i][j]=='J':
                    seq2_aa=random.choice(listj)
                else:
                    pass

                strain1.append(matchfile[seq1_aa].tolist())
                strain2.append(matchfile[seq2_aa].tolist())
                
            ar1=extract_esm(info['seqid'][info['Isolate_Id']==name1[i]].iloc[0],subtype)
            ar2=extract_esm(info['seqid'][info['Isolate_Id']==name2[i]].iloc[0],subtype)
            strain=np.concatenate((ar1, strain1, ar2, strain2),axis=1)
            strains.append(strain)
        mychar=np.array(strains).astype(np.float64)
        return mychar

    def generate_seq(self,seq1, seq2 ,label,name1,name2,matchfile,subtype):
        print('begin')
        #this is a two-dimensional array
        info=pd.read_csv(subtype+'_HA1_forhi.csv')
        lista=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
        listb=['D','N']
        listz=['E','Q']
        listj=['L','I']

        strains=[]
        for i in range(len(seq1)):
            strain1=[]
            strain2=[]
            labels=[]
            for j in range(len(seq1[i])):
                seq1_aa=seq1[i][j]
                seq2_aa=seq2[i][j]
                if seq1[i][j]=='X':
                    seq1_aa=random.choice(lista)
                elif seq1[i][j]=='B':
                    seq1_aa=random.choice(listb)
                elif seq1[i][j]=='Z':
                    seq1_aa=random.choice(listz)
                elif seq1[i][j]=='J':
                    seq1_aa=random.choice(listj)
                else:
                    pass
                if seq2[i][j]=='X':
                    seq2_aa=random.choice(lista)
                elif seq2[i][j]=='B':
                    seq2_aa=random.choice(listb)
                elif seq2[i][j]=='Z':
                    seq2_aa=random.choice(listz)
                elif seq2[i][j]=='J':
                    seq2_aa=random.choice(listj)
                else:
                    pass

                strain1.append(matchfile[seq1_aa].tolist())
                strain2.append(matchfile[seq2_aa].tolist())
                labels.append([label[i]])
            ar1=extract_esm(info['seqid'][info['Isolate_Id']==name1[i]].iloc[0],subtype)
            print(info['seqid'][info['Isolate_Id']==name1[i]].iloc[0])
            print(ar1.shape)
            ar2=extract_esm(info['seqid'][info['Isolate_Id']==name2[i]].iloc[0],subtype)
            print(info['seqid'][info['Isolate_Id']==name2[i]].iloc[0])
            print(ar2.shape)
            strain=np.concatenate((ar1, strain1, ar2, strain2,labels),axis=1)
            strains.append(strain)
        mychar=np.array(strains).astype(np.float64)
        return mychar

    def do_generate(self,type,matchfile,subtype):
        print('start generating seq....')
        if type=='predict':
            self.all_data=self.generate_seq_test(self.seq1, self.seq2,self.name1,self.name2,matchfile,subtype)
        else:
            self.all_data=self.generate_seq(self.seq1, self.seq2,self.label,self.name1,self.name2,matchfile,subtype)
        print('generate seq done')

    def save_to_npy(self,dir,filename):
        arr = np.array(self.all_data)
        np.save(dir+'/'+filename+'.npy', arr)


##################################################


def matrix_generate(record):
    seq_file,aj_dic,dir,type,filename,matchfile,subtype=record
    seq_all_0=pd.read_csv(seq_file,sep='\t',index_col=0)
    seq_all=seq_all_0.reset_index(drop=True)
    s = str_to_num()
    s.read_from_csv(seq_all,type)
    s.do_generate(type,matchfile,subtype)
    s.save_to_npy(dir,filename)
    print('done')








if __name__=='__main__':
    import time
    start_time=time.time()
    aaindex=pd.read_csv(args.aaindex_file,sep='\t',index_col=0)

    #aaindex=pd.read_csv('/mnt/alamo01/users/liujz/cnntest/PREDAV-CNN-main/aaindex_feature_H1N1.txt',sep='\t',index_col=0)
    #aaindex=aaindex[['number']]

    matchfile=aaindex.T
    dir=args.dir
    seq_dir=args.seq_dir
    all_dict=aaindex.T.to_dict()
    aj_dic=all_dict
    type=args.type
    subtype=args.esm_dir
    thread=int(args.thread)
    seq_file_list=os.listdir(seq_dir)
    print(seq_file_list)
    for seqfile in seq_file_list:
        filelist=[seqfile]
        pool=multiprocessing.Pool(processes=thread)
        for i in filelist:#seq_file_list:
            map_args=(seq_dir+'/'+i,aj_dic,dir,type,i,matchfile,subtype)
            pool.apply_async(matrix_generate,(map_args,))
        pool.close()
        pool.join()

    print("time is %s"%(time.time()-start_time))
##################################################

