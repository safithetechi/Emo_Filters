import scipy.spatial.distance as dis
import cv2
import csv
import numpy
from numpy import genfromtxt
from shutil import copyfile
import os






emo6_path = 'Emotion6/images';

inDir = 'input_Dir/';
outDir = 'output_Dir/';



def ReadImageName():
    with open('input_Dir/emo6_names.txt') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def mean_bc(a,b):
    prod = a*b
    sqt = prod**(1.0/2)
    coeff = sum(sqt)
    return coeff.mean()



def compute(ImageName,des_pdist,target_hlfeat):

    src_hlfeat = genfromtxt('input_Dir/'+ImageName+'output.csv', delimiter=',')

    targetImgNames=ReadImageName()

    target_gtPdist = genfromtxt('input_Dir/' +'emo6_prob.csv', delimiter=',')


    ntop = 10;
    accumu_dist = [];

    if(not os.path.isdir(outDir)):
        os.mkdir(outDir)

    bhattaC =numpy.ones(len(target_hlfeat))

    for i in range(0,len(target_hlfeat)):
        bhattaC[i]=mean_bc(target_gtPdist[i,:], des_pdist)

    mean1=(bhattaC.mean())
    mean2=(bhattaC.mean())/2

    SumOfMean=mean1+mean2

    des_ind=(bhattaC > SumOfMean).nonzero()


    indexes=des_ind[0].tolist()





    if(len(des_ind[0])>0):
        des_hlfeat = target_hlfeat[des_ind[0],:]
        des_images=[]
        for i in indexes:
            des_images.append(targetImgNames[i])
            if i==9:break

        selec_pDist = target_gtPdist[des_ind[0],:]



        src_hlfeat=numpy.array(src_hlfeat)
        des_hlfeat=numpy.array(des_hlfeat)

        src_hlfeat = src_hlfeat.reshape(( 1,src_hlfeat.shape[0]))



        if(len(des_images) >= ntop):
            V = dis.cdist(des_hlfeat, src_hlfeat,'euclidean','Smallest',ntop)
        else:
            V = dis.cdist(des_hlfeat, src_hlfeat,'euclidean','Smallest',len(des_images))


        Selected_Images=[]

        if(len(V)>0):
            for i in range(0,len(V)):
                desPath = outDir+'target_'+str(i)+'.jpg'
                print(desPath)
                Selected_Images.append(desPath)
                copyfile(emo6_path+'/'+des_images[i],desPath)
                if i==9:
                    break



        for i in range(0,len(des_ind[0])):
            accumu_dist.append(mean_bc(selec_pDist[i,:], des_pdist))



        accumu_dist= accumu_dist/sum(accumu_dist)


        works=0

        return des_images,accumu_dist,works


    else:
        return None,None,None
