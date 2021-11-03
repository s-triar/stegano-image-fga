
import math
import random
import numpy as np
from . import stegonize_with_sp as steg
import csv



class Ga():
    def __init__(self, 
            crossoverrate, 
            mutationrate, 
            maxIter, 
            jumlahPopulasi, 
            img,
            secret,
            shapeSecret,
            payloadType,
            path_log
            ):
        self.path_log = path_log
        self.flat_img=img.flatten()
        self.coverWidthBin, self.coverHeightBin = self.generateCoverBin(coverShape=img.shape,secretShape=shapeSecret, payloadType=payloadType)
        self.img=img
        self.secret = secret
        self.payloadType=payloadType
        self.biner = 0
        self.shapeSecret = shapeSecret
        self.crossoverrate =  crossoverrate
        self.mutationrate = mutationrate
        self.elitismrate = 0.4
        self.maxIter = maxIter
        self.jumlahPopulasi = jumlahPopulasi
        self.populasi, self.nKromosom, self.startPointX, self.startPointY, self.scanDir,self.shiftingSecretData, self.repeatShifting,self.swapping,self.swappingStartPoint,self.swappingDirection,self.dataPolarity = self.generatePopulationInit(self.jumlahPopulasi)
        self.best = None
        self.bestScore = 999999999999999
        self.bestAt=99999999999999
        self.dataFitness=[]

    def generateCoverBin(self, coverShape, secretShape,payloadType):
        print("payloadType",payloadType)
        if(payloadType==1): #image
            binX = [0 for i in range(0, len(bin(coverShape[0]-1)[2:]))]
            binY = [0 for i in range(0, len(bin(coverShape[1]-1)[2:]))]
            s_binX = bin(secretShape[0])[2:]
            s_binY = bin(secretShape[1])[2:]
            idx_i = len(binX)-1
            for i in range(len(s_binX)-1, -1,-1):
                binX[idx_i]=int(s_binX[i])
                idx_i=idx_i-1
            idx_j = len(binY)-1
            for i in range(len(s_binY)-1, -1,-1):
                binY[idx_j]=int(s_binY[i])
                idx_j=idx_j-1
            return binX, binY
        else:
            binX = [0 for i in range(0, len(bin((coverShape[0]*coverShape[1])-1)[2:]))]
            s_binX = bin(secretShape[0])[2:]
            idx_i = len(binX)-1
            for i in range(len(s_binX)-1, -1,-1):
                binX[idx_i]=int(s_binX[i])
                idx_i=idx_i-1
            return binX, []


    def generatePopulationInit(self, jumlahPopulasi):
        nKromosom, start_point_x, start_point_y, scan_dir, shiftingSecretData, repeatShifting,swapping,swappingStartPoint,swappingDirection,dataPolarity = steg.generateKromosom(self.secret.shape[0],self.secret.shape[1], (self.img[:-1,:].copy()).shape)
        populasi = []
        for i in range(jumlahPopulasi):
            individu = []
            for j in range(nKromosom):
                individu.append(random.randint(0, 1))
            populasi.append(individu)
        return populasi, nKromosom, start_point_x, start_point_y, scan_dir, shiftingSecretData, repeatShifting,swapping,swappingStartPoint,swappingDirection,dataPolarity

    def mutation(self, parent1):
        child = []
        cutPoint = random.randint(0,len(parent1)-1)
        for i in range(len(parent1)):
            c = parent1[i] if i!=cutPoint else random.randint(0,1)
            child.append(c)
        return child

    def crossover(self, parent1, parent2):
        child = []
        child2 = []
        cutPoint = random.randint(0,len(parent1)-1) #ini adalah point pertama
        cutPoint2 = random.randint(cutPoint,len(parent1)-1) #ini adalah point kedua dimulai dari titik point pertama
        #disitu tadi terdapat error karena bisa jadi cutpoint bernilai index maximum sehingga jika ditambah 1 maka akan error
        #errornya adalah parameter pertama random integer saat nilainya lebih besar dari parameter keduanya.
        for i in range(len(parent1)):
            if i <= cutPoint:
                child.append(parent1[i])
                child2.append(parent2[i])
            elif i>cutPoint and i<=cutPoint2:
                child.append(parent2[i])
                child2.append(parent1[i])
            else:
                child.append(parent1[i])
                child2.append(parent2[i])
        return child,child2


    def calculateFitness(self, individu):
        
        bins = [self.startPointX, self.startPointY, self.scanDir, self.shiftingSecretData, self.repeatShifting,self.swapping,self.swappingStartPoint,self.swappingDirection,self.dataPolarity]
        secret_bin = steg.combineImgBinWithSecretBin(self.img.copy(), self.secret.copy(), self.coverWidthBin, self.coverHeightBin,self.payloadType,bins, individu)
        img_bin = self.flat_img.copy()
        c= np.logical_xor(img_bin,secret_bin)
        fitness= np.sum(c==True)
        return fitness
    
    def bubblesort(self, fitness,populasi):
        n=len(fitness)
        for i in range(n):
            for j in range(0,n-i-1):
                if(fitness[j] > fitness[j+1]):
                    fitness[j], fitness[j+1] = fitness[j+1],fitness[j]
                    populasi[j], populasi[j+1] = populasi[j+1], populasi[j]
        return fitness, populasi

    def selectionElitism(self, fitnesses, iterasi):
        fitnesses, self.populasi = self.bubblesort(fitnesses, self.populasi)
        self.dataFitness.append(fitnesses[0])
        if(self.bestScore>fitnesses[0]):
            # print("=================================", iterasi)
            self.bestAt=iterasi
            self.best = self.populasi[0]
            self.bestScore = fitnesses[0]
        while(len(self.populasi)>self.jumlahPopulasi):
            del self.populasi[-1]
            del fitnesses[-1]
    
    def random_injection(self, rate):
        nGenerate = int(round(rate*self.jumlahPopulasi,0))
        # for i in range(0,nGenerate):
        #     del self.populasi[-1]
        for i in range(0,nGenerate):
            indiv = []              
            for j in range(self.nKromosom):
                indiv.append(random.randint(0,1)) 
            self.populasi.append(indiv) 

    def Run(self):
        
        iterasi = 0
        while iterasi < self.maxIter:
            
            fitnesses = []
            for ind in range(len(self.populasi)):
                fit = self.calculateFitness(self.populasi[ind])
                fitnesses.append(fit)
            self.selectionElitism(fitnesses, iterasi)
            print('iterasi',iterasi,self.best,self.bestScore)
            f = open(self.path_log, 'a+',encoding='UTF8', newline='')
            writer = csv.writer(f)
            writer.writerow([iterasi,"".join([str(ki) for ki in self.best]),self.bestScore])
            f.close()
            # if(iterasi%10==0):
            #     self.random_injection(0.6)
            print("populasi seblum cross and mutate",len(self.populasi))

            strt = int(round(self.crossoverrate*self.jumlahPopulasi,0))
            for i in range(0, int(round(strt/2,0))):
                point1 = random.randint(0, self.jumlahPopulasi-1)
                p1 = self.populasi[point1]
                point2 = random.randint(0, self.jumlahPopulasi-1)
                p2 = self.populasi[point2]
                child,child2 = self.crossover(p1,p2)
                self.populasi.append(child)
                self.populasi.append(child2)

            strt = int(round(self.mutationrate*self.jumlahPopulasi,0))
            for i in range(0, int(round(strt))):
                point1 = random.randint(0, self.jumlahPopulasi-1)
                p1 = self.populasi[point1]
                child = self.mutation(p1)
                self.populasi.append(child)
            print("populasi setelah cross and mutate",len(self.populasi))
            iterasi+=1

            
        print('hasil terbaik : ')
        print('pada iterasi : ', self.bestAt)
        print("chromosome terpilih", self.best)

        
        bins = [self.startPointX, self.startPointY, self.scanDir, self.shiftingSecretData, self.repeatShifting,self.swapping,self.swappingStartPoint,self.swappingDirection,self.dataPolarity]
        combined = steg.combineImgBinWithSecretBin(self.img.copy(), self.secret.copy(), self.coverWidthBin, self.coverHeightBin,self.payloadType,bins, self.best)
        return combined, self.bestAt, self.best
        # extractKromosom = steg.extractKromosom(bins, self.best)
        # secretRearranged = steg.doStegano(self.secret.copy(),extractKromosom)
        # img_bin = steg.startPointCover([extractKromosom[0],extractKromosom[1]],extractKromosom[2],self.img.copy()[:-1,:])
        # emb = np.concatenate((secretRearranged, img_bin.copy()[len(secretRearranged):]))
        # emb = steg.deStartPointCover([extractKromosom[0],extractKromosom[1]],extractKromosom[2],emb, self.img.copy()[:-1,:].shape)
        # emb = emb.flatten()
        # emb = np.concatenate((emb, self.img.copy()[-1,:].flatten()))
        # return np.concatenate((emb.copy()[:-(len(self.best)+len(self.coverWidthBin)+len(self.coverHeightBin)+1)], self.best, self.coverWidthBin, self.coverHeightBin,[self.payloadType]), axis=None), self.bestAt, self.best