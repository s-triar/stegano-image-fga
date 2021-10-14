
import math
import random
import numpy as np
from . import stegonize as steg

class Fga():
    def __init__(self, 
            globalLearningRate, 
            diversityRate, 
            threshold, 
            maxIter, 
            jumlahPopulasi, 
            img,
            secret,
            ):
        self.flat_img=img.flatten()
        # self.flat_secret=flat_secret.flatten()
        self.img=img
        self.secret = secret
        self.biner = 0
        self.globalLearningRate =  globalLearningRate
        self.diversityRate = diversityRate
        self.threshold = threshold
        self.maxIter = maxIter
        self.jumlahPopulasi = jumlahPopulasi
        self.populasi, self.blueprint, self.nKromosom, self.shiftingSecretData, self.repeatShifting,self.swapping,self.swappingStartPoint,self.swappingDirection,self.dataPolarity = self.generatePopulationInit(self.jumlahPopulasi)
        # print("populasi", self.populasi, len(self.populasi))
        # print("blue print", self.blueprint, len(self.blueprint))
        self.best = None
        self.bestScore = 999999999999999
        self.bestAt=99999999999999
        self.dataFitness=[]
        # self.dataPekerja = dataPekerja
        # self.jumlahPekerja = jumlahPekerja
        # self.jumlahData = jumlahData
        # self.pembagianPekerja=pembagianPekerja
    
    # def calculateNChromosome(self, jumlahData):
    #     i = 0
    #     while(True):
    #         if(math.pow(2,i) > jumlahData):
    #             break
    #         i+=1
    #     return i

    def generatePopulationInit(self, jumlahPopulasi):
        shiftingSecretData = len(bin(min(self.secret.shape[0],self.secret.shape[1])-1)[2:])
        # print((shiftingSecretData))
        repeatShifting = len(bin(min(self.secret.shape[0],self.secret.shape[1])-1)[2:])
        # print(repeatShifting)
        # check operation if not devided by 4
        swapping = len(bin(int((self.secret.shape[0]*self.secret.shape[1])/4)-1)[2:])
        print(swapping)
        swappingStartPoint = 1
        swappingDirection = 1
        dataPolarity = 4
        nKromosom = shiftingSecretData+repeatShifting+swapping+swappingStartPoint+swappingDirection+dataPolarity
        populasi = []
        for i in range(jumlahPopulasi):
            individu = []
            for j in range(nKromosom):
                individu.append(random.random())
            populasi.append(individu)
        blueprint = []
        for i in range(nKromosom):
            blueprint.append(0.5)
        return populasi, blueprint, nKromosom, shiftingSecretData, repeatShifting,swapping,swappingStartPoint,swappingDirection,dataPolarity

    def generateBlueprint(self, population):
        blueprint = [0 for i in (population[0])]
        for individu in population:
            for idx_k in range(len(individu)):
                blueprint[idx_k] += individu[idx_k]
        for idx_b in range(len(blueprint)):
            blueprint[idx_b] /= len(population)
        return blueprint

    def crossover(self, parent1, parent2):
        child = []
        cutPoint = random.randint(0,len(parent1)-1) #ini adalah point pertama
        cutPoint2 = random.randint(cutPoint,len(parent1)-1) #ini adalah point kedua dimulai dari titik point pertama
        #disitu tadi terdapat error karena bisa jadi cutpoint bernilai index maximum sehingga jika ditambah 1 maka akan error
        #errornya adalah parameter pertama random integer saat nilainya lebih besar dari parameter keduanya.
        for i in range(len(parent1)):
            if i <= cutPoint:
                child.append(parent1[i])
            elif i>cutPoint and i<=cutPoint2:
                child.append(parent2[i])
            else:
                child.append(parent1[i])
            bornFunction = (self.globalLearningRate*self.blueprint[i]) + ((1-self.globalLearningRate)*child[i])
            if(bornFunction < self.diversityRate):
                temp = self.diversityRate
                if(random.random() <= temp):
                    child[i]+=self.globalLearningRate
                else:
                    child[i]-=self.globalLearningRate
            elif(bornFunction > 1-self.diversityRate):
                temp = 1-self.diversityRate
                if(random.random() <= temp):
                    child[i]+=self.globalLearningRate
                else:
                    child[i]-=self.globalLearningRate
            else:
                temp = bornFunction
                if(random.random() <= temp):
                    child[i]+=self.globalLearningRate
                else:
                    child[i]-=self.globalLearningRate
        return child
    
    def bornAnIndividual(self, chromosome):
        individu = []
        for i in range(len(chromosome)):
            bornFunction = (self.globalLearningRate*self.blueprint[i]) + ((1-self.globalLearningRate)*chromosome[i])
            if(bornFunction < self.diversityRate):
                temp = self.diversityRate
                if(random.random() <= temp):
                    individu.append(1)
                else:
                    individu.append(0)
            elif(bornFunction > 1-self.diversityRate):
                temp = 1-self.diversityRate
                if(random.random() <= temp):
                    individu.append(1)
                else:
                    individu.append(0)
            else:
                temp = bornFunction
                if(random.random() <= temp):
                    individu.append(1)
                else:
                    individu.append(0)
        return individu

    def extractKromosom(self, individu):
        bins = [self.shiftingSecretData, self.repeatShifting,self.swapping,self.swappingStartPoint,self.swappingDirection,self.dataPolarity]
        startInd = 0
        intChr =[]
        # print(intChr)
        for i in range(0, len(bins)-1):
            r = ""
            startIndCP =startInd
            for ichr in range(startIndCP, bins[i]+startIndCP+1):
                r = r + str(individu[ichr])
                startInd=ichr
            # print(r)
            intChr.append(int(r,2))
            # print(intChr)
        t=""
        for i in individu[-bins[-1]:]: 
            t=t+str(i)
        intChr.append(t)
        
        return intChr
    
    def doStegano(self, intChro):
        secretCopy = self.secret.copy()
        offset = intChro[0]
        repeatShift = intChro[1]
        for i in range(0, repeatShift):
            secretCopy = steg.shifting(secretCopy,offset)
        n_member, start_flag, direction, data_polarity = intChro[2], intChro[3], intChro[4], intChro[5]
        secretCopy = steg.swapping(secretCopy,n_member, start_flag, direction, data_polarity)
        return secretCopy


    def calculateFitness(self, individu):
        # print("individu", individu)
        intChr = self.extractKromosom(individu)
        # print("int Chromosome", intChr)
        secret_bin = self.doStegano(intChr)
        img_bin = self.flat_img.copy()[:len(secret_bin)]
        # print(len(img_bin),len(secret_bin))
        c= np.logical_xor(img_bin,secret_bin)
        # print("c",c)
        fitness= np.sum(c==True)
        # print("fitness",fitness)
        return fitness
    
    def bubblesort(self, fitness,individu,populasi):
        n=len(fitness)
        for i in range(n):
            for j in range(0,n-i-1):
                if(fitness[j] < fitness[j+1]):
                    fitness[j], fitness[j+1] = fitness[j+1],fitness[j]
                    individu[j], individu[j+1] = individu[j+1], individu[j]
                    populasi[j], populasi[j+1] = populasi[j+1], populasi[j]
        return fitness, individu, populasi

    def selectionElitism(self, individu, fitnesses, iterasi):
        fitnesses, individu,  self.populasi = self.bubblesort(fitnesses,individu, self.populasi)
        self.dataFitness.append(fitnesses[0])
        if(self.bestScore>fitnesses[0]):
            # print("=================================", iterasi)
            self.bestAt=iterasi
            self.best = individu[0]
            self.bestScore = fitnesses[0]
        while(len(self.populasi)>self.jumlahPopulasi):
            del self.populasi[-1]
            del fitnesses[-1]
            del individu[-1]
    
    def random_injection(self, persen):
        # disini akan melakukan pembangkitan individu dengan nilai random
        # cetakan code individu dapat dicopy dari generate populasi awal
        # self.biner= self.calculateNChromosome(self.jumlahData)  #baris ini akan menghitung berapa kromosom yang diperlukan
        # pada setiap pekerja
        # nKromosom = self.biner*self.jumlahPekerja #pada baris ini akan menhitung total kromosom pada 1 individu

        nGenerate = math.ceil(persen*self.jumlahPopulasi) #pada baris ini akan menghitung berapa banyak individu yang digenerate (dibulatkan ke atas)
        for i in range(nGenerate):
            indiv = []              #pada baris ini membuat wadah kromosom
            for j in range(self.nKromosom):
                indiv.append(random.random()) #pada baris ini mengisi wadah indiv dengan nilai random 0-1 sebanyak jumlah kromosom tiap individu
            index = -1*(i+1) #ini merupakan index untuk populasi yang akan diganti nilainya. dan agar pergantian dimulai dari index paling belakang
            self.populasi[index] = indiv #mengganti populasi dengan hasil random injection

    def Run(self):
        iterasi = 0
        while iterasi < self.maxIter:
            point1 = random.randint(0, self.jumlahPopulasi-1)
            p1 = self.populasi[point1]
            point2 = random.randint(0, self.jumlahPopulasi-1)
            p2 = self.populasi[point2]
            child = self.crossover(p1,p2)
            self.populasi.append(child)
            individus = []
            for ch in self.populasi:
                individus.append(self.bornAnIndividual(ch))
            fitnesses = []
            for ind in range(len(individus)):
                # print('sebelum',individus[ind])
                # fit, tempIndividu = self.calculateFitness(individus[ind])
                fit = self.calculateFitness(individus[ind])
                # print('sesudah',tempIndividu)
                # individus[ind]= tempIndividu
                fitnesses.append(fit)
            self.selectionElitism(individus,fitnesses, iterasi)
            print('iterasi',iterasi,self.best,self.bestScore)
            iterasi+=1
            # memanggil method random injection setiap 10 iterasi sekali
            if(iterasi%10==0):
                self.random_injection(0.6)
            self.blueprint = self.generateBlueprint(self.populasi)
        print('hasil terbaik : ')
        print('pada iterasi : ', self.bestAt)
        print("chromosome terpilih", self.best)
        secretRearranged = self.doStegano(self.extractKromosom(self.best))
        img_bin = self.flat_img.copy()[:len(secretRearranged)]
        return np.concatenate((secretRearranged, self.flat_img.copy()[len(secretRearranged):-len(self.best)], self.best), axis=None)