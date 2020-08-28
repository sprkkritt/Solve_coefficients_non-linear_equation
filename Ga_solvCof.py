
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random as rnd
def Format_Eq(lst_cof):
    #--format equation
    lst_cof = lst_cof.split(',')
    lst_cof = [int(x) for x in lst_cof]
    lst_cof_screen = [str(lst_cof[len(lst_cof)-1-x]) + 'x^'+ str(x) for x in range(len(lst_cof)-1,-1,-1)]
    eq = ' + '.join(lst_cof_screen)
    eq = eq + ' = 0'
    #--format screen
    print('-'*(len(eq)+23))
    print('|'+' '*(len(eq)+21)+'|')
    print('|     Equation : {}     |'.format(eq))
    print('|'+' '*(len(eq)+21)+'|')
    print('-'*(len(eq)+23))
    return lst_cof
    #return [[cof,pow],[],...,[]]


def Init_Population(n_cof_pow): # n_cof_pow = numbers of cof
    #--Init Population --> chr (a_(i),a_(i-1)....,a_(0))
    print('Initial Population :')
    chr = []
    for i in range(100):
        gene = []
        for _ in range(n_cof_pow):
            gene.append(round(rnd.uniform(-1000,1000),6))
        chr.append(gene)
        #print init chromosome
        print('Chromosome {} : {}'.format(i+1, chr[i]))
    return chr



def solveEq(lst_cof):
    lst_cof = Format_Eq(lst_cof)
    cof_pow = [[int(lst_cof[len(lst_cof)-1-x]), int(x)] for x in range(len(lst_cof)-1,-1,-1)]
    #-Y
    start = -50
    stop = 60
    X = np.arange(start, stop)
    Y = []
    for x in X:
        Y_VAL = 0
        for c_p in cof_pow:
            Y_VAL += c_p[0]*(x**c_p[1])
        Y.append(Y_VAL)
    Y_SUM = sum(Y)
    
    #-----------------Start GA Algorithm--------------------
    #--init chromosome and variables
    pop = Init_Population(len(cof_pow))
    mean_fitness = []
    kill_point = len(pop)//2
    stop = False
    best_mse_each_round = []
    count = 0
    while not stop :
        count += 1
        if count == 10000:
            stop = True
        #-MEAN S ERROR
        MSE = [] 
        Fitness = []
        mse_idx = [] # 50 first mse val(ASC)
        mse_val = 0 #best mse each round
        Y_Hat = [] #list of yhat value
        print('-'*50)
        print('round : {}'.format(count))

        #--Selection
        for idx, ch in enumerate(pop):
            #for each chromosome
            Y_hat = 0
            for x in X:
                for c, p in  zip(ch, cof_pow):
                    Y_hat += c*(x**p[1])
            Y_Hat.append(round(Y_hat, 20))
            # print(Y_Hat_Sum)
            MSE.append(round((((Y_SUM - Y_Hat[idx])**2)/len(pop)), 20))
            if MSE[-1] < 0:
                print('Y_Sum: {} and Y_hat: {}'.format(Y_SUM, Y_Hat[-1]))
                print('Ysum  - Yhat  = {} - {} = {}'.format(Y_SUM, Y_Hat[-1], Y_SUM - Y_Hat[-1] ))
                print('((Ysum - Yhat)**2)/ = {}'.format(((Y_SUM - Y_Hat[-1])/Y_SUM**2)))
                print()
                stop = True
        mse_idx =  np.array(MSE).argsort()
        mse_val = round(MSE[mse_idx[0]],5)
        best_mse_each_round.append(mse_val)
        print('best chromosome :\nidx = {}\nchr = {}\nMSE = {}'.format(mse_idx[0]+1, pop[mse_idx[0]], mse_val))
        print('-'*50)
        print()

        #--Fitness
        Fitness = [round(1/(1+MSE[x]), 20) for x in mse_idx[:kill_point]]
        mean_fitness.append(round(sum(Fitness)/len(Fitness),20))
        if Fitness[0] >= 0.95:
            stop = True
            print('Chromosome :', pop[mse_idx[0]])
            print('Fitness :', Fitness[0])
            y_hat = []
            for x in X:
                Yhat_VAL = 0
                for c, e in zip(pop[mse_idx[0]], cof_pow):
                    Yhat_VAL += c*(x**e[1])
                y_hat.append(Yhat_VAL)
            plt.plot(X, Y, label='Y')
            plt.plot(X, y_hat, label='Y hat')
            plt.legend(loc='upper right')
            plt.title('Ganetic algorithm for Solv Coefficients Nth-degree Polynomial')
            plt.show()

            #plot graph mean fitness
            X = np.arange(len(mean_fitness))
            plt.plot(X, mean_fitness, label = 'mean fitness')
            plt.xlabel('ROUND')
            plt.ylabel('MEAN FITNESS')
            plt.legend(loc='upper left')
            plt.show()

        #--Cross Over
        offspring1 = []
        offspring2 = []
        for i in range(kill_point,len(pop),2):
            cross_point = np.random.randint(0, len(pop[0]), size=len(pop[0]))
            offspring1 = np.concatenate( ( pop[i][:cross_point[0]], pop[i+1][cross_point[0]:] ) )
            offspring2 = np.concatenate( ( pop[i+1][:cross_point[1]], pop[i][cross_point[1]:] ) )
            pop[i] = offspring1
            pop[i+1] = offspring2
        
        #--Mutation
        for i in range(1, len(pop)):
            val = np.random.randint(-10,10)
            pos = np.random.randint(0,len(pop[0]))
            pop[i][pos] = val
        pop = pop

    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--coefficient', required=True, help='Cofficient of equation, From x^n,x^n-1,...,x^0 // Ex. 2,3,4 -> 2x^2+3x+4 // Ex. 2,-,3 -> 2x^2+3 // Ex. " -3,2,3" -> -3x^2+2x+3 ')
    args = vars(ap.parse_args())
    solveEq(args['coefficient'])

