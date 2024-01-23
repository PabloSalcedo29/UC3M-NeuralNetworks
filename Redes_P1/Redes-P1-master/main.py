from distutils.log import error
import tensorflow as tf
import numpy as np
import pandas as pd
import random
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

from sklearn import preprocessing
import os
import time

def ADAline(train, val, test):
    if os.path.exists("salidas.txt"):
        os.remove('salidas.txt')
    else:
        print("No existe el archivo no existe")

    if os.path.exists("errores_train.txt"):
        os.remove('errores_train.txt')
    else:
        print("No existe el archivo no existe")
    
    if os.path.exists("errores_val.txt"):
        os.remove('errores_val.txt')
    else:
        print("No existe el archivo no existe")

    if os.path.exists("errores_test.txt"):
        os.remove('errores_test.txt')
    else:
        print("No existe el archivo no existe")

    errorPrevio = 10000
    peso = []
    tasaAprendizaje = 0.012
    umbral = random.random()

    for x in range(22):
        peso.append(random.random())

    salida_real()
    
    for numCiclos in range(0, 500):
        
      
        inicio = time.time()
        
        for fila in train:
            sum = 0
            cont = 0
            for columna in fila[:-1]:
                sum += columna * peso[cont]
               
            sum += umbral
            
            for columna in fila[:-1]:
                peso[cont] += tasaAprendizaje*(fila[-1] - sum) * columna
                cont += 1
            umbral += tasaAprendizaje * (fila[-1]-sum)

        final = time.time()

        suma_desnorm = (sum*(max_aux-min_aux))+min_aux

        with open("salidas.txt", "a") as p:
            p.writelines("\n"+ str(suma_desnorm)+ '\n')

        errorTotal_train = evaluacionError(umbral, peso, train)
        errorTotal_val = evaluacionError(umbral, peso, val)
        errorTotal_test = evaluacionError(umbral, peso, test)

        errores_train = []
        errores_val = []
        errores_test = []
        errores_train.append(errorTotal_train)
        errores_val.append(errorTotal_val)
        errores_test.append(errorTotal_test)

        with open("errores_train.txt", "a") as et:
            et.writelines("\n"+ str(errores_train)+ '\n')
        with open("errores_val.txt", "a") as ev:
            ev.writelines("\n"+ str(errores_val)+ '\n')
        with open("errores_test.txt", "a") as ev:
            ev.writelines("\n"+ str(errores_test)+ '\n')
            
        print("Entrenamiento: Error en ciclo " +str(numCiclos)+ " = " +str(errorTotal_train))
        print("Validación: Error en ciclo " +str(numCiclos)+ " = " +str(errorTotal_val)) #se para porque el error ha aumentado
        print("Test: Error en ciclo " +str(numCiclos)+ " = " +str(errorTotal_test)) #se para porque el error ha aumentado


        if errorTotal_val > errorPrevio:

            
            print("La red esta sobreaprendiendo, no se ejecutan mas ciclos, el error mínimo es: "+str(errorPrevio))
            
            with open("pesos.txt", "w") as p:
                p.writelines("\n"+ str(peso)+ '\n')
                p.writelines("\n"+"umbral: "+str(umbral) + '\n') #guardar los pesos y el umbral en un fichero
            
            tiempo = final-inicio
            print("Tiempo de ejecución: " +str(tiempo)) 
            quit()

        else:
            errorPrevio = errorTotal_val


    with open("pesos.txt", "w") as p:
        p.writelines("%sn" % line for line in peso)
        p.writelines("\n"+"umbral "+str(umbral) + '\n')

    print("El error mínimo es: "+ str(errorPrevio))
    
    tiempo = final-inicio
    print("Tiempo de ejecución: " +str(tiempo)) 

def evaluacionError(umbral, peso, datos):
    error=0
    for fila in datos:
        sum = 0
        for columna in fila[:-1]:
            sum += columna * peso[int(columna)]
        sum += umbral
        error += ((fila[21]-sum)**2)
    
    return error/len(datos)  #numero filas del conjunto

#se para cuando el error medio de la iteracion i sea mayor que el error en i-1
#se guarda el numero ce ciclos i-1, cuando el error fue mínimo
#formula del error = sumatorio((salida_esperada[i] - sum[i])^2)/numero filas

def evaluation(datos, peso, umbral):
    error = 0
    for fila in datos:
        sum = 0
        cont = 0
        for columna in fila[:-1]:
            sum += columna * peso[cont]
            cont += 1
        sum += umbral
        error += ((fila[-1] - sum)**2)       #No se si esto va al cuadrado
    return (error/len(datos))

aux_set = pd.read_csv('compactiv.csv', delimiter=',')
#train = np.array(aux_set.iloc[:, 0])

aux_normed = (aux_set - aux_set.min(axis=0)) / \
    (aux_set.max(axis=0) - aux_set.min(axis=0))

max_aux = aux_set.max(axis=0)
min_aux = aux_set.min(axis=0)

array = np.array(aux_normed)
np.random.shuffle(array)

training_set = []
for x in range(0, 5736):
    training_set.append(array[x])


validation_set = []
for x in range(5736, 6964):
    validation_set.append(array[x])

test_set = []
for x in range(6964, 8192):
    test_set.append(array[x])


def salida_real():
    salida_deseada = []
    salida_set = pd.read_csv('compactiv.csv', delimiter=',')
    array = np.array(salida_set)
    salida_deseada.append(array[:,-1])
            
    with open("salidas.txt", "a") as p:
        p.writelines("\n"+ str(salida_deseada)+ '\n')


ADAline(training_set, validation_set, test_set)
