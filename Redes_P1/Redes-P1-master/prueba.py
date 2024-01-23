from operator import le
import pandas as pd

aux_set = pd.read_csv('prueba.csv')
print(len(aux_set)**2)