# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 23:53:36 2020

@author: robert monjo
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
#https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

os.getcwd()

#q = variable de posición, dq0 = \dot{q}(0) = valor inicial de la derivada
#d = granularidad del parámetro temporal
def deriv(q,dq0,d):
   #dq = np.empty([len(q)])
   dq = (q[1:len(q)]-q[0:(len(q)-1)])/d
   dq = np.insert(dq,0,dq0) #dq = np.concatenate(([dq0],dq))
   return dq

#Ecuación de un sistema dinámico continuo
#Ejemplo de oscilador simple
def F(q):
    k = 1
    ddq = - k*q
    return ddq

#Resolución de la ecuación dinámica \ddot{q} = F(q), obteniendo la órbita q(t)
#Los valores iniciales son la posición q0 := q(0) y la derivada dq0 := \dot{q}(0)
def orb(n,q0,dq0,F, args=None, d=0.001):
    #q = [0.0]*(n+1)
    q = np.empty([n+1])
    q[0] = q0
    q[1] = q0 + dq0*d
    for i in np.arange(2,n+1):
        args = q[i-2]
        q[i] = - q[i-2] + d**2*F(args) + 2*q[i-1]
    return q #np.array(q),

def periodos(q,d,max=True):
    #Si max = True, tomamos las ondas a partir de los máximos/picos
    #Si max == False, tomamos las ondas a partir de los mínimos/valles
    epsilon = 5*d
    dq = deriv(q,dq0=None,d=d) #La primera derivada es irrelevante
    if max == True:
        waves = np.where((np.round(dq,int(-np.log10(epsilon))) == 0) & (q >0))
    if max != True:
        waves = np.where((np.round(dq,int(-np.log10(epsilon))) == 0) & (q <0))
    diff_waves = np.diff(waves)
    waves = waves[0][1:][diff_waves[0]>1]
    pers = diff_waves[diff_waves>1]*d
    return pers, waves

#################################################################    
#  CÁLUCLO DE ÓRBITAS
################################################################# 

#Ejemplo gráfico del oscilador simple
q0 = 0.
dq0 = 1.
fig, ax = plt.subplots(figsize=(12,5))
plt.ylim(-1.5, 1.5)  
plt.rcParams["legend.markerscale"] = 6
ax.set_xlabel("t = n $\delta$", fontsize=12)
ax.set_ylabel("q(t)", fontsize=12)
iseq = np.array([1,1.1,1.5,1.8,3])
horiz = 32
for i in iseq:
    d = 10**(-i)
    n = int(horiz/d)
    t = np.arange(n+1)*d
    q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
    plt.plot(t, q, 'ro', markersize=0.5/i,label='$\delta$ ='+str(np.around(d,3)),c=plt.get_cmap("winter")(i/np.max(iseq)))
    ax.legend(loc=3, frameon=False, fontsize=12)
#plt.savefig('Time_granularity.png', dpi=250)

#Ejemplo de coordenadas canónicas (q, p)
#Nos quedamos con el más fino y calculamos la coordenada canónica 'p'
q0 = 0.
dq0 = 1.
horiz = 32
d = 10**(-3)
n = int(horiz/d)
t = np.arange(n+1)*d
q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
dq = deriv(q,dq0=dq0,d=d)
p = dq/2

#Ejemplo gráfico de la derivada de q(t)
fig, ax = plt.subplots(figsize=(12,5))
plt.ylim(-1.5, 1.5)  
plt.rcParams["legend.markerscale"] = 6
ax.set_xlabel("t = n $\delta$", fontsize=12)
ax.set_ylabel("dq(t)", fontsize=12)
plt.plot(t, dq, '-')

#Ejemplo de diagrama de fases (q, p) para una órbita completa
fig, ax = plt.subplots(figsize=(5,5))
plt.xlim(-1.1, 1.1)  
plt.ylim(-1, 1) 
plt.rcParams["legend.markerscale"] = 6
ax.set_xlabel("q(t)", fontsize=12)
ax.set_ylabel("p(t)", fontsize=12)
plt.plot(q, p, '-')
plt.show()


#Ejemplo de diagrama de fases (q, p) para un tiempo determinado
horiz = 0.251
ax = fig.add_subplot(1,1, 1)
seq_q0 = np.linspace(0.,1.,num=20)
seq_dq0 = np.linspace(0.,2,num=20)
q2 = np.array([])
p2 = np.array([])
for i in range(len(seq_q0)):
    for j in range(len(seq_dq0)):
        q0 = seq_q0[i]
        dq0 = seq_dq0[j]
        d = 10**(-3)
        n = int(horiz/d)
        t = np.arange(n+1)*d
        q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
        dq = deriv(q,dq0=dq0,d=d)
        p = dq/2
        q2 = np.append(q2,q[-1])
        p2 = np.append(p2,p[-1])
        plt.xlim(-2.2, 2.2)
        plt.ylim(-1.2, 1.2)
        plt.rcParams["legend.markerscale"] = 6
        ax.set_xlabel("q(t)", fontsize=12)
        ax.set_ylabel("p(t)", fontsize=12)
        plt.plot(q[-1], p[-1], marker="o", markersize= 10, markeredgecolor="red",markerfacecolor="red")
plt.show()

X = np.array([q2,p2]).T
hull = ConvexHull(X)
convex_hull_plot_2d(hull)

print("Perímetro:", hull.area)
print("Área:", hull.volume)        
print("Vértices:", X[hull.vertices])

#IMPORTANTE: ES EL VALOR MÁXIMO??
#IMPORTANTE: Variar d = 10**(-3) más fino para encontrar intervalo de error


#################################################################    
#  ESPACIO FÁSICO
################################################################# 

## Pintamos el espacio de fases
def simplectica(q0,dq0,F,col=0,d = 10**(-4),n = int(16/d),marker='-'): 
    q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
    dq = deriv(q,dq0=dq0,d=d)
    p = dq/2
    plt.plot(q, p, marker,c=plt.get_cmap("winter")(col))


Horiz = 12
d = 10**(-3)

fig = plt.figure(figsize=(8,5))
fig.subplots_adjust(hspace=0.4, wspace=0.2)
ax = fig.add_subplot(1,1, 1)
#Condiciones iniciales:
seq_q0 = np.linspace(0.,1.,num=14)
seq_dq0 = np.linspace(0.,2,num=14)
for i in range(len(seq_q0)):
    for j in range(len(seq_dq0)):
        q0 = seq_q0[i]
        dq0 = seq_dq0[j]
        col = (1+i+j*(len(seq_q0)))/(len(seq_q0)*len(seq_dq0))
        #ax = fig.add_subplot(len(seq_q0), len(seq_dq0), 1+i+j*(len(seq_q0)))
        simplectica(q0=q0,dq0=dq0,F=F,col=col,marker='ro',d= 10**(-3),n = int(Horiz/d))
ax.set_xlabel("q(t)", fontsize=12)
ax.set_ylabel("p(t)", fontsize=12)
#fig.savefig('Simplectic.png', dpi=250)
plt.show()

