"""
Enrique Carro Garrido 

Práctica 4: Transformación isométrica afín    
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
from skimage import io
from math import cos, sin, pi
from scipy.spatial import ConvexHull

# Transformación afín que se utilizará en ambos apartados. Sólo se necesita
# definir el centroide, el diámetro y el vector de la traslación
def f(t, v, c, X):
    # Cambiamos la referencia afín para que C sea el centro de la referencia
    X_ = X - c
    # Definimos la rotación
    R = np.array([[cos(3*pi*t), -sin(3*pi*t), 0],
                  [sin(3*pi*t),  cos(3*pi*t), 0],
                  [0,            0,           1]])
    # Aplicamos la transformación afín siendo C el centro de la referencia
    Y_ = np.matmul(X_, R.T) + v*t
    # Restauramos la referencia afín a la original
    Y = Y_ + c
    # Devolvemos el resultado
    return Y

# Cálculo del centroide y del diámetro del sistema que se utilizará en ambos
# apartados y para evitar que se repita el código.
def datos(X):
    # Definimos el centroide del sistema
    c = 1/N * np.sum(X, axis=0)

    # Definimos el diámetro del sistema (= diámetro de la envolvente convexa)
    hull = ConvexHull(X)
    d = 0
    for i in range(len(hull.vertices)):
        for j in range(i+1, len(hull.vertices)):
            p1 = X[hull.vertices[i]]
            p2 = X[hull.vertices[j]]
            distance = np.linalg.norm(p1-p2)
            if distance > d:
                d = distance
    return c, d

# i) Genera una figura en 3 dimensiones (puedes utilizar la figura 1 de la plantilla) y realiza
# una animación de una familia paramétrica continua que reproduzca desde la identidad hasta la
# transformación simultánea de una rotación de θ = 3π y una translación con v = (0, 0, d), donde
# d es el diámetro mayor de S.

# Utilizamos la figura 1 de la plantilla, a partir de la cual obtenemos un vector de puntos
# en el espacio para simplificar los cálculos y definimos N, el número total de puntos
X1, X2, X3 = axes3d.get_test_data(0.05)
X = np.array([X1.reshape(-1), X2.reshape(-1), X3.reshape(-1)]).T
N = X.shape[0] # N := número de puntos = número de columnas de X.

# Recuperamos el shape de las coordenadas para reestablecerlo en la imagen, ya que 
# contour sólo funciona con matrices.
shape = X1.shape

# Definimos el centroide y el vector de traslación
c, d = datos(X)
v = np.array([0,0,d])

# Creamos la figura sobre la que dibujaremos la animación
fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')

# Creamos la animación
def animate(t):
    # Obtenemos la imagen de la transformación afín en el instante t
    Y = f(t, v, c, X)
    # Para meter los datos en Axes.contour se necesita que sean 2D
    Y1 = np.reshape(Y[:,0], shape)
    Y2 = np.reshape(Y[:,1], shape)
    Y3 = np.reshape(Y[:,2], shape)
    # Borramos lo que había antes en Axes
    ax.clear()
    # Definimos los límites
    ax.set_xlim(np.min(X[:,0]),np.max(f(t=1,
                                    v=v,
                                    c=c,
                                    X=X)[:,0]))
    ax.set_ylim(np.min(X[:,1]),np.max(f(t=1,
                                        v=v,
                                        c=c,
                                        X=X)[:,1]))
    ax.set_zlim(np.min(X[:,2]),np.max(f(t=1,
                                        v=v,
                                        c=c,
                                        X=X)[:,2]))
    # Ploteamos el resultado
    cset = ax.contour(Y1, Y2, Y3, 16, extend3d=True, cmap = plt.cm.get_cmap('viridis'))
    ax.clabel(cset, fontsize=9, inline=1)
    # Actualizamos el contador del tiempo
    ax.text(0.2,0.2,0.0, "t = {:,.3f}".format(t), transform=ax.transAxes, ha="right", 
            bbox={'facecolor':'w', 'alpha':0.5, 'pad':5})
    return ax,

def init():
  return animate(0),

ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,1.025,0.025), init_func=init,
                             interval=20)
ani.save("Practica4\\p4i.gif", fps = 10)

# ii) Dado el sistema representado por la imagen digital ‘arbol.png’, considera el
# subsistema σ dado por el segundo color(verde) cuando verde < 240. ¿Dónde se sitúa
# el centroide? Realiza la misma transformación que en el apartado anterior, con 
# θ=3π y v=(d,d,0), donde d es el diámetro mayor de σ.

# Obtenemos la imagen digital que representa el sistema
tree = io.imread('Practica4\\arbol.png')

# Obtenemos las coordenadas de la imagen
xyz = tree.shape
x = np.arange(0,xyz[0],1)
y = np.arange(0,xyz[1],1)
xx,yy = np.meshgrid(x, y)
xx = np.asarray(xx).reshape(-1)
yy = np.asarray(yy).reshape(-1)
z  = tree[:,:,1]
zz = np.asarray(z).reshape(-1)

# Variables de estado: coordenadas
x0 = xx[zz<240]
y0 = yy[zz<240]
z0 = zz[zz<240]/256.
X = np.array([x0, y0, z0]).T
N = X.shape[0]

# Variable de estado: color
col = plt.get_cmap("viridis")(np.array(0.1+z0))

# Obtenemos el centroide y el vector de traslación para la transformación afín
c, d = datos(X)
v = np.array([d,d,0])

# Definimos los límites del plot
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(np.min(X[:,0]),np.max(f(t=1,
                                    v=v,
                                    c=c,
                                    X=X)[:,0]))
ax.set_ylim(np.min(X[:,1]),np.max(f(t=1,
                                    v=v,
                                    c=c,
                                    X=X)[:,1]))

# Creamos los Artist de la animación
time = ax.text(0.1, 0.9, "",
          bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, 
          transform=ax.transAxes, ha="center")
scat = ax.scatter(x0, y0, c=col, s=0.1)

# Creamos la animación
def animate(t):
  # Ploteamos la imagen en el instante t
  scat.set_offsets(f(t=t,
                     v=v,
                     c=c,
                     X=X)[:,:2])
  # Actualizamos el contador del tiempo
  time.set_text("t = {:,.3f}".format(t))
  return scat, time,

ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,1.025,0.025), interval=20)
ani.save("Practica4\\p4ii.gif", fps = 10)