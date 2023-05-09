import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# En primer lugar definimos la función que obtiene las coordenadas cartesianas 
# de la esfera unidad menos el polo sur.
def paralelos(R, N):
  """
  Función que define 2*N puntos de cada circunferencia de radios r \in R
  de la esfera unidad.
  """
  X, Y, Z = np.empty(shape=(2*len(R) - 2, 2*N)), np.empty(shape=(2*len(R) - 2, 2*N)), np.empty(shape=(2*len(R) - 2, 2*N))
  for i in range(len(R)):
    # Coordenada X
    X[i,:N] = np.linspace(-np.sqrt(R[i]), np.sqrt(R[i]), N)
    X[i,N:] = -X[i,:N]
    if 0 < i < len(R) - 1: X[2*(len(R) - 1) - i,:] = X[i,:]
    # Coordenada Y
    Y[i,:N] = np.sqrt(np.abs(R[i] - X[i,:N]**2))
    Y[i,N:] = -Y[i,:N]
    if 0 < i < len(R) - 1: Y[2*(len(R) - 1) - i,:] = Y[i,:]
    # Coordenada Z
    Z[i,:] = np.sqrt(np.abs(1. - X[i,:]**2 - Y[i,:]**2))
    if 0 < i < len(R) - 1: Z[2*(len(R) - 1) - i,:] = -Z[i, :]
  return X, Y, Z

# Obtenemos las coordenadas cartesianas
R = np.linspace(0., 1., 100)
N = 100
X, Y, Z = paralelos(R, N)

fig=plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

ax.set_xticks(np.linspace(-1.0, 1.0, 5))
ax.set_yticks(np.linspace(-1.0, 1.0, 5))
ax.set_zticks(np.linspace(-1.0, 1.0, 5))

plt.title('Esfera unidad')
plt.savefig("Esfera unidad.png")
plt.show()

# Definimos la familia paramétrica de funciones
def ft(t, x, y, z):
  # Para que se pueda calcular siempre el ángulo phi
  eps = 1e-16
  # Cambiar cartesianas a esféricas
  theta = np.pi - np.arctan2(np.sqrt(x**2 + y**2), z)
  phi = np.arctan2(y,x + eps)
  # Deformamos la esfera
  xt = x*(1 - t) + np.cos(phi)*t / np.tan(theta/2)
  yt = y*(1 - t) + np.sin(phi)*t / np.tan(theta/2)
  zt = z*(1 - t)
  return xt, yt, zt

Xt, Yt, Zt = ft(0.5, X, Y, Z)

# Podemos ver el resultado de la proyección estereográfica
fig  = plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')

X1, Y1, Z1 = ft(1, X, Y, Z)

ax.plot_surface(X1, Y1, Z1, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_zlim(-1., 1.)

plt.title('Proyección estereográfica')
plt.savefig("Proyección estereográfica.png")

# Representamos 15 fotogramas de la proyección estereográfica, haciendo zoom o sin hacerlo

# Haciendo zoom
# Creamos la figura sobre la que dibujaremos la animación
fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')

# Creamos la animación
def animate(t):
    # Obtenemos la imagen de la transformación afín en el instante t
    Xt, Yt, Zt = ft(t, X, Y, Z)
    # Borramos lo que había antes en Axes
    ax.clear()
    # Determinamos los límites del plot
    ax.set_xlim(-1., 1.)
    ax.set_ylim(-1., 1.)
    ax.set_zlim(-1., 1.)
    # Ploteamos el resultado
    ax.plot_surface(Xt, Yt, Zt, rstride=1, cstride=1, cmap='viridis', alpha=0.5, edgecolor='none')
    # Actualizamos el contador del tiempo
    ax.text(0.2,0.2,0.0, "t = {:,.3f}".format(t), transform=ax.transAxes, ha="right", 
            bbox={'facecolor':'w', 'alpha':0.5, 'pad':5})
    return ax,

def init():
  return animate(0),

ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,1 + 1/15,1/15), init_func=init)
ani.save("Proyección estereográfica zoom.gif", fps = 5)

# Sin hacerlo
# Creamos la figura sobre la que dibujaremos la animación
fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')

# Creamos la animación
def animate(t):
    # Obtenemos la imagen de la transformación afín en el instante t
    Xt, Yt, Zt = ft(t, X, Y, Z)
    # Borramos lo que había antes en Axes
    ax.clear()
    # Determinamos los límites del plot
    ax.set_zlim(-1., 1.)
    # Ploteamos el resultado
    ax.plot_surface(Xt, Yt, Zt, rstride=1, cstride=1, cmap='viridis', alpha=0.5, edgecolor='none')
    # Actualizamos el contador del tiempo
    ax.text(0.2,0.2,0.0, "t = {:,.3f}".format(t), transform=ax.transAxes, ha="right", 
            bbox={'facecolor':'w', 'alpha':0.5, 'pad':5})
    return ax,

def init():
  return animate(0),

ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,1 + 1/15,1/15), init_func=init)
ani.save("Proyección estereográfica.gif", fps = 5)
