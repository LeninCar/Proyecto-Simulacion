import numpy as np
from config import FILAS, COLUMNAS, VY_CONSTANTE

def inicializar_malla():
    """Inicializa la malla con condiciones de frontera"""
    vx = np.zeros((FILAS, COLUMNAS))
    vy = np.zeros((FILAS, COLUMNAS))
    
    # Condiciones de frontera
    vx[0, :] = 0; vx[-1, :] = 0
    vy[0, :] = VY_CONSTANTE; vy[-1, :] = VY_CONSTANTE
    vx[1:6, 0] = 1
    vx[0, 0] = 0
    vx[6, 0] = 0
    vy[:, 0] = VY_CONSTANTE
    vx[:, -1] = 0
    vy[:, -1] = VY_CONSTANTE
    
    # Condición inicial en puntos internos
    for i in range(1, FILAS-1):
        for j in range(1, COLUMNAS-1):
            # vx[i, j] = max(0, 1 - (j / (cols-2)))  # Disminución lineal
            vx[i, j] = 0.5
            vy[i, j] = VY_CONSTANTE
    
    return vx, vy

def obtener_dimensiones():
    """Retorna las dimensiones de la malla interna"""
    nx = FILAS - 2
    ny = COLUMNAS - 2
    N = nx * ny
    return nx, ny, N
