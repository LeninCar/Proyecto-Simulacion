import numpy as np
from config import FILAS, COLUMNAS  # Cambiar de ..config a config

def inicializar_malla():
    """Inicializa la malla física con condiciones de frontera"""
    malla = np.empty((FILAS, COLUMNAS))
    
    malla[0, :] = 0              # Borde superior
    malla[-1, :] = 0             # Borde inferior
    malla[:, -1] = 0             # Borde derecho
    malla[1:6, 0] = 1            # Entrada del fluido
    
    return malla

def obtener_dimensiones():
    """Retorna las dimensiones de la malla interna"""
    nx = FILAS - 2  # 5
    ny = COLUMNAS - 2  # 50
    N = nx * ny  # 250
    return nx, ny, N

def idx(i, j, ny):
    """Convierte índices 2D a índice 1D"""
    return i * ny + j