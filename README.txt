Autores: Valentina Barbetty Arango | Lenin Esteban Carabalí Moreno | Juan José Cortés Rodríguez
Fecha: Febrero-Julio 2025

# Método de Newton-Raphson para el sistema NO LINEAL

En esta parte del proyecto se implementa la simulación de flujo de fluidos en una malla bidimensional utilizando el método de Newton-Raphson para resolver sistemas de ecuaciones no lineales.


El método de Newton-Raphson está implementado en:
- **[`core/solvers.py`](core/solvers.py)**: Funciones `newton_raphson_step_spsolve()` y `newton_raphson_step_gauss()`
- **[`core/equations.py`](core/equations.py)**: Función `F()` (sistema no lineal) y `Jacobiano()` (matriz jacobiana)

## Ejecución

### Requisitos
```bash
pip install numpy scipy matplotlib
```

**Con Método spsolve**
```bash
python main_spsolve.py
```

**Con Eliminación de Gauss**
```bash
python main_gauss.py
```

## Qué Esperar al Ejecutar

### 1. Visualización en Tiempo Real
- **Gráficas**: Muestra la evolución del campo de velocidades en cada iteración y al final la grafica de la matriz Jacobiana

### 2. Información en Consola
```
Iteración 1, error: 2.345678e-01
Iteración 2, error: 1.234567e-02
...
Convergencia alcanzada.
```