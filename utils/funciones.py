# Librerias
import sympy as sp
import numpy as np 
import plotly.graph_objects as go 
import plotly.figure_factory as ff
from scipy.integrate import odeint
from sympy import symbols, sympify, Matrix, solve, simplify

# Funciones

# ----------------------------------------------------------------------------------------------------------------------
#       PAGE 1
# ----------------------------------------------------------------------------------------------------------------------

# Función Logistica Profesor
def ecuacion_logistica(K:float, P0:float, r:float, t0:float, t:float, cant:float, scale:float):
    """
    Retorna una gráfica de la ecuacion logistica con su campo vectorial.

    Parámetros:
    -------
    - K: Capacidad de carga.
    - P0: Poblacion Inicial.
    - r: Tasa de crecimineto poblacional.
    - t0: Tiempo inicial.
    - t: Tiempo final.
    - cant: Las particiones para el eje temporal y espacial.
    - scale: Tamaño del vector del campo vectorial.
    """

    # Rango de P y t
    P_values = np.linspace(0, K+5, cant)
    t_values = np.linspace(0, t, cant)

    # Crear una malla de puntos (P, t)
    T, P = np.meshgrid(t_values, P_values)

    # Definir la EDO
    dP_dt = r * P * (1 - P / K)

    # Solucion exacta de la Ecuación Logística
    funcion = K*P0*np.exp(r*t_values) / (P0*np.exp(r*t_values) + (K-P0)*np.exp(r*t0))

    # Campo vectorial: dP/dt (componente vertical)
    U = np.ones_like(T)  # Componente en t (horizontal)
    V = dP_dt           # Componente en P (vertical)

    # Crear el campo de vectores con Plotly
    fig = ff.create_quiver(
        T, P, U, V,
        scale=scale,
        line=dict(color='black', width=1),
        showlegend=False
    )

    # Crear la función logística
    fig.add_trace(
        go.Scatter(
            x = t_values,
            y = funcion,
            #mode = 'markers+lines',
            line=dict(color='blue'),
            name = 'Ecuación Logística'
        )
    )

    fig.add_trace(
        go.Scatter(
            x = [0, t],
            y = [K, K],
            mode = 'lines',
            line = dict(color='red', dash='dash'),
            name = 'Capacidad de carga'
        )
    )

    # Etiquetas para la gráfica
    fig.update_layout(
        title={
            'text':'Campo de vectores de dP/dt = rP(1 - P/k)',
            'x':0.5,
            'y':0.92,
            'xanchor':'center'
        },
        xaxis_title='Tiempo (t)',
        yaxis_title='Población (P)',
        width=800,
        template='plotly_white',
        margin=dict(l=10,r=10,t=90,b=0),
        legend=dict(orientation='h',y=1.1)
    )

    # contorno a la grafica
    fig.update_xaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='gray',
        showgrid=False
    )
    fig.update_yaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='gray',
        showgrid=False
    )

    return fig

# Funcion Logistica Alumnos
# 1. Averguar aguna Ecuacion de algun modelo y desarrollarlo con Sympy.
# 2. Mejorar la apariencia visual de la pagina.
# 3. Agregar un botón en la cual me permita activar y desactivar el campo de vectores (RETO)

# ----------------------------------------------------------------------------------------------------------------------
#       PAGE 2
# ----------------------------------------------------------------------------------------------------------------------

def ecuacion_armonica(m: float, k: float, x0: float, v0: float, t: float, cant: int):
    """
    Retorna una gráfica de la ecuación del oscilador armónico simple.

    Parámetros:
    -------
    - m: Masa del objeto oscilante.
    - k: Constante del resorte.
    - x0: Desplazamiento inicial.
    - v0: Velocidad inicial.
    - t: Tiempo máximo.
    - cant: Número de puntos de tiempo para la simulación.
    """

    # Frecuencia angular
    omega = np.sqrt(k / m)
    
    # Rango de tiempo
    t_values = np.linspace(0, t, cant)
    
    # Solución de la ecuación armónica
    x_values = x0 * np.cos(omega * t_values) + (v0 / omega) * np.sin(omega * t_values)
    
    # Crear la gráfica de la solución
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=t_values,
        y=x_values,
        mode='lines',
        line=dict(color='blue', shape='spline', width=2),  # Aquí se usa 'spline' para suavizar las líneas
        name='Oscilador Armónico'
    ))
    
    # Etiquetas y estilo
    fig.update_layout(
        title={'text': 'Movimiento del Oscilador Armónico', 'x': 0.5, 'y': 0.9, 'xanchor': 'center'},
        xaxis_title='Tiempo (t)',
        yaxis_title='Posición (x)',
        template='plotly_white',
        width=800,
        margin=dict(l=10, r=10, t=90, b=0),
        legend=dict(orientation='h', y=1.1)
    )
    
    fig.update_xaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='gray',
        showgrid=False
    )
    
    fig.update_yaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='gray',
        showgrid=False
    )
    
    return fig

# ----------------------------------------------------------------------------------------------------------------------
#       PAGE 3
# ----------------------------------------------------------------------------------------------------------------------

# Función para modelar la Ley de Enfriamiento/Calentamiento de Newton
def ley_enfriamiento_newton(T0: float, T_env: float, k: float, t: float, cant: int):
    """
    Resuelve la ecuación de la Ley de Enfriamiento/Calentamiento de Newton.

    Parámetros:
    -------
    - T0: Temperatura inicial del objeto.
    - T_env: Temperatura del ambiente.
    - k: Coeficiente de enfriamiento (positivo).
    - t: Tiempo máximo para la simulación.
    - cant: Número de puntos de tiempo para la simulación.
    """

    # Rango de tiempo
    t_values = np.linspace(0, t, cant)

    # Solución de la ecuación de enfriamiento
    T_values = T_env + (T0 - T_env) * np.exp(-k * t_values)

    # Crear la gráfica de la solución
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t_values,
        y=T_values,
        mode='lines',
        line=dict(color='blue'),
        name='Temperatura'
    ))

    # Etiquetas y estilo
    fig.update_layout(
        title={'text': 'Ley de Enfriamiento/Calentamiento de Newton', 'x': 0.5, 'y': 0.9, 'xanchor': 'center'},
        xaxis_title='Tiempo (t)',
        yaxis_title='Temperatura (°C)',
        template='plotly_white',
        width=800,
        margin=dict(l=10, r=10, t=90, b=0),
        legend=dict(orientation='h', y=1.1)
    )

    fig.update_xaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='gray',
        showgrid=False
    )

    fig.update_yaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='gray',
        showgrid=False
    )

    return fig

# ----------------------------------------------------------------------------------------------------------------------
#       PAGE 4
# ----------------------------------------------------------------------------------------------------------------------

def model_SIR(condiciones_iniciales, beta, gamma, t, t_cambio, nuevo_beta=None, nuevo_gamma=None):
    """
    Simula el modelo SIR (Susceptible-Infectado-Recuperado) para predecir la 
    propagación de una enfermedad infecciosa en una población, permitiendo cambios en 
    beta o gamma en un tiempo especificado.

    Parámetros:
    - condiciones_iniciales (list): Lista de tres elementos que representan las condiciones 
      iniciales para las poblaciones susceptibles (S), infectadas (I) y recuperadas (R).
    - beta (float): Tasa de transmisión inicial de la enfermedad (probabilidad de infección).
    - gamma (float): Tasa de recuperación inicial (proporción de infectados que se recuperan por unidad de tiempo).
    - t (int): Tiempo máximo de simulación.
    - t_cambio (int): Tiempo en el que se producirán cambios en los parámetros.
    - nuevo_beta (float, opcional): Nuevo valor de beta después de t_cambio.
    - nuevo_gamma (float, opcional): Nuevo valor de gamma después de t_cambio.

    Retorna:
    - fig (plotly.graph_objects.Figure): Gráfica de las poblaciones susceptibles, infectadas y 
      recuperadas a lo largo del tiempo.
    """

    # Definir las ecuaciones diferenciales para el modelo SIR
    def SIR_odeint(y, t, beta, gamma, tiempo_cambio, nuevo_beta, nuevo_gamma):
        S, I, R = y
        
        # Aplicar condición para cambiar beta y gamma
        if t > tiempo_cambio:
            beta = nuevo_beta  # Cambiar a nuevo beta
            gamma = nuevo_gamma  # Cambiar a nuevo gamma

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        
        return dSdt, dIdt, dRdt

    # Definir los puntos de tiempo sobre los que se resolverán las ODE
    puntos_tiempo = np.linspace(0, t, 100)
    
    # Resolver el sistema de ecuaciones diferenciales
    solucion = odeint(SIR_odeint, condiciones_iniciales, puntos_tiempo, 
                      args=(beta, gamma, t_cambio, nuevo_beta, nuevo_gamma))

    # Separar las soluciones para S, I y R
    S, I, R = solucion.T  

    # Crear la figura de Plotly
    fig = go.Figure()

    # Añadir trazas para cada población a la figura con colores actualizados
    fig.add_trace(go.Scatter(x=puntos_tiempo, y=S, mode='lines', name='Susceptibles', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=puntos_tiempo, y=I, mode='lines', name='Infectados', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=puntos_tiempo, y=R, mode='lines', name='Recuperados', line=dict(color='cyan')))

    # Actualizar el layout de la gráfica
    fig.update_layout(
        title={
            'text': 'Modelo SIR',
            'x': 0.5,
            'y': 0.92,
            'xanchor': 'center'
        },
        xaxis_title='Tiempo (t)',
        yaxis_title='Población (P)',
        width=800,
        template='plotly_white',
        margin=dict(l=10, r=10, t=90, b=0),
        legend=dict(orientation='v', x=1.05, y=0.5)  # Mover la leyenda a la derecha
    )

    # Añadir contornos a los ejes con transparencia
    fig.update_xaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='rgba(0, 255, 0, 0.3)',  # Línea de fondo verde con algo de transparencia
        showgrid=True
    )
    fig.update_yaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='rgba(0, 255, 0, 0.3)',  # Línea de fondo verde con algo de transparencia
        showgrid=True
    )

    return fig


# ----------------------------------------------------------------------------------------------------------------------
#       PAGE 5
# ----------------------------------------------------------------------------------------------------------------------

def classify_point(T, D):
    # Fórmula de la discriminante
    delta = T**2 - 4 * D
    
    if D < 0:
        return "Punto Silla "
    elif D > 0:
        if delta > 0:
            return "Nodo Estable " if T < 0 else "Nodo Inestable "
        elif delta < 0:
            return "Foco Estable " if T < 0 else "Foco Inestable "
        elif delta == 0:
            return "Nodo Degenerado "
    elif D == 0 and delta < 0:
        return "Centro "
    
    return "Inclasificable "

def jacobian_text(X, Y, crit_pts, vars):
    x, y = vars
    J = simplify(X.jacobian(Y))

    classification_results = []
    for p in crit_pts:
        J_at_point = J.subs({x: p[0], y: p[1]})
        T = simplify(J_at_point.trace())
        D = simplify(J_at_point.det())
        classification = classify_point(T, D)
        classification_results.append(f"({p[0]:.2f}, {p[1]:.2f}) es un {classification}")

    # Devuelve solo una lista de clasificaciones
    return classification_results


def calcular_jacobiano(funcion_f, funcion_g):
    x, y = symbols("x y")
    dx = sympify(funcion_f)
    dy = sympify(funcion_g)

    crit_pts = np.array(solve([dx, dy], (x, y))).astype(float)
    
    # Asegúrate de que solo generes el texto de resultados una vez
    text = jacobian_text(Matrix([dx, dy]), Matrix([x, y]), crit_pts, [x, y])

    # Aquí podrías definir el rango y los valores para la visualización del campo vectorial
    a, b = -10, 10  # Rango de ejemplo
    n = 20          # Número de puntos para la cuadrícula
    scale = 1       # Escala para la visualización

    # Crear cuadrícula para el campo vectorial
    x_vals = np.linspace(a, b, n)
    y_vals = np.linspace(a, b, n)
    X_, Y_ = np.meshgrid(x_vals, y_vals)

    U = np.zeros_like(X_)
    V = np.zeros_like(Y_)

    for i in range(X_.shape[0]):
        for j in range(X_.shape[1]):
            U[i, j] = dx.subs({x: X_[i, j], y: Y_[i, j]})
            V[i, j] = dy.subs({x: X_[i, j], y: Y_[i, j]})

    N = np.hypot(U, V) + 1e-8
    U /= N
    V /= N

    fig = go.Figure()
    quiver = ff.create_quiver(X_, Y_, U, V, scale=scale, name="Campo Vectorial")
    fig.add_traces(quiver.data)

    fig.add_traces(go.Scatter(x=crit_pts[:, 0], y=crit_pts[:, 1], mode='markers',
                               name="Puntos Críticos", marker=dict(size=15)))

    fig.update_layout(
        title={'text': 'Jacobiano', 'x': 0.5, 'y': 0.92, 'xanchor': 'center'},
        xaxis_title='X(t)',
        yaxis_title='Y(t)',
        width=800,
        template='plotly_white',
        margin=dict(l=10, r=10, t=90, b=0),
        legend=dict(orientation='h', y=1.1)
    )

    return fig, text