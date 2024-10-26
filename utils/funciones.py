# Librerias
import sympy as sp
import numpy as np 
import plotly.graph_objects as go 
import plotly.figure_factory as ff
from scipy.integrate import odeint


# Funciones

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
            'text': 'Modelo SIR con Cambio de Parámetros',
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

def calcular_jacobiano(functions):
    """
    Calcula el Jacobiano y los puntos críticos para una lista de funciones.
    """
    # Definir variables simbólicas
    num_vars = len(functions)
    vars = sp.symbols(f'x0:{num_vars}')
    
    # Convertir funciones de texto a expresiones simbólicas
    symbolic_functions = [sp.sympify(func) for func in functions]

    # Calcular Jacobiano
    jacobian_matrix = sp.Matrix([[sp.diff(f, var) for var in vars] for f in symbolic_functions])

    # Encontrar puntos críticos (cuando todas las derivadas parciales son 0)
    critical_points = sp.solve(symbolic_functions, vars)

    # Convertir Jacobiano a formato NumPy para mostrar en la gráfica
    jacobian_np = np.array(jacobian_matrix).astype(np.float64)

    # Crear gráfica con la matriz Jacobiana como tabla
    fig = go.Figure()
    fig.add_trace(
        go.Table(
            header=dict(values=["Jacobian"] + [f'x{i}' for i in range(num_vars)], align="center"),
            cells=dict(values=[["f" + str(i) for i in range(num_vars)]] + jacobian_np.T.tolist(), align="center")
        )
    )

    # Configuración de layout
    fig.update_layout(
        title="Jacobiano y Puntos Críticos",
        width=600,
        height=300,
        template='plotly_white'
    )

    # Generar texto para mostrar el Jacobiano y puntos críticos
    jacobian_text = f"Jacobian Matrix:\n{sp.pretty(jacobian_matrix)}"
    critical_points_text = f"Puntos Críticos: {critical_points}"

    return fig, jacobian_text, critical_points_text
