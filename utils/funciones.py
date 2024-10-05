# Librerias
import numpy as np 
import plotly.graph_objects as go # Grafica
import plotly.figure_factory as ff # mallado de vectores


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