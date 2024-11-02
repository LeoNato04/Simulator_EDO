from dash import html, dcc
import dash

dash.register_page(
    __name__,
    path='/',
    name='Welcome'
)

layout = html.Div(className='welcome-page', children=[

    # Título de bienvenida
    html.H1('Bienvenido a la Interfaz Gráfica'),

    # Descripción de la aplicación
    html.P('Esta aplicación te permite explorar diferentes modelos matemáticos y sus aplicaciones.'),

    # Enlaces a las diferentes páginas de modelos
    dcc.Link('Ir a Crecimiento Poblacional', href='/Crecimiento_poblacional'),
    html.Br(),
    dcc.Link('Ir a Movimiento Armónico Simple', href='/Movimiento_Armonico'),
    html.Br(),
    dcc.Link('Ir a Ley de Enfriamiento de Newton', href='/Ley_Enfriamiento'),
    html.Br(),
    dcc.Link('Ir a Modelo con Parámetro Cambiante', href='/Modelo_Parametro_Cambiante'),
    html.Br(),
    dcc.Link('Ir a Cálculo del Jacobiano y Puntos Críticos', href='/Jacobiano_Puntos_Criticos'),
    html.Br(),

    # Información del autor y propósito de la página
    html.Div(className='info-section', children=[
        html.P('Soy Leonel Fortunato Lizarbe Almeyda, estudiante de Computación Científica.'),
        html.P('Esta página tiene el propósito de compartir mis conocimientos y facilitar el entendimiento de las Ecuaciones Diferenciales Ordinarias (EDOs) y sus aplicaciones en modelamiento matemático.')
    ])
])
