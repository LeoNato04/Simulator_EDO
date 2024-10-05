from dash import html, dcc
import dash

dash.register_page(
    __name__,
    path='/',
    name='Welcome'
)

layout = html.Div(className='welcome-page', children=[
    html.H1('Bienvenido a la Interfaz Gráfica'),
    html.P('Esta aplicación te permite explorar diferentes modelos matemáticos.'),
    dcc.Link('Ir a Crecimiento Poblacional', href='/Crecimiento_poblacional'),
    html.Br(),
    dcc.Link('Ir a Movimiento Armónico simple', href='/Movimiento_Armonico'),
    html.Br(),
    dcc.Link('Ir a Ley de Enfriamiento de Newton', href='/Ley_Enfriamiento'),
    html.Br(),

    html.Div(className='info-section', children=[
        html.P('Soy Leonel Fortunato Lizarbe Almeyda, estudiante de Computación científica.'),
        html.P('Esta página tiene el propósito de compartir mis conocimientos y facilitar el entendimiento de las EDOs.')
    ])
])