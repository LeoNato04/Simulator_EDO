###################################################################################
#
# Librerías
#
###################################################################################
import dash
from dash import dcc, html, Input, Output, callback
from utils import ley_enfriamiento_newton 

dash.register_page(
    __name__,
    path='/Ley_Enfriamiento',
    name='Ley_Enfriamiento'
)

###################################################################################
#
# Layout HTML
#
###################################################################################

layout = html.Div(className='Pages', children=[

    html.Div(className='div_parametros', children=[

        html.H2('PARÁMETROS DE ENFRIAMIENTO'),

        html.Div(className='div_flex', children=[
            html.Div([
                html.H3('Temperatura Inicial (°C)'),
                dcc.Input(type='number', value=90, id='T0')
            ]),
            html.Div([
                html.H3('Temperatura Ambiente (°C)'),
                dcc.Input(type='number', value=20, id='T_amb')
            ]),
            html.Div([
                html.H3('Constante de Enfriamiento (h)'),
                dcc.Input(type='number', value=0.1, id='h')
            ]),
        ]),

        html.H3('Tiempo Final (s)'),
        dcc.Input(type='number', value=60, id='t_f'),

        html.H3('Número de Puntos para la Gráfica'),
        dcc.Slider(min=1, max=100, step=1, value=20, marks=None, tooltip={'placement': 'bottom', 'always_visible': True}, id='cant_puntos')
    ]),

    html.Div(className='div_grafica', children=[
        html.H2('GRÁFICA DE LA LEY DE ENFRIAMIENTO'),
        dcc.Loading(
            type='default',
            children=dcc.Graph(id='figura_3')
        )
    ])
])

###################################################################################
#
# Callback
#
###################################################################################

@callback(
    Output('figura_3', 'figure'),
    Input('T0', 'value'),
    Input('T_amb', 'value'),
    Input('h', 'value'),
    Input('t_f', 'value'),
    Input('cant_puntos', 'value'),
)

def grafica_enfriamiento(T0, T_amb, h, t_f, cant_puntos):
    fig = ley_enfriamiento_newton(T0, T_amb, h, t_f, cant_puntos)
    return fig