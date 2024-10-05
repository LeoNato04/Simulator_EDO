###################################################################################
#
# Librerías
#
###################################################################################
import dash
from dash import dcc, html, Input, Output, callback
from utils import ecuacion_armonica

dash.register_page(
    __name__,
    path='/Movimiento_Armonico',
    name='Movimiento_Armonico'
)

###################################################################################
#
# Layout HTML
#
###################################################################################

layout = html.Div(className='Pages', children=[

    html.Div(className='div_parametros', children=[

        html.H2('PARÁMETROS ECUACIÓN ARMÓNICA SIMPLE'),

        html.Div(className='div_flex', children=[
            html.Div([
                html.H3('Condición Inicial: Desplazamiento (x₀)'),
                dcc.Input(type='number', value=10, id='despl_ini')
            ]),
            html.Div([
                html.H3('Condición Inicial: Velocidad (v₀)'),
                dcc.Input(type='number', value=10, id='vel_ini')
            ]),
            html.Div([
                html.H3('Tiempo Final'),
                dcc.Input(type='number', value=50, id='time_fin_2')
            ]),
        ]),

        html.H3('Frecuencia Natural (ω)'),
        dcc.Input(type='number', value=1, id='omega'),

        html.H3('Amortiguamiento (β)'),
        dcc.Input(type='number', value=0.1, id='beta'),

    ]),

    html.Div(className='div_grafica', children=[
        html.H2('GRÁFICA DE LA EDO DE 2º ORDEN'),
        dcc.Loading(
            type='default',
            children=dcc.Graph(id='figura_2')
        )
    ])

])

###################################################################################
#
# Callback
#
###################################################################################

@callback(
    Output('figura_2', 'figure'),
    Input('despl_ini', 'value'),
    Input('vel_ini', 'value'),
    Input('time_fin_2', 'value'),
    Input('omega', 'value'),
    Input('beta', 'value'),
)
def grafica_edo2(x0, v0, t_f, omega, beta):
    cant = 100
    fig = ecuacion_armonica(x0, v0, omega, beta, t_f, cant)
    return fig