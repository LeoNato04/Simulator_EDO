###################################################################################
#
# Página 4: Modelo con Parámetro Cambiante
#
###################################################################################
import dash
from dash import dcc, html, Input, Output, callback
from utils import model_SIR  # Asegúrate de que la función esté en utils.py o el archivo correcto

dash.register_page(
    __name__,
    path='/Modelo_Parametro_Cambiante',
    name='Modelo_Parametro_Cambiante'
)

layout = html.Div(className='Pages', children=[
    html.Div(className='div_parametros', children=[
        html.H2('PARÁMETROS DEL MODELO CON PARÁMETRO CAMBIANTE'),
        html.Div(className='div_flex', children=[
            html.Div([
                html.H3('Valor Inicial de Susceptibles (S0)'),
                dcc.Input(type='number', value=990, id='S0')
            ]),
            html.Div([
                html.H3('Valor Inicial de Infectados (I0)'),
                dcc.Input(type='number', value=10, id='I0')
            ]),
            html.Div([
                html.H3('Valor Inicial de Recuperados (R0)'),
                dcc.Input(type='number', value=0, id='R0')
            ]),
        ]),
        html.Div(className='div_flex', children=[
            html.Div([
                html.H3('Tasa de Transmisión (Beta)'),
                dcc.Input(type='number', value=1.0, id='param_inicial')
            ]),
            html.Div([
                html.H3('Tasa de Recuperación (Gamma)'),
                dcc.Input(type='number', value=0.1, id='gamma')
            ]),
            html.Div([
                html.H3('Tiempo Final (s)'),
                dcc.Input(type='number', value=60, id='t_f_param_cambiante')
            ]),
        ]),
        html.Div(className='div_flex', children=[
            html.Div([
                html.H3('Tiempo de Cambio (s)'),
                dcc.Input(type='number', value=30, id='t_change')
            ]),
            html.Div([
                html.H3('Nueva Beta (Beta después del cambio)'),
                dcc.Input(type='number', value=0.5, id='new_beta')
            ]),
            html.Div([
                html.H3('Nueva Gamma (Gamma después del cambio)'),
                dcc.Input(type='number', value=0.2, id='new_gamma')
            ]),
        ]),
        html.H3('Número de Puntos para la Gráfica'),
        dcc.Slider(
            min=1, max=100, step=1, value=20, 
            marks=None, 
            tooltip={'placement': 'bottom', 'always_visible': True}, 
            id='cant_puntos_param_cambiante'
        )
    ]),
    html.Div(className='div_grafica', children=[
        html.H2('GRÁFICA DEL MODELO CON PARÁMETRO CAMBIANTE'),
        dcc.Loading(
            type='default',
            children=dcc.Graph(id='figura_4')
        )
    ])
])

@callback(
    Output('figura_4', 'figure'),
    Input('S0', 'value'),
    Input('I0', 'value'),
    Input('R0', 'value'),
    Input('param_inicial', 'value'),
    Input('gamma', 'value'),
    Input('t_f_param_cambiante', 'value'),
    Input('t_change', 'value'),
    Input('new_beta', 'value'),
    Input('new_gamma', 'value'),
    Input('cant_puntos_param_cambiante', 'value'),
)
def grafica_parametro_cambiante(S0, I0, R0, param_inicial, gamma, t_f_param_cambiante, t_change, new_beta, new_gamma, cant_puntos_param_cambiante):
    # Condiciones iniciales
    initial_conditions = [S0, I0, R0]
    
    # Generar la gráfica utilizando la función model_SIR
    fig = model_SIR(initial_conditions, param_inicial, gamma, t_f_param_cambiante, t_change, new_beta, new_gamma)
    
    return fig