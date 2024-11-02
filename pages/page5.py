###################################################################################
#
# Página 5: Jacobiano
#
###################################################################################
import dash
from dash import dcc, html, Input, Output, callback
from utils import calcular_jacobiano
from sympy import sympify, SympifyError

DEBUG_MODE = False

# Registrar la página con ruta y nombre específico para el Jacobiano
dash.register_page(
    __name__,
    path='/Jacobiano_Puntos_Criticos',
    name='Puntos Críticos - Jacobiano'
)

###################################################################################
#
# Layout HTML
#
###################################################################################

# Estructura del layout con clases CSS para el diseño y estructura
layout = html.Div(className='Pages', children=[
    html.Div(className='content', children=[
        
        html.Div(className='div_parametros', children=[
            html.H2('PARÁMETROS PARA CALCULAR EL JACOBIANO'),
            html.Div(className='div_flex', children=[

                html.Div(className='input_div', children=[  # Entrada para función F
                    html.H3('Función F(x, y)'),
                    dcc.Input(type='text', id='funcion_f', value='(x+0)*(y+3)', style={'width': '100%'})  
                ], style={'margin-bottom': '30px'}),

                html.Div(className='input_div', children=[  # Entrada para función G
                    html.H3('Función G(x, y)'),
                    dcc.Input(type='text', id='funcion_g', value='(y+0)*(x+3)', style={'width': '100%'}),
                ]),
            ]),
        ]),

        # Div para mostrar los resultados calculados
        html.Div(className='div_resultados', children=[
            html.H2('RESULTADOS', style={'text-align': 'center'}),
            html.Div(id='jacobiano_texto', style={'margin': '10px 0'}),
            html.Div(id='puntos_criticos_texto', style={'margin': '10px 0'})
        ]),
    ], style={'flex': '1'}),

    # Sección para la gráfica
    html.Div(className='div_grafica', children=[
        html.H2('CAMPO VECTORIAL - JACOBIANO'),
        
        html.Div(className='grafica', children=[
            dcc.Loading(type='default', children=dcc.Graph(id='jacobiano_figura'))
        ])
    ], style={'flex': '2'})
])

###################################################################################
#
# Callback principal
#
###################################################################################

@callback(
    Output('jacobiano_figura', 'figure'),
    Output('jacobiano_texto', 'children'),
    Output('puntos_criticos_texto', 'children'),
    Input('funcion_f', 'value'),
    Input('funcion_g', 'value')
)
def update_jacobiano(funcion_f, funcion_g):
    # Verificar si las expresiones son válidas
    try:
        print(f"Function F: {funcion_f}, Function G: {funcion_g}")  # Para depuración
        fig, jacobiano_text = calcular_jacobiano(funcion_f, funcion_g)

        # Configurar los ejes para mostrar
        fig.update_layout(
            xaxis=dict(title='Eje X'),
            yaxis=dict(title='Eje Y') 
        )
        return fig, jacobiano_text, jacobiano_text  
    
    except SympifyError:
        return {}, "Error de sintaxis en la expresión ingresada.", "Error de sintaxis en la expresión ingresada."
    except Exception as e:
        print(f"Error: {e}")  # Para depuración
        return {}, "Se produjo un error al calcular el Jacobiano.", "Se produjo un error al calcular el Jacobiano."