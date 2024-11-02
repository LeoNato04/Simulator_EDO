import dash
from dash import Dash, html, dcc

# Inicializar la aplicación Dash
app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True
)

# Definir el layout de la aplicación
app.layout = html.Div(children=[

    # Encabezado con logotipo y título
    html.Div(className='header', children=[
        html.Img(className='sm_logo', src='assets/imgs/FCM.png'),
        html.H1('INTERFAZ GRÁFICA', className='main_title')
    ]),
    
    # Contenedor de navegación con botones para cada modelo
    html.Div(className='contenedor_navegacion', children=[
        dcc.Link(html.Button('Crecimiento Poblacional', className='boton'), href='/Crecimiento_poblacional'),
        dcc.Link(html.Button('Movimiento Armónico Simple', className='boton'), href='/Movimiento_Armonico'),
        dcc.Link(html.Button('Ley de Enfriamiento y Calentamiento', className='boton'), href='/Ley_Enfriamiento'),
        dcc.Link(html.Button('Modelo con Parámetro Cambiante', className='boton'), href='/Modelo_Parametro_Cambiante'),
        dcc.Link(html.Button('Jacobiano y Puntos Críticos', className='boton'), href='/Jacobiano_Puntos_Criticos')
    ]),

    # Botón de navegación para regresar a la página de bienvenida
    dcc.Link(html.Button('🏠', className='home-button'), href='/'), 

    # Contenedor para el contenido de las páginas
    dash.page_container
])

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True, port=1254)