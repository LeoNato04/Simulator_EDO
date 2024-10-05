from dash import Dash, html, dcc
import dash

app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True
)

app.layout = html.Div(children=[
    html.Div(className='header', children=[
        html.Img(className='sm_logo', src='assets/imgs/FCM.png'),
        html.H1('INTERFAZ GRÁFICA', className='main_title')
    ]),
    
    html.Div(className='contenedor_navegacion', children=[
        dcc.Link(html.Button('Crecimiento Poblacional', className='boton edo_1'), href='/Crecimiento_poblacional'),
        dcc.Link(html.Button('Movimiento Armónico simple', className='boton edo_2'), href='/Movimiento_Armonico'),
        dcc.Link(html.Button('Ley de Enfriamiento de Newton', className='boton edo_3'), href='/Ley_Enfriamiento')
    ]),

    dcc.Link(html.Button('🏠', className='home-button'), href='/'), 

    dash.page_container
])

if __name__ == '__main__':
    app.run(debug=True, port=1254)