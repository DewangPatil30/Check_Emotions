import dash
import dash_html_components as html
import dash_core_components as dbc


appClone = None

def takeApp(clone):
    global appClone
    appClone = clone

def create_header():
    header_layout=html.Div(className='hfeed site', id='page',children=[
        html.Header(className='header',id='masthead', children=[
            html.Div(className='header-div', children=[    
                    html.A(href='#dot',className='active',children="Let's Go"),
                    html.A(href='#revCheckDiv',children='Review Check'),
                    html.A(href='#wordCloudDiv',children='Word Cloud'),
                    html.A(href='#footSec2',children='Contact Me')
    ])
    ])
    ])
 
    return header_layout

def create_footer():

    footer_layout=html.Footer(className='site-footer', id='colophon', children=[

        html.Section([
            html.Div([
                # html.Img(src='data:image/png;base64,{}'.format(encoded_image), className='img'),
                html.P(['Lakshmi Narain College of Technology (LNCT), '
                , 'Kalchuri Nagar, Raisen Rd, Bhopal, Madhya Pradesh (MP)'
                ], className='clg')
            ], className='fDiv1'),

            html.Div([
                html.A(['Dewangpatil30081999@gamil.com'], 
                    href='mailto: Dewangpatil30081999@gamil.com', 
                    className='mail'),
                
                html.A(['HackerRank @dewangpatil30081'], 
                    href='https://www.hackerrank.com/dewangpatil30081?hr_r=1',
                    className='hackerrank', target='_blank'),
                html.A(['GitHub'], href='https://github.com/dewangpatil30', className='git', target='_blank'),
            ], className='fDiv2'),

            html.Div([
                html.H3(['Navigation'], className='nav'),
                
                html.A(href='#masthead',className='home',children="Home"),
                html.A(href='#revCheckDiv',children='Review Check'),
                html.A(href='#wordCloudDiv',children='Word Cloud'),
                html.A(href='#site-footer',children='Contact Me')
            ], className='fDiv3')
        ], className='footSec1'),

        html.Section([
            html.P(['Copyright © 2021 Check Emotions | Created by Dewang Patil'],
             className='copyRight')
        ], className='footSec2', id='footSec2')

    ])

    return footer_layout
