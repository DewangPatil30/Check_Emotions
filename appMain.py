
# Importing the libraries........

import pickle
from dash_html_components.Button import Button
from dash_html_components.Div import Div
from dash_html_components.Section import Section
from dash_html_components.Span import Span
from dash_html_components.Strong import Strong
import pandas as pd
import webbrowser
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express as px

from dash.dependencies import Input, Output, State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import header_footer as h
nltk.download('stopwords')

################################################################ GLOBAL VARIABLES ######################################################################################

project_name = None
bs = 'https://cdn.jsdelivr.net/npm/bootswatch@4.5.2/dist/slate/bootstrap.min.css'

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
h.takeApp(app)

df = pd.read_csv('predictions.csv')
df2 = pd.read_csv('ScrapedReviews.csv')
feats = df2.iloc[:,-1].sample(2000)

words = None

header =h.create_header()
footer = h.create_footer()


################################################################ GLOBAL VARIABLES ######################################################################################

######################################################################################################################################################

def create_app_ui():

    main_layout = html.Div(
    [
        html.Div([
            header,

            html.Div([
                html.Div([
                    html.H1(id='main_title', className='checkEmotions', children='Check Emotions'), 
                ], className='firstDiv'),
                
                html.Div([
                    html.H2(id='site_details', className='Site_details', children='Sentiment Analysis using Machine Learning and Deep Learning'),                        
                    ], className='secondDiv'),
            
            ], className='front_parent_div'),

        ],className='bg_image'),

        html.Section(children=[
            html.Div([

                html.P(['.'], id='dot', className='dot'),
                html.Div([
                    
                    dcc.Dropdown(
                        id='my_dropdown',
                        options=[
                                {'label': 'status', 'value': 'status'}
                        ],
                        value='status',
                        multi=False,
                        clearable=False,
                    ),

                    html.Div([

                            html.H1([
                                'Pie chart showing the percent of +ve and -ve reviews' 
                            ], id='chart_title', className="chart_title"),
                            
                            dcc.Graph(id='graph', className='graph')
                            
                        ], id='graph_div', className='graph_div'),

                        html.H2(['Review Stats Pie'], className='stats')
            
                    
            ], id='pie', className='pie'),

            html.Div([
                html.Div([
                    html.H1(id='pie_title', className='pie_title', children=['What does these graph tells us ?']),
                    ]),
                    html.Span(id='span'),
                    html.Div([
                    html.P(id='pie_Details', className='pie_details', 
                            children=['By the data given by the pie-chart made by the predictions of the review we find that 92.1 percent of the reviews from the site, ', 
                            html.Strong('i.e 9,62,148 reviews are Positive  '), 'rest remaining 20%, ' ,
                            html.Strong('i.e 2,42,148 reviews are Negative')
                            
                            ]),
                    ]),
                    html.H2(['Your site', 
                        html.A( children=[" Etsy.com "], href='https://www.etsy.com/', className='etsy', target='_blank'), 
                        'is doing great job satisfying customers !']
                    
                    , id='site_review', className='site_review'),


                        ], id='pie_info', className='pie_info'),

                ], id='pie_main', className='pie_main'),
            ], id='pie_extreme_main', className='pie_extreme_main'),

        
        html.Section([

            html.Div([
                html.H1(['Review Check'], id='revCheck')
            ], id='revCheckDiv'),

            html.Section([

                html.Div([

                    html.H1(['Check Review '], id='check_rev_heading1'),
                
                    html.P([html.Strong(['This feature allows you to test our '+ 
                        'model i.e Predictor if it is working fine or not. ' + 
                        'Just enter any review and click on submit now than our model ' +
                        'will tell the review is Positive or Negative.'])], id='rev_para1'),
                
                ], id='text_info1'),

                html.Div([
                    html.Div([
                        html.H3(['Review Status Below !!!'], id='review_status'),
                        dbc.Input(
                            id='textarea_review',
                            placeholder = 'Enter the review here....',
                            type = "text",
                            style = {'width':'100%', 'height':50}
                            ),

                        dbc.Button(['Check Review'], id='check_review', className='check_review', n_clicks=0)
                    ]),
                    html.H1(id='result', children=None), 
                
                ], id='text_area1', className='text_area1'),

            ], id='review_section1', className='review_section1'),

            html.Section([

                html.Div([
                    html.H3(['Choose Any Review !!!'], id='revDrop'),
                    dcc.Dropdown(
                        id='rev_drop',
                        options=[{'label': i, 'value': i} for i in feats],
                        value=None,
                        multi=False,
                        clearable=True,
                        style={"width": "100%"},
                        optionHeight = 150,
                        placeholder= 'Select Any Review From Drop...'
                    ),

                    dbc.Button(['Check Review'], id='check_drop', className='check_drop', n_clicks=0),
                    html.H1(id='result2', children=None)

                ], id='text_area2', className='text_area2'),
                
                html.Div([
                    
                    html.H1(['Review Dropdown'], id='check_rev_heading2', className='check_rev_heading2'),
                    
                    html.P([html.Strong(['This is another feature which allows you to test the Predictor model' + 
                        ' by choosing the review from the Dropdown menu.' +
                        ' Choose review and than click on check now.' +
                        ' You will see result of your Review.'])], id='rev_para2'),

                        ], id='text_info2', className='text_info2' ),
                    
            ], id='review_section2', className='review_section2'),

        ], id='review_sec', className='review_sec'),


        html.Div([
            html.H1(['Word Cloud'], id='wordCloud')
            ], id='wordCloudDiv'),

        html.Div([

            html.Div([
                html.H2(['20 Most Frequent Words '], className='cloud_head'),
                html.H2(['with Count'], className='cloud_head_part2'),

                html.Div([
                
                    html.Div([
                        html.H5([str(words[i][0])], className='word'), 
                        html.P([str(words[i][1])],  className='word_count')  
                    ], className='word_wraper') for i in range(20)

                ], className='words_container'),  


            ], className='left_sec'),
            html.Div([

                html.H2([
                    'Wondering how we got these words ?'
                ], className='cloud_info_heading'),

                html.P([
                    'To accomplish this task we took the help of the ',
                    html.A(['Natural Language Toolkit aka NLTK .'],href='https://www.nltk.org/',  className='nltk') , 
                    ' By the use of it, we removed all the unuseful words and kept useful words only. ' , 
                    'Later by the use of some simple functions like counter we extracted Top 20 words'
                ], className='cloud_info_para')

            ], className='cloud_info_div')

        ], className='cloud'),

        html.Div([

            html.Span(id='cloud_span1'),
            html.Div ([
                
                html.Div([
                    html.H1(['452300+'], className='count'),
                    html.P(['Words'], className='count_word'),
                ], className='count_wrap'),

                html.Div([
                    html.H1(['50000+'], className='count'),
                    html.P(['Usefull Words'], className='count_word'),
                ], className='count_wrap'),

                html.Div([
                    html.H1(['20'], className='count'),
                    html.P(['Words shown here'], className='count_word'),
                ], className='count_wrap'),
            ], className='count_external_wrapper'),
            html.Span(id='cloud_span2')
        ], className='counter'),


        footer
    
    ], className= 'main_div')
    
    
    return main_layout

###################################################################### APP UI ENDS HERE ################################################################################


################################################################### FUNCTIONS DEFINED HERE ##################################################################################################################

# Defining My Functions
def load_model():
    global scrappedReviews
    scrappedReviews = pd.read_csv('balanced_reviews.csv')
  
    global pickle_model
    file = open("model_pickle.pkl", 'rb') 
    pickle_model = pickle.load(file)

    global vocab
    file = open("vocab_pickle.pkl", 'rb') 
    vocab = pickle.load(file)
    


def check_review(reviewText):

    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    
    return pickle_model.predict(vectorised_review)



def open_browser():
    webbrowser.open_new('http://192.168.182.56:8050/ ')



@app.callback(
    Output(component_id='graph', component_property='figure'),
    [Input(component_id='my_dropdown', component_property='value')]
)

def pie_chart(my_dropdown):
    
    
    dff = df
    piechart = px.pie(
                data_frame=dff,
                names=my_dropdown,
                color_discrete_sequence=['darkorchid', 'black'],

               )
               
    '''
    piechart.update_traces(textposition='outside', 
        textinfo='percent+label', 
        marker=dict(line=dict(color='#00000', width=4)),
        pull=[0.2, 0],
        opacity = 0.7
    )
    
    '''
    return (piechart)
        
    
@app.callback(
    Output('result2',  'children' ),
    [
    Input('check_drop', 'n_clicks')
    ],
    [State('rev_drop', 'value')]
    )
def update_app_ui2(dropN, rev_drop):
    print('Data Type of ', str(type(rev_drop)))
    print('Value = ', str(rev_drop) )

    response = check_review(rev_drop)
    print('response = ', response)

    if dropN > 0 :
        if (response[0] == 0):
            result1 = 'Sorry! The review is Negative'
        elif  (response[0] == 1):
            result1 = 'Hurray! The review is Positive'
        else:
            result1 = 'Unknown'
   
    if rev_drop:    
        return result1
    else:
        return None


@app.callback(
    Output('result',  'children' ),
    [Input('check_review', 'n_clicks')], 
    [State('textarea_review', 'value')]
    )
def update_app_ui(n, textarea_value):
    print('Data Type of ', str(type(textarea_value)))
    print('Value = ', str(textarea_value) )

    response = check_review (textarea_value)
    print('response = ', response)

    if n > 0:
        if (response[0] == 0):
            result1 = 'Sorry! The review is Negative'
        elif  (response[0] == 1):
            result1 = 'Hurray! The review is Positive'
        else:
            result1 = 'Unknown'
        
    return result1


# MOST UESD WORDS FUNCTIONS:-

def get_top_n_words(corpus):
        
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items() if word not in stopwords.words('english')]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True) 
    
    return words_freq[:21]

################################################################### FUNCTIONS ENDS HERE ##################################################################################################################

################################################################### MAIN FUNCTION HERE ##################################################################################################################


# Main Function to control the Flow of your Project
def main():
    print("Start of my project")
    
    load_model()    
    global words
    global df2
    
    words = get_top_n_words(df2["Review"])
        
    global project_name
    global scrappedReviews
    global app
    
    
    project_name = "Sentiment Analysis with Insights"
    
    app.title = project_name
    app.layout = create_app_ui()
    open_browser()
    app.run_server(host='0.0.0.0', port=8050)

    
    print("End of my project")
    project_name = None
    scrappedReviews = None
    app = None


################################################################### MAIN FUNCTIONS CALLING ##################################################################################################################


# Calling the main function 
if __name__ == '__main__':
    main()
    
    
################################################################### END OF APP ##################################################################################################################

