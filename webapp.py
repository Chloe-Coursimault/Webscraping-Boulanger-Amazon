#-- Imports
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import dash_table
from dash.dependencies import Input, Output,State
import dash_bootstrap_components as dbc
import pandas as pd
from io import BytesIO
import base64
import tomotopy as tp
#import nltk
from nltk.stem import WordNetLemmatizer
from nltk.metrics import *
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
#nltk.download('punkt')
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import swifter

#-- Creation of some colors variables used in the webapp
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

#-- Definition of the general theme of the webapp
app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY])

#-- Import data
df=pd.read_csv("clean_data_boulanger.csv")
amazon=pd.read_csv("details.csv")

def wocl(data):
    stopwords_fr=set(["a","à","â","abord","afin","ah","ai","aie","ainsi","allaient","allo","allô","allons","après","assez","attendu","au","aucun","aucune","aujourd","aujourd'hui","auquel","aura","auront","aussi","autre","autres","aux","auxquelles","auxquels","avaient","avais","avait","avant","avec","avoir","ayant","b","bah","beaucoup","bien","bigre","boum","bravo","brrr","c","ça","car","ce","ceci","cela","celle","celle-ci","celle-là","celles","celles-ci","celles-là","celui","celui-ci","celui-là","cent","cependant","certain","certaine","certaines","certains","certes","ces","cet","cette","ceux","ceux-ci","ceux-là","chacun","chaque","cher","chère","chères","chers","chez","chiche","chut","ci","cinq","cinquantaine","cinquante","cinquantième","cinquième","clac","clic","combien","comme","comment","compris","concernant","contre","couic","crac","d","da","dans","de","debout","dedans","dehors","delà","depuis","derrière","des","dès","désormais","desquelles","desquels","dessous","dessus","deux","deuxième","deuxièmement","devant","devers","devra","différent","différente","différentes","différents","dire","divers","diverse","diverses","dix","dix-huit","dixième","dix-neuf","dix-sept","doit","doivent","donc","dont","douze","douzième","dring","du","duquel","durant","e","effet","eh","elle","elle-même","elles","elles-mêmes","en","encore","entre","envers","environ","es","ès","est","et","etant","étaient","étais","était","étant","etc","été","etre","être","eu","euh","eux","eux-mêmes","excepté","f","façon","fais","faisaient","faisant","fait","feront","fi","flac","floc","font","g","gens","h","ha","hé","hein","hélas","hem","hep","hi","ho","holà","hop","hormis","hors","hou","houp","hue","hui","huit","huitième","hum","hurrah","i","il","ils","importe","j","je","jusqu","jusque","k","l","la","là","laquelle","las","le","lequel","les","lès","lesquelles","lesquels","leur","leurs","longtemps","lorsque","lui","lui-même","m","ma","maint","mais","malgré","me","même","mêmes","merci","mes","mien","mienne","miennes","miens","mille","mince","moi","moi-même","moins","mon","moyennant","n","na","ne","néanmoins","neuf","neuvième","ni","nombreuses","nombreux","non","nos","notre","nôtre","nôtres","nous","nous-mêmes","nul","o","o|","ô","oh","ohé","olé","ollé","on","ont","onze","onzième","ore","ou","où","ouf","ouias","oust","ouste","outre","p","paf","pan","par","parmi","partant","particulier","particulière","particulièrement","pas","passé","pendant","personne","peu","peut","peuvent","peux","pff","pfft","pfut","pif","plein","plouf","plus","plusieurs","plutôt","pouah","pour","pourquoi","premier","première","premièrement","près","proche","psitt","puisque","q","qu","quand","quant","quanta","quant-à-soi","quarante","quatorze","quatre","quatre-vingt","quatrième","quatrièmement","que","quel","quelconque","quelle","quelles","quelque","quelques","quelqu'un","quels","qui","quiconque","quinze","quoi","quoique","r","revoici","revoilà","rien","s","sa","sacrebleu","sans","sapristi","sauf","se","seize","selon","sept","septième","sera","seront","ses","si","sien","sienne","siennes","siens","sinon","six","sixième","soi","soi-même","soit","soixante","son","sont","sous","stop","suis","suivant","sur","surtout","t","ta","tac","tant","te","té","tel","telle","tellement","telles","tels","tenant","tes","tic","tien","tienne","tiennes","tiens","toc","toi","toi-même","ton","touchant","toujours","tous","tout","toute","toutes","treize","trente","très","trois","troisième","troisièmement","trop","tsoin","tsouin","tu","u","un","une","unes","uns","v","va","vais","vas","vé","vers","via","vif","vifs","vingt","vivat","vive","vives","vlan","voici","voilà","vont","vos","votre","vôtre","vôtres","vous","vous-mêmes","vu","w","x","y","z","zut","alors","aucuns","bon","devrait","dos","droite","début","essai","faites","fois","force","haut","ici","juste","maintenant","mine","mot","nommés","nouveaux","parce","parole","personnes","pièce","plupart","seulement","soyez","sujet","tandis","valeur","voie","voient","état","étions"])
    stopwords_fr.update(["qu'elle","bref","dit","meme","puis","pris","disant","euro","euros","c'est","j'ai","ca","n'ai","bonjour","faire","mot","passe","m'a","tre","l'on","déjà","n'est","d'un","tres","vien","d'une","n'a","vraiment","faut","viens","pourtant","également","j'avais","auprès","qu'il","lors","avez","d'autre","qu'on","n'y","pu","mi","d'avoir","qu'ils"])
    stopwords_fr.update(["pc","l'appareil","appareil", "produit","ordinateur","portable","tr","pr","re"])

    wc_tous_avis=WordCloud(stopwords=stopwords_fr, background_color="white", width=600, height=400).generate(data.lower())
    return wc_tous_avis.to_image()

#-- CSS style for the three first dropdowns
dropdown_style = {"margin-left": "20px",
                  'display': "inline-block",
                  "width": "200px",
                  "margin-top": "10px",
                  'color':'black',
                  'font-family':'arial black'
                  }

#-- CSS style for the two graphs
style_graph={"margin-left": "20px","margin-right": "20px","margin-top":"5px"}

img_style={"margin":"20px"}

#-- Create the layout, containing all elements that will be print on the page
app.layout = html.Div(children=[
    
    #-- General title of the Webapp, on top of the page
    html.H1(children='Webapp for PC analysis',
            style={'textAlign': 'center','font-family':'arial black'}),

    #-- Line to make a separation between two parts
    html.Hr(),
    html.Hr(),
    
    #-- Div containing the three first dropdowns and the clear button
    html.Div([
    
        #-- Dropdown to select which brand the user wants to display
        dcc.Dropdown(id="brand-filter", 
                    options=[{"label": col, "value": col} for col in df["brand"].unique()],
                    style=dropdown_style,placeholder="Brand"),
        
        #-- Dropdown to select which RAM value the user wants to display
        dcc.Dropdown(id="ram-filter", 
                    options=[{"label": col, "value": col} for col in sorted(df["RAM"].unique())],
                    style=dropdown_style,placeholder="RAM"),
        
        #-- Dropdown to select which OS the user wants to display
        dcc.Dropdown(id="os-filter", 
                    options=[{"label": col, "value": col} for col in sorted(df["OS"].unique())],
                    style=dropdown_style,placeholder="OS"),
        
        #-- Button proposing to clear all filters selection
        html.Button('Clear Selection', id='clear-button',
                    style={
                        "margin-left": "100px",
                        'display': "inline-block",
                        "width": "200px",
                        'color':'black',
                        'font-family':'arial black'
                        }),
    
    ],style=dict(display="inline-block")),
    
    
    #-- Title for the slider
    html.P(children="Taille de l'écran : ", style={"margin-top":"15px",
                                                    "margin-left": "20px",
                                                    "margin-bottom":"20px",
                                                    'font-family':'arial black'}),
    
    #-- Range slider to choose range of values for the column "taille_ecran"
    dcc.RangeSlider(df["taille_ecran"].min(), df["taille_ecran"].max(), 
                    value=[df["taille_ecran"].min(), df["taille_ecran"].max()], id='taille_slider'),

    #-- Line to make a separation between two parts
    html.Hr(),

    html.H3(children="Tous les ordinateurs vendus chez Boulanger correspondant à votre recherche",
            style={
                "margin-top":"15px",
                "margin-left": "20px",
                "margin-bottom":"20px",
                'font-family':'arial black'
            }),
    
    
    #-- Table to show all products coresponding to filters of the user
    dash_table.DataTable(
            id='grouped-data',
            columns=[{'name': i, 'id': i} for i in sorted(df.columns)],
            page_current=0,
            page_size=5,
            page_action='native',
            sort_action='native',
            style_table={'overflowX':'scroll','maxHeight':'300px',"margin-left":"10px","margin-top":"20px"},
            style_header={'backgroundColor':'rgb(30, 30, 30)','color':"white"},
            style_cell={'backgroundColor':'rgb(50,50,50)','color':'white'},
            sort_by=[]),
    
    
    #-- Title of the second table
    html.H3(children="Moyennes sur votre sélection en fonction d'un attribut", 
            style={
                "margin-top":"15px",
                "margin-left": "20px",
                "margin-bottom":"20px",
                'font-family':'arial black'
                }),
    
    
    #-- Dropdown to select grouping column for the next table
    dcc.Dropdown(
        id='grouping-column',
        options=[{'label': c, 'value': c} for c in df.columns.difference(["price","product_type"])],
        style={"color":"black","margin-top":"20px",
               "margin-bottom":"10px",
               "width":"500px",
               "margin-left":"20px",
               'font-family':'arial black'},
        value='brand',
        clearable=False
    ),
    
    #-- Second table, showing means of attributes, grouped by a choosen column
    dash_table.DataTable(
            id='stats-data',
            page_current=0,
            page_size=5,
            page_action='native',
            sort_action='native',
            style_table={'overflowX':'scroll','maxHeight':'300px',"margin-left":"10px","margin-top":"20px"},
            style_header={'backgroundColor':'rgb(30, 30, 30)','color':"white"},
            style_cell={'backgroundColor':'rgb(50,50,50)','color':'white'},
            sort_by=[]),
    
    #-- Line to make a separation between two parts
    html.Hr(),
    html.Hr(),
    
    #-- Title of the section
    html.H3(children='Vue globale du marché',style={'textAlign': 'center','font-family':'arial black'}),
    
    #-- Dropdown to select x-axis on the next two graphs
    dcc.Dropdown(
        id='column-selector',
        #-- We remove price and product_type from the possible selection
        options=[{'label': c, 'value': c} for c in df.columns.difference(["price","product_type"])],
        style={"color":"black",
               "margin-top":"20px",
               "margin-bottom":"10px",
               "width":"500px",
               "margin-left":"20px",
               'font-family':'arial black'},
        value='brand',
        clearable=False
    ),
    
    #-- Graph showing price means, by values on a choosen column
    dcc.Graph(id='graph-price',config={'displayModeBar': False},style=style_graph),
    
    #-- Graph showing product counts, by values on a choosen column
    dcc.Graph(id='graph-countplot',config={'displayModeBar': False},style=style_graph),

    #-- Line to make a separation between two parts
    html.Hr(),
    html.Hr(),

    html.H3(children='Analyse des avis et descriptions Amazon pour des produits ressemblants',style={'textAlign': 'center','font-family':'arial black'}),

    html.Div([

        html.Div([
            #-- Title of the section
            html.H4(children='Mots fréquents dans les descriptions',style={'textAlign': 'center','font-family':'arial black'}),

            #-- Wordcloud for reviews corresponding to selection
            html.Img(id='description-wordcloud', style=img_style)

        ], style={"display":"inline-block", "margin":"20px"}),

        html.Div([
            #-- Title of the section
            html.H4(children='Mots fréquents dans les avis',style={'textAlign': 'center','font-family':'arial black'}),

            #-- Wordcloud for reviews corresponding to selection
            html.Img(id='reviews-wordcloud', style=img_style)

        ], style={"display":"inline-block", "margin":"20px"}),
    
    ]),

    #-- Line to make a separation between two parts
    html.Hr(),

    html.H3(children='Thèmes ressortant dans les avis Amazon',style={'textAlign': 'center','font-family':'arial black'}),

    html.Div(id='lda'),

])

#----------------------------------------------------------------------------------------------------------

# Create callbacks

#-- Callback to update rows in the first table, based on three first dropdowns and sliders
@app.callback(
    Output("grouped-data", "data"),
    [
     Input("brand-filter", "value"),
     Input("ram-filter", "value"),
     Input("os-filter", "value"),
     Input("taille_slider","value")
    ]
)
def update_table(brand,RAM,os,taille):
    dff=df.copy()
    if(brand):
        dff=dff[dff["brand"]==brand]
    if(RAM):
        dff=dff[dff["RAM"]==RAM]
    if(os):
        dff=dff[dff["OS"]==os]
    if(taille):
        dff=dff[dff["taille_ecran"]>=taille[0]]
        dff=dff[dff["taille_ecran"]<=taille[1]]
    return dff.to_dict('records')


#-- Callback to clear selection of first three dropdowns and slider based on a click of the clear-button
@app.callback([
    Output('brand-filter', 'value'),
    Output('ram-filter', 'value'),
    Output('os-filter', 'value'),
    Output("taille_slider","value")
    ],
    [Input('clear-button', 'n_clicks')]
)
def clear_dropdown(n_clicks):
    return None,None,None,[df["taille_ecran"].min(), df["taille_ecran"].max()]


#-- Callback to update rows of second table (showing means), grouped by grouping column and based on products corresponding
#-- to filters choosen by the user
@app.callback([
    Output("stats-data", "data"),
    Output("stats-data","columns")],
    [
     Input("grouping-column",'value'),
     Input("grouped-data", "data"),
    ]
)
def update_table_stats(grouping_column,rows):
    df=pd.DataFrame.from_dict(rows)
    if(grouping_column in df.columns):
        dff=df.groupby(grouping_column,as_index=False).mean().round(2)
    else :
        dff=pd.DataFrame()
    cols = [{"name": i, "id": i} for i in dff.columns]
    return dff.to_dict('records'),cols


#-- Callback to update the first graph, based on the dropdown value choosen by the user
@app.callback(
    Output(component_id='graph-price', component_property='figure'),
    [Input(component_id='column-selector', component_property='value')]
)
def update_graph(column_name):
    dff=df.groupby(column_name,as_index=False).mean()
    return {
        'data': [{'x': dff[column_name], 'y': dff['price'], 'type': 'bar', 'name': column_name}],
        'layout': {'title': f' Prix moyen en fonction de {column_name}',
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {'color': colors['text']}
                }
    }
    
#-- Callback to update the second graph, based on the dropdown value choosen by the user 
@app.callback(
    Output(component_id='graph-countplot', component_property='figure'),
    [Input(component_id='column-selector', component_property='value')]
)
def update_graph2(column_name):
    dff=df.groupby(column_name,as_index=False).count()
    return {
        'data': [{'x': dff[column_name], 'y': dff['price'], 'type': 'bar', 'name': column_name}],
        'layout': {'title': f' Nombre de produits en fonction de {column_name}',
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {'color': colors['text']}
                }
    }


@app.callback(Output('reviews-wordcloud', 'src'), [ Input("brand-filter", "value"), Input("ram-filter", "value"), Input("os-filter", "value"), Input("taille_slider","value")])
def make_image(brand, ram, os, taille):

    amazon_selection=amazon.copy()

    if(brand):
        amazon_selection=amazon_selection[amazon_selection.Marque==brand]
    
    if(ram):
        amazon_selection=amazon_selection[amazon_selection["Taille de la mémoire RAM installée"]==ram]
    
    if(os):
        amazon_selection=amazon_selection[amazon_selection["Système d'exploitation"]==os]

    if(taille):
        amazon_selection=amazon_selection[amazon_selection["Taille de l'écran"]>=taille[0]]
        amazon_selection=amazon_selection[amazon_selection["Taille de l'écran"]<=taille[1]]
    
    all_reviews=" ".join(str(review) for review in amazon_selection.reviews)

    if all_reviews=="":
        all_reviews="sélection insoluble"

    img = BytesIO()
    wocl(data=all_reviews).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

@app.callback(Output('description-wordcloud', 'src'), 
            [ 
                Input("brand-filter", "value"), 
                Input("ram-filter", "value"), 
                Input("os-filter", "value"), 
                Input("taille_slider","value")
            ])
def make_image(brand, ram, os, taille):

    amazon_selection=amazon.copy()

    if(brand):
        amazon_selection=amazon_selection[amazon_selection.Marque==brand]
    
    if(ram):
        amazon_selection=amazon_selection[amazon_selection["Taille de la mémoire RAM installée"]==ram]
    
    if(os):
        amazon_selection=amazon_selection[amazon_selection["Système d'exploitation"]==os]

    if(taille):
        amazon_selection=amazon_selection[amazon_selection["Taille de l'écran"]>=taille[0]]
        amazon_selection=amazon_selection[amazon_selection["Taille de l'écran"]<=taille[1]]
    
    all_reviews=" ".join(str(review) for review in amazon_selection.description)

    if all_reviews=="":
        all_reviews="sélection insoluble"

    img = BytesIO()
    wocl(data=all_reviews).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

@app.callback(
    Output("lda", component_property='children'),
    [
     Input("brand-filter", "value"),
     Input("ram-filter", "value"),
     Input("os-filter", "value"),
     Input("taille_slider","value")
    ]
)
def make_lda(brand, ram, os, taille):
    amazon_selection=amazon.copy()

    if(brand):
        amazon_selection=amazon_selection[amazon_selection.Marque==brand]
    
    if(ram):
        amazon_selection=amazon_selection[amazon_selection["Taille de la mémoire RAM installée"]==ram]
    
    if(os):
        amazon_selection=amazon_selection[amazon_selection["Système d'exploitation"]==os]

    if(taille):
        amazon_selection=amazon_selection[amazon_selection["Taille de l'écran"]>=taille[0]]
        amazon_selection=amazon_selection[amazon_selection["Taille de l'écran"]<=taille[1]]
    
    if amazon_selection.empty:
        return "Nous n'avons pas de produit sur Amazon correspondant à votre recherche, essayez-en une autre !"
    
    else:
    
        amazon_selection.reviews=amazon_selection.reviews.swifter.apply(lambda x: str(x).lower().replace('\t',' ').replace('\n',' ').replace('\u200b',' ').replace('\r',' ').replace('[',' ').replace(']',' '))
        
        tokenizer=RegexpTokenizer(r'[a-zA-Zàéèù]+')
        amazon_selection.reviews=amazon_selection.reviews.swifter.apply(lambda x: tokenizer.tokenize(x))

        stopwords_fr=set(["a","à","â","abord","afin","ah","ai","aie","ainsi","allaient","allo","allô","allons","après","assez","attendu","au","aucun","aucune","aujourd","aujourd'hui","auquel","aura","auront","aussi","autre","autres","aux","auxquelles","auxquels","avaient","avais","avait","avant","avec","avoir","ayant","b","bah","beaucoup","bien","bigre","boum","bravo","brrr","c","ça","car","ce","ceci","cela","celle","celle-ci","celle-là","celles","celles-ci","celles-là","celui","celui-ci","celui-là","cent","cependant","certain","certaine","certaines","certains","certes","ces","cet","cette","ceux","ceux-ci","ceux-là","chacun","chaque","cher","chère","chères","chers","chez","chiche","chut","ci","cinq","cinquantaine","cinquante","cinquantième","cinquième","clac","clic","combien","comme","comment","compris","concernant","contre","couic","crac","d","da","dans","de","debout","dedans","dehors","delà","depuis","derrière","des","dès","désormais","desquelles","desquels","dessous","dessus","deux","deuxième","deuxièmement","devant","devers","devra","différent","différente","différentes","différents","dire","divers","diverse","diverses","dix","dix-huit","dixième","dix-neuf","dix-sept","doit","doivent","donc","dont","douze","douzième","dring","du","duquel","durant","e","effet","eh","elle","elle-même","elles","elles-mêmes","en","encore","entre","envers","environ","es","ès","est","et","etant","étaient","étais","était","étant","etc","été","etre","être","eu","euh","eux","eux-mêmes","excepté","f","façon","fais","faisaient","faisant","fait","feront","fi","flac","floc","font","g","gens","h","ha","hé","hein","hélas","hem","hep","hi","ho","holà","hop","hormis","hors","hou","houp","hue","hui","huit","huitième","hum","hurrah","i","il","ils","importe","j","je","jusqu","jusque","k","l","la","là","laquelle","las","le","lequel","les","lès","lesquelles","lesquels","leur","leurs","longtemps","lorsque","lui","lui-même","m","ma","maint","mais","malgré","me","même","mêmes","merci","mes","mien","mienne","miennes","miens","mille","mince","moi","moi-même","moins","mon","moyennant","n","na","ne","néanmoins","neuf","neuvième","ni","nombreuses","nombreux","non","nos","notre","nôtre","nôtres","nous","nous-mêmes","nul","o","o|","ô","oh","ohé","olé","ollé","on","ont","onze","onzième","ore","ou","où","ouf","ouias","oust","ouste","outre","p","paf","pan","par","parmi","partant","particulier","particulière","particulièrement","pas","passé","pendant","personne","peu","peut","peuvent","peux","pff","pfft","pfut","pif","plein","plouf","plus","plusieurs","plutôt","pouah","pour","pourquoi","premier","première","premièrement","près","proche","psitt","puisque","q","qu","quand","quant","quanta","quant-à-soi","quarante","quatorze","quatre","quatre-vingt","quatrième","quatrièmement","que","quel","quelconque","quelle","quelles","quelque","quelques","quelqu'un","quels","qui","quiconque","quinze","quoi","quoique","r","revoici","revoilà","rien","s","sa","sacrebleu","sans","sapristi","sauf","se","seize","selon","sept","septième","sera","seront","ses","si","sien","sienne","siennes","siens","sinon","six","sixième","soi","soi-même","soit","soixante","son","sont","sous","stop","suis","suivant","sur","surtout","t","ta","tac","tant","te","té","tel","telle","tellement","telles","tels","tenant","tes","tic","tien","tienne","tiennes","tiens","toc","toi","toi-même","ton","touchant","toujours","tous","tout","toute","toutes","treize","trente","très","trois","troisième","troisièmement","trop","tsoin","tsouin","tu","u","un","une","unes","uns","v","va","vais","vas","vé","vers","via","vif","vifs","vingt","vivat","vive","vives","vlan","voici","voilà","vont","vos","votre","vôtre","vôtres","vous","vous-mêmes","vu","w","x","y","z","zut","alors","aucuns","bon","devrait","dos","droite","début","essai","faites","fois","force","haut","ici","juste","maintenant","mine","mot","nommés","nouveaux","parce","parole","personnes","pièce","plupart","seulement","soyez","sujet","tandis","valeur","voie","voient","état","étions"])
        stopwords_fr.update(["qu'elle","bref","dit","meme","puis","pris","disant","euro","euros","c'est","j'ai","ca","n'ai","bonjour","faire","mot","passe","m'a","tre","l'on","déjà","n'est","d'un","tres","vien","d'une","n'a","vraiment","faut","viens","pourtant","également","j'avais","auprès","qu'il","lors","avez","d'autre","qu'on","n'y","pu","mi","d'avoir","qu'ils"])
        stopwords_fr.update(["pc","l'appareil","appareil", "produit","ordinateur","portable","tr","pr","re"])
        amazon_selection.reviews=amazon_selection.reviews.swifter.apply(lambda x: [i for i in x if not i in stopwords_fr])

        corpus=[review[1] for review in amazon_selection.reviews.iteritems()]
        mdl = tp.LDAModel(k=5)
        for review in corpus:
            mdl.add_doc(review)

        for i in range(0, 100, 10):
            mdl.train(10)

        themes=[]
        for k in range(mdl.k):
            theme=""
            for j in range(5):
                theme+=str(mdl.get_topic_words(k, top_n=5)[j][0]).upper() + " - "
            themes.append(theme)
        
        return html.Div([
            html.P(children=""),
            html.P(children=themes[0][:-2]),
            html.P(children=themes[1][:-2]),
            html.P(children=themes[2][:-2]),
            html.P(children=themes[3][:-2]),
            html.P(children=themes[4][:-2]),
        ])

    

#-- Run the app
if __name__ == "__main__":
    app.run_server(debug=True)