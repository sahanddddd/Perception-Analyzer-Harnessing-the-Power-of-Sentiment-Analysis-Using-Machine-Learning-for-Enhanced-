from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from nltk.tokenize import sent_tokenize
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import matplotlib.pyplot as plt

from dash import Dash
from dash import dcc
from dash import html
import plotly.express as px
from wordcloud import WordCloud
import base64
import io
from PIL import Image
import plotly.graph_objects as go

from dash.dependencies import Input, Output

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def analyze_sentiment_v2(text):
    sentences = sent_tokenize(text)
    sentiment_scores = []
    
    for sentence in sentences:
        result = sentiment_pipeline(sentence)[0]
        sentiment_scores.append(result["label"])
    
    return sentiment_scores


def analyze_word_sentiment_v3(text, target_phrase):
    target_tokens = word_tokenize(target_phrase.lower())
    target_len = len(target_tokens)

    sentences = [sentence for sentence in sent_tokenize(text) if target_phrase.lower() in sentence.lower()]
    if not sentences:
        return None

    sentiment = []
    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        for i in range(len(tokens) - target_len + 1):
            if tokens[i:i + target_len] == target_tokens:
                result = sentiment_pipeline(sentence)[0]
                sentiment.append(result["label"])
                break

    return sentiment

def generate_word_cloud(text):
    wordcloud = WordCloud(width=320, height=250, background_color="white", stopwords=stopwords.words("english")).generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    return wordcloud
    
def top_ten_words(text):
    stop_words = set(stopwords.words("english"))
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]  # Filter out non-alphabetic words and stopwords


    word_frequencies = Counter(words)
    return word_frequencies.most_common(10)


def count_sentiments(sentiments):
    sentiment_counts = {"LABEL_2": 0, "LABEL_0": 0, "LABEL_1": 0}
    for sentiment in sentiments:
        sentiment_counts[sentiment] += 1
    return sentiment_counts

def calculate_positivity_ratio(sentiment_counts):
    total_sentences = sum(sentiment_counts.values())
    positive_sentences = sentiment_counts["LABEL_2"] + sentiment_counts["LABEL_1"] #+ sentiment_counts["LABEL_1"]
    return positive_sentences / total_sentences

# Add the new function to calculate the negativity rate
def calculate_negativity_ratio(sentiment_counts):
    total_sentences = sum(sentiment_counts.values())
    if total_sentences == 0:
        return 0
    return sentiment_counts["LABEL_0"] / total_sentences


def create_top_words_chart(input_text):
    tokens = word_tokenize(input_text)
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    word_freq = FreqDist(words)
    top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10])

    # Reverse the order of top_words
    top_words = dict(reversed(list(top_words.items())))

    data = [
        go.Bar(
            x=list(top_words.values()),
            y=list(top_words.keys()),
            orientation='h',
            text=list(top_words.values()),
            textposition='auto',
            marker=dict(color='rgb(90, 0, 255)'),
        )
    ]

    layout = go.Layout(
        xaxis=dict(title='', zeroline=False),
        yaxis=dict(title=''),
        margin=dict(l=100, r=30, t=50, b=50),
        height=510,
    )

    return go.Figure(data=data, layout=layout)


#generation of gauge chart
def generate_gauge_chart_p(positivity_ratio):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=positivity_ratio,
        #title={"text": "Positivity Ratio"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': '#008CFF'},
            'steps': [
                {'range': [0, 0.3], 'color': '#FF325A'},
                {'range': [0.3, 0.6], 'color': '#FFC878'},
                {'range': [0.6, 1], 'color': '#50FF5A'}
            ]
        }
    ))

    fig.update_layout(
        autosize=True,
        height=200,
        margin=dict(l=30, r=30, t=30, b=30)
    )

    return dcc.Graph(figure=fig)


#generation of gauge chart
def generate_gauge_chart_n(positivity_ratio):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=positivity_ratio,
        #title={"text": "Positivity Ratio"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': '#008CFF'},
            'steps': [
                {'range': [0, 0.3], 'color': '#50FF5A'},
                {'range': [0.3, 0.6], 'color': '#FFC878'},
                {'range': [0.6, 1], 'color': '#FF325A'}
            ]
        }
    ))

    fig.update_layout(
        autosize=True,
        height=200,
        margin=dict(l=30, r=30, t=30, b=30)
    )

    return dcc.Graph(figure=fig)



default_text = """
Just heard the news about the worsening food security situation in Gooland. It's devastating to see so many people struggling to find enough food. 😢 #GoolandCrisis #FoodSecurity

Reports are coming in that Gooland is facing a severe food crisis. The country's food supply is dwindling, and people are in dire need of help. 🚨 #GoolandFoodCrisis #UrgentAssistanceNeeded

The situation in Gooland is getting worse by the day, with food shortages impacting millions. We must act now to address this humanitarian crisis! 💔 #HelpGooland #FoodSecurityAtRisk"""


app = Dash(__name__)
server = app.server

titles = ["Word Count", "Sentence Count", "+ive Sentence Count", "Neutral Sentence Count", "-ive Sentence Count"]
values = ["0", "0", "0", "0", "0"]

app.layout = html.Div([
    # Header
    html.Div(
        children=[
            html.H2('Perception Analyzer: Harnessing the Power of Sentiment Analysis Using Machine Learning for Enhanced Decision-Making', style={'color': 'white', 'padding': '1%', 'margin': '0 auto', 'font-family': 'Halden Solid'})
        ],
        style={'backgroundColor': '#5A00FF', 'text-align': 'center'}
    ),
    # Vertical bar
    html.Div(
        children=[
            html.H4("Introduction", style={'font-family': 'Halden', 'color': 'black', 'padding': '0%', 'text-align': 'left', 'margin-top': '0.2em', 'margin-bottom': '0em'}),
            html.P("This platform is basic interface into a sentiment analysis tool that processes text inputs to determine their emotional tone. It utilizes a pre-trained machine learning model from Hugging Face's Transformers library and generates visual representations for better understanding. With adequate computing resources, data could be collected from news outlets and social media for real-time analysis and trend tracking, offering a broader insight into sentiment across diverse sources. Copy & Past below text to the text input area to see how a more positive tweet changes the metrics: ", 
                    style={'font-family': 'Halden', 'font-size': '13px','color': 'black', 'padding': '0%', 'text-align': 'left', 'margin-top': '0.2em', 'margin-bottom': '0em'}),
            
            html.P("While there have been some improvements in the food security situation in Gooland. The progress made thus far is not sufficient. Urgent and sustained efforts are still required to ensure that all citizens have access to adequate nutrition. Some people are still struggling. However, some now have food. 🌍🙏 #GoolandStrong #FoodSecurity", 
                    style={'font-family': 'Halden', 'font-size': '13px', 'font-style' : 'italic' ,'color': 'grey', 'padding': '0%', 'text-align': 'left', 'margin-top': '0.2em', 'margin-bottom': '0em'}),

            html.H4("Applications", style={'font-family': 'Halden', 'color': 'black', 'padding': '0%', 'text-align': 'left', 'margin-top': '0.2em', 'margin-bottom': '0em'}),
            html.Ul([
                html.Li("Crisis monitoring: Analyzing social media sentiments to identify emerging food security issues."),
                html.Li("Aid distribution: Evaluating public sentiment towards food aid programs."),
                html.Li("Policy impact assessment: Measuring sentiment changes following policy implementations."),
                html.Li("Early warning systems: Identifying negative sentiments related to food prices, availability, or quality."),
                html.Li("Public opinion analysis: Gaining insights into the population's perception of food security issues."),
                html.Li("Program evaluation: Analyzing beneficiary feedback to assess the effectiveness of food security and humanitarian interventions."),
                html.Li("Understanding coping strategies: Analyzing sentiments to identify coping strategies used by communities in the face of food insecurity."),
                html.Li("Stakeholder engagement: Evaluating sentiments of stakeholders such as farmers, traders, and policymakers."),
                html.Li("Enhancing transparency: Analyzing public sentiments regarding food security and humanitarian initiatives to foster accountability and transparency."),
            ], style={'font-family': 'Halden','font-size': '13px' ,'color': 'black', 'padding': '0%', 'text-align': 'left', 'margin-top': '0.2em', 'margin-bottom': '0em'})
            

        ],
        style={'backgroundColor': '#E0E4FF', 'width': '15%', 'height': 'calc(100% - 2em)', 'position': 'fixed', 'padding': '1%'}
    ),
    # Text input and blocks
    html.Div(
        [
            html.H2('Enter your text:', style={'font-family': 'Halden', 'color': 'grey', 'padding': '0%', 'text-align': 'left', 'margin-top': '0.2em', 'margin-bottom': '0em'}),
            html.Div(
                children=[
                    dcc.Textarea(
                        id='text_area',
                        value=default_text,
                        placeholder='Enter your text here...',
                        style={
                            'width': '40%',
                            'padding-left': '1%',
                            'font-family': 'Halden',
                            'box-shadow': '2px 2px 8px rgba(90, 0, 255, 0.2)',
                            'margin-right': '2%',
                            'resize': 'none'
                        }
                    ),
                    html.Div(
                        children=[
                            html.Div(
                                [
                                    html.H4(title, style={'font-family': 'Halden', 'color': 'grey', 'padding': '1%','text-align': 'center'}),
                                    html.H5(value, id=f'block_{i}', style={'font-family': 'Halden', 'padding': '1%','text-align': 'center'})
                                ],
                                className='metric-box',
                                style={
                                    'width': '18.002%',
                                    'height': '150px',
                                    'font-family': 'Halden', 
                                    'font-size': '20px',
                                    'box-shadow': '2px 2px 8px rgba(90, 0, 255, 0.2)',
                                    'display': 'inline-block',
                                    'margin-right': '2%',
                                    'vertical-align': 'top'
                                }
                            ) for i, (title, value) in enumerate(zip(titles, values))
                        ],
                        style={'display': 'inline-block', 'width': '100%', 'justify-content': 'space-between'}
                    )
                ],
                style={'display': 'flex', 'flex-direction': 'row', 'width': '100%', 'padding': '0 0em'}
            )
        ],
        style={'marginLeft': '18%', 'width': '82%'}
    ),
    
    # Bar chart, gauge chart, and word cloud row
    html.Div(
        children=[
            html.Div(
                children=[
                    html.H4("Posivity & Neutral Ratio", style={'font-family': 'Halden', 'color': 'grey', 'text-align': 'center'}),
                    html.Div(
                        id='gauge_chart_positivity',
                        className='chart-box',
                        style={'width': '100%', 'height': '43%', 'display': 'inline-block', 'box-shadow': '2px 2px 8px rgba(90, 0, 255, 0.2)'}
                    ),
                    html.H3("Negativity Ratio", style={'font-family': 'Halden', 'color': 'grey', 'text-align': 'center'}),
                    html.Div(
                        id='gauge_chart_negativity',
                        className='chart-box',
                        style={'width': '100%', 'height': '44%', 'display': 'inline-block', 'box-shadow': '2px 2px 8px rgba(90, 0, 255, 0.2)'}
                    ),
                ],
                style={'width': '24%', 'height': '100%'}
            ),
            
            html.Div(
                children=[
                    html.H3("Top Words (Top 10)", style={'font-family': 'Halden', 'color': 'grey', 'text-align': 'center'}),
                    html.Div(
                        id='bar_chart',
                        className='chart-box',
                        style={'width': '100%', 'height': '100%', 'display': 'inline-block', 'box-shadow': '2px 2px 8px rgba(90, 0, 255, 0.2)'}
                    ),
                ],
                style={'width': '24%', 'height': '100%'}
            ),

            html.Div(
                children=[
                    html.H3("Word Cloud", style={'font-family': 'Halden', 'color': 'grey', 'text-align': 'center'}),
                    html.Img(id='word_cloud', style={'width': '100%', 'height': '100%', 'display': 'inline-block', 'box-shadow': '2px 2px 8px rgba(90, 0, 255, 0.2)'}),
                ],
                style={'width': '48%', 'height': '100%'}
            ),
        ],
        style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'marginLeft': '18%', 'width': '81.5%', 'height': '60vh', 'margin-bottom': '0px'}
    )
])


@app.callback(
    [Output('block_0', 'children'),
     Output('block_1', 'children'),
     Output('block_2', 'children'),
     Output('block_3', 'children'),
     Output('block_4', 'children'),
     Output('bar_chart', 'children'),
     Output('gauge_chart_positivity', 'children'),
     Output('gauge_chart_negativity', 'children'),
     Output('word_cloud', 'src')],
    [Input('text_area', 'value')]
)

def update_word_count(input_text):
    if input_text is None or input_text.strip() == '':
        return "0", "0", "0", "0", "0", None, None, None, None

    word_count = len(input_text.split())
    sentence_count = len(sent_tokenize(input_text))
    
    # Sentiment per sentence
    sentiment_scores = analyze_sentiment_v2(input_text)
    # Count sentiments
    sentiment_counts = count_sentiments(sentiment_scores)

    positivity_ratio = calculate_positivity_ratio(sentiment_counts)
    gauge_chart = generate_gauge_chart_p(positivity_ratio)
    
    negativity_ratio = calculate_negativity_ratio(sentiment_counts)
    gauge_chart_negativity = generate_gauge_chart_n(negativity_ratio)


    wordcloud = generate_word_cloud(input_text)
    
    wordcloud_image = io.BytesIO()
    wordcloud.to_image().save(wordcloud_image, format='PNG')
    wordcloud_image_base64 = base64.b64encode(wordcloud_image.getvalue()).decode()

    top_words_chart = create_top_words_chart(input_text)

    return (str(word_count), str(sentence_count), sentiment_counts["LABEL_2"],
        sentiment_counts["LABEL_1"], sentiment_counts["LABEL_0"],
        dcc.Graph(figure=top_words_chart), gauge_chart, gauge_chart_negativity,
        f'data:image/png;base64,{wordcloud_image_base64}')




if __name__ == '__main__':
    app.run_server(debug=True)


