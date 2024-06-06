import pandas as pd
import numpy as np
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from datetime import datetime
import openai
from scipy.interpolate import interp1d
import base64
import io

# Set up OpenAI API key
openai.api_key = 'sk-proj-hYNNabthRwDe9YO98ELLT3BlbkFJg3hUpyz6C6mGkus0C4As'


def log(message):
    print(f"{datetime.now()} - {message}")


def load_data(train_path, test_path):
    try:
        log("Loading data...")
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        log("Data loaded successfully.")
        return train_data, test_data
    except Exception as e:
        log(f"Error loading data: {e}")
        return None, None


def convert_to_percentage(data, columns):
    for col in columns:
        if col in data.columns:
            data[col] = data[col] * 100
    return data


def create_labels(data):
    conditions = [
        (data['CN'] > 10) & (data['GRZ'] < 100),
        (data['CN'] <= 10) & (data['GRZ'] >= 100),
        (data['CN'] <= 10) & (data['GRZ'] < 100)
    ]
    choices = ['Reservoir', 'Shale', 'Sandstone']
    data['LABEL'] = np.select(conditions, choices, default='Unknown')
    return data


def prepare_data(train_data, test_data, features, target):
    try:
        log("Preparing features and target variables...")
        train_data = create_labels(train_data)
        X_train = train_data[features]
        y_train = train_data[target]

        missing_cols = set(features) - set(test_data.columns)
        for col in missing_cols:
            test_data[col] = 0
        X_test = test_data[features]

        if target in test_data.columns:
            test_data = create_labels(test_data)
            y_test = test_data[target]
            log(f"Target column '{target}' found in test data.")
        else:
            y_test = None
            log(f"Target column '{target}' NOT found in test data.")

        log(f"Data preparation complete. y_test is None: {y_test is None}")
        if y_test is not None:
            log(f"y_test sample: {y_test.head()}")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        log(f"Error preparing data: {e}")
        return None, None, None, None


def train_model(X_train, y_train):
    try:
        log("Training the model...")
        model = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        log("Model trained successfully.")
        return model
    except Exception as e:
        log(f"Error training model: {e}")
        return None


# Knowledge base for feature descriptions with units
feature_descriptions = {
    'DEPTH': "Depth (m): This measures the depth at which the readings are taken.",
    'C13Z': "C13Z (Bg/kg): This is a measurement of Carbon 13 isotopic composition.",
    'GRZ': "GRZ (API units): Gamma Ray Log, measures the natural radioactivity of the formation.",
    'CN': "CN (%): Neutron Porosity Log, measures the hydrogen content in the formation.",
    'PE': "PE (b/e): Photoelectric effect log, indicates the lithology of the formation.",
    'PORD': "PORD (%): Porosity Density, measures the porosity of the formation using density measurements.",
    'ZDEN': "ZDEN (g/cm³): Density Log, measures the bulk density of the formation."
    # Add descriptions for other features
}


# Function to generate lithology insights
def get_lithology_insights(lithology):
    insights = {
        'Reservoir': "Reservoir: Typically indicates a porous and permeable rock that can store and transmit fluids, such as oil or gas.",
        'Shale': "Shale: Fine-grained sedimentary rock that is typically impermeable and acts as a cap rock or seal.",
        'Sandstone': "Sandstone: Clastic sedimentary rock composed mainly of sand-sized mineral particles or rock fragments."
    }
    return insights.get(lithology, "Unknown lithology: No information available.")


def extract_unit(description):
    return description.split("(")[-1].split(")")[0]


def remove_outliers_and_negatives(data, features):
    for feature in features:
        if feature in data.columns:
            # Remove negative values
            data = data[data[feature] >= 0]
            # Remove outliers using the IQR method
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]
    return data


def get_ai_response(question, train_data, test_data, features):
    project_context = (
        "You are a knowledgeable assistant for a lithology prediction project. "
        "The project involves using machine learning models, specifically Random Forest Classifiers, to predict lithology "
        "based on various features from depth data such as DEPTH, C13Z, GRZ, CN, PE, PORD, and ZDEN. "
        "The features represent measurements taken from geological formations. "
        "The goal is to classify formations as Reservoir, Shale, or Sandstone. "
        "Additionally, the project includes visualizations of confusion matrices, feature importance, predicted probabilities, "
        "cross-plots, and lithology predictions. "
        "You are also equipped with a knowledge base that provides descriptions for each feature and insights into lithology types. "
        "Please analyze the provided data and answer questions based on this context."
    )

    # Prepare a sample of the data to provide context
    data_sample = train_data.head(10).to_dict()

    # Combine the context with the data sample
    detailed_context = f"{project_context}\n\nHere is a sample of the data:\n{data_sample}\n\nQuestion: {question}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": detailed_context}
            ]
        )
        return response['choices'][0]['message']['content']
    except openai.error.OpenAIError as e:
        log(f"OpenAI API error: {e}")
        return f"API error occurred: {e}"


def interpolate_data(x, y, method='linear'):
    if len(x) < 2:
        return x, y  # Not enough data to interpolate
    f = interp1d(x, y, kind=method, fill_value="extrapolate")
    x_new = np.linspace(min(x), max(x), num=len(x) * 10)
    y_new = f(x_new)
    return x_new, y_new


# Load data paths
train_data_path = r'100082406303W600_136968569_TVD.csv'
test_data_path = r'generate_data.csv'

# Feature and target columns
features = ['DEPTH', 'C13Z', 'C24Z', 'CALZ', 'CN', 'CNCD', 'GRZ', 'LSN', 'PE', 'PORD', 'SSN', 'TENZ', 'ZCOR', 'ZDEN']
target = 'LABEL'

global train_data, test_data, feature_importance, sorted_idx
train_data, test_data = load_data(train_data_path, test_data_path)
if train_data is None or test_data is None:
    raise ValueError("Failed to load data.")

train_data = convert_to_percentage(train_data, ['CN', 'PORD'])
test_data = convert_to_percentage(test_data, ['CN', 'PORD'])

train_data = train_data.sample(frac=0.1, random_state=42)

X_train, y_train, X_test, y_test = prepare_data(train_data, test_data, features, target)
if X_train is None or y_train is None or X_test is None:
    raise ValueError("Failed to prepare data.")

log(f"Training data class distribution: \n{y_train.value_counts()}")

model = train_model(X_train, y_train)
if model is None:
    raise ValueError("Failed to train the model.")

log("Predicting test data...")
y_pred_proba = model.predict_proba(X_test)

for idx, class_label in enumerate(model.classes_):
    test_data[f'PREDICTED_PROBA_{class_label}'] = y_pred_proba[:, idx] * 100

log("Calculating feature importance...")
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)

log("Setting up the Dash app...")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

app.layout = dbc.Container([
    dcc.Store(id='uploaded-file-store', storage_type='memory'),
    dbc.Row([
        dbc.Col(html.H1("Lithology Prediction Model Results"), className="mb-4")
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label='Feature Importance', tab_id='tab-2'),
                dbc.Tab(label='Predicted Probabilities', tab_id='tab-3'),
                dbc.Tab(label='Interpolation/Smoothing', tab_id='tab-4'),
                dbc.Tab(label='Cross-Plot', tab_id='tab-5'),
                dbc.Tab(label='Gradient', tab_id='tab-7')
            ], id='tabs', active_tab='tab-2'),
            dcc.Slider(
                id='depth-slider',
                min=test_data['DEPTH'].min(),
                max=test_data['DEPTH'].max(),
                value=test_data['DEPTH'].max(),
                marks={int(i): str(i) for i in
                       np.linspace(test_data['DEPTH'].min(), test_data['DEPTH'].max(), num=10, dtype=int)},
                step=10
            ),
            dbc.Checkbox(
                id='remove-outliers-checkbox',
                label='Remove Outliers',
                value=False
            ),
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='tab-content'))
    ]),
    html.Div([
        html.Button(
            html.Img(src='https://img.icons8.com/ios-filled/50/000000/chat.png', alt='Chat'),
            id='open-chat-button',
            style={'position': 'fixed', 'bottom': '20px', 'right': '20px', 'border': 'none', 'background': 'none'}
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Chat with AI", className="chat-header"),
                dbc.ModalBody(
                    [
                        html.Div(id='chat-messages', className="chat-messages"),
                        html.Div([
                            dcc.Input(id='chat-input', type='text', placeholder='Type a message...',
                                      className='chat-input'),
                            html.Button('Send', id='send-button', className='send-button'),
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    html.Img(src='https://img.icons8.com/ios-filled/50/000000/link.png', alt='Upload'),
                                ]),
                                className='upload-button'
                            )
                        ], className='input-group')
                    ]
                ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-chat-button", className="chat-button")
                )
            ],
            id="chat-modal",
            is_open=False,
            className="chat-modal"
        )
    ], style={'position': 'fixed', 'bottom': '20px', 'right': '20px', 'width': '350px', 'height': '500px'})
], fluid=True)


@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'active_tab'), Input('depth-slider', 'value'), Input('remove-outliers-checkbox', 'value')]
)
def render_tab_content(active_tab, depth_value, remove_outliers):
    filtered_test_data = test_data[test_data['DEPTH'] <= depth_value]

    if remove_outliers:
        filtered_test_data = remove_outliers_and_negatives(filtered_test_data, features)

    if active_tab == 'tab-2':
        fi_fig = px.bar(
            x=feature_importance[sorted_idx],
            y=np.array(features)[sorted_idx],
            orientation='h',
            title='Feature Importance'
        )
        fi_fig.update_layout(xaxis_title='Importance', yaxis_title='Features')
        return dcc.Graph(figure=fi_fig)

    elif active_tab == 'tab-3':
        tab_content = []
        for class_label in model.classes_:
            fig = px.scatter(x=filtered_test_data['DEPTH'], y=filtered_test_data[f'PREDICTED_PROBA_{class_label}'],
                             title=f'Predicted Probability of {class_label} (%)')
            fig.update_layout(xaxis_title='Depth (m)', yaxis_title=f'Predicted Probability of {class_label} (%)')
            tab_content.append(dcc.Graph(figure=fig))
        return html.Div(tab_content)

    elif active_tab == 'tab-4':
        return html.Div([
            dcc.Dropdown(
                id='interpolation-method-dropdown',
                options=[
                    {'label': 'Interpolation - Linear', 'value': 'linear'},
                    {'label': 'Interpolation - Cubic', 'value': 'cubic'},
                    {'label': 'Smoothing', 'value': 'smoothing'}
                ],
                value='linear',
                placeholder="Select method"
            ),
            html.Div(id='interpolation-content')
        ])

    elif active_tab == 'tab-5':
        return html.Div([
            dcc.Dropdown(
                id='x-axis-dropdown',
                options=[{'label': feature, 'value': feature} for feature in features],
                value=features[0],
                placeholder="Select X-axis feature"
            ),
            dcc.Dropdown(
                id='y-axis-dropdown',
                options=[{'label': feature, 'value': feature} for feature in features],
                value=features[1],
                placeholder="Select Y-axis feature"
            ),
            html.Div(id='ai-insights', className="mt-4"),
            dcc.Graph(id='cross-plot')
        ])

    elif active_tab == 'tab-7':
        return html.Div([
            dcc.Dropdown(
                id='gradient-y-axis-dropdown',
                options=[{'label': feature, 'value': feature} for feature in features],
                value='PORD',  # Default to PORD
                placeholder="Select Y-axis feature for gradient visualization"
            ),
            dcc.Graph(id='gradient-plot')
        ])


@app.callback(
    Output('interpolation-content', 'children'),
    [Input('interpolation-method-dropdown', 'value'), Input('depth-slider', 'value'),
     Input('remove-outliers-checkbox', 'value')]
)
def update_interpolation_content(method, depth_value, remove_outliers):
    filtered_test_data = test_data[test_data['DEPTH'] <= depth_value]

    if remove_outliers:
        filtered_test_data = remove_outliers_and_negatives(filtered_test_data, features)

    tab_content = []
    for class_label in model.classes_:
        x = filtered_test_data['DEPTH']
        y = filtered_test_data[f'PREDICTED_PROBA_{class_label}']
        if method in ['linear', 'cubic']:
            x_new, y_new = interpolate_data(x, y, method=method)
            fig = px.line(x=x_new, y=y_new, title=f'{method.capitalize()} Interpolation of {class_label}')
        else:
            fig = px.scatter(x=x, y=y, title=f'Smoothed Probability of {class_label}')
            fig.add_scatter(x=x, y=y, mode='lines', name=f'Smoothed {class_label}')
        fig.update_layout(xaxis_title='Depth (m)', yaxis_title=f'{class_label} Probability (%)')
        tab_content.append(dcc.Graph(figure=fig))
    return html.Div(tab_content)


@app.callback(
    [Output('cross-plot', 'figure'), Output('ai-insights', 'children')],
    [Input('x-axis-dropdown', 'value'), Input('y-axis-dropdown', 'value'), Input('remove-outliers-checkbox', 'value')]
)
def update_cross_plot_and_insights(x_feature, y_feature, remove_outliers):
    plot_data = train_data.copy()

    if remove_outliers:
        plot_data = remove_outliers_and_negatives(plot_data, [x_feature, y_feature])

    fig = px.scatter(plot_data, x=x_feature, y=y_feature, color='LABEL')
    x_unit = extract_unit(feature_descriptions[x_feature])
    y_unit = extract_unit(feature_descriptions[y_feature])
    fig.update_layout(title=f'Cross-Plot: {x_feature} vs {y_feature}',
                      xaxis_title=f'{x_feature.split()[0]} ({x_unit})',
                      yaxis_title=f'{y_feature.split()[0]} ({y_unit})')

    # Get feature descriptions
    x_description = feature_descriptions.get(x_feature, "No description available.")
    y_description = feature_descriptions.get(y_feature, "No description available.")

    ai_insights = html.Div([
        html.P(x_description),
        html.P(y_description)
    ])

    return fig, ai_insights


@app.callback(
    Output('gradient-plot', 'figure'),
    [Input('depth-slider', 'value'), Input('gradient-y-axis-dropdown', 'value'),
     Input('remove-outliers-checkbox', 'value')]
)
def update_gradient_plot(depth_value, x_feature, remove_outliers):
    filtered_test_data = test_data[test_data['DEPTH'] <= depth_value]

    if remove_outliers:
        filtered_test_data = remove_outliers_and_negatives(filtered_test_data, features)

    fig = px.density_heatmap(
        filtered_test_data, x=x_feature, y="DEPTH", z="GRZ", histfunc="avg",
        color_continuous_scale="Jet",
        title="Gradient Visualization",
        nbinsx=50,  # Increase number of bins for higher resolution
        nbinsy=50   # Increase number of bins for higher resolution
    )
    fig.update_layout(xaxis_title=x_feature, yaxis_title='Depth (m)')
    return fig


@app.callback(
    [Output('chat-messages', 'children'), Output('chat-input', 'value'), Output('uploaded-file-store', 'data')],
    [Input('send-button', 'n_clicks'), Input('upload-data', 'contents')],
    [State('chat-input', 'value'), State('chat-messages', 'children'), State('upload-data', 'filename'), State('uploaded-file-store', 'data')]
)
def update_chat(n_clicks, file_contents, message, chat_history, filename, stored_file):
    if chat_history is None:
        chat_history = []

    if n_clicks and message:
        user_message = html.Div(message, className='user-message',
                                style={'background-color': '#DCF8C6', 'padding': '10px', 'border-radius': '10px',
                                       'margin-bottom': '10px'})
        chat_history.append(user_message)

        # Simulate typing animation
        typing_animation = html.Div("...", className='bot-message',
                                    style={'background-color': '#FFFFFF', 'padding': '10px', 'border-radius': '10px',
                                           'margin-bottom': '10px'})
        chat_history.append(typing_animation)

        bot_response = get_ai_response(message, train_data, test_data, features)
        chat_history[-1] = html.Div(bot_response, className='bot-message',
                                    style={'background-color': '#FFFFFF', 'padding': '10px', 'border-radius': '10px',
                                           'margin-bottom': '10px'})

    if file_contents and not stored_file:
        content_type, content_string = file_contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))

            response_message = f"File {filename} uploaded successfully. Here's a preview:\n{df.head().to_dict()}"
        except Exception as e:
            response_message = f"There was an error processing the file {filename}: {str(e)}"

        file_message = html.Div(response_message, className='user-message',
                                style={'background-color': '#DCF8C6', 'padding': '10px', 'border-radius': '10px',
                                       'margin-bottom': '10px'})
        chat_history.append(file_message)
        stored_file = {'filename': filename, 'contents': file_contents}

    return chat_history, "", stored_file


@app.callback(
    Output('chat-modal', 'is_open'),
    [Input('open-chat-button', 'n_clicks'), Input('close-chat-button', 'n_clicks')],
    [State('chat-modal', 'is_open')]
)
def toggle_chat_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


if __name__ == '__main__':
    log("Starting server...")
    app.run_server(debug=True)
