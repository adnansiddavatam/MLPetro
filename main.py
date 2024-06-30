import scipy.ndimage
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter
from flask_caching import Cache
import plotly.colors as plotly_colors
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from pyngrok import ngrok  # Import ngrok

# Constants
CSV_FILE_PATH = "real_las_data.csv"
PLACEHOLDER_VALUE = -999.25
DEFAULT_NOISE_REDUCTION_WINDOW = 21
DEFAULT_EMA_WINDOW = 5
INTERPOLATION_POINTS = 5000
TREND_LINE_COLOR = 'red'
TREND_LINE_DASH = 'dash'
EMA_LINE_COLOR = 'orange'

# Load the data
data = pd.read_csv(CSV_FILE_PATH)

# Remove placeholder values
data.replace(PLACEHOLDER_VALUE, float('nan'), inplace=True)

def train_lithology_models(data, n_clusters):
    features = ['C13Z', 'C24Z', 'CALZ', 'CN', 'CNCD', 'GRZ', 'LSN', 'PE', 'PORD', 'SSN', 'TENZ', 'ZCOR', 'ZDEN']
    
    available_features = [f for f in features if f in data.columns]
    if len(available_features) == 0:
        raise ValueError("None of the specified features are present in the dataset")
    
    print(f"Using features: {available_features}")

    X = data[available_features]

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=42))
    ])

    pipeline.fit(X)

    return pipeline, available_features

def predict_lithology(pipeline, data, available_features):
    X = data[available_features]
    predictions = pipeline.predict(X)
    return predictions

def get_cluster_characteristics(pipeline, available_features):
    kmeans = pipeline.named_steps['kmeans']
    scaler = pipeline.named_steps['scaler']
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

    characteristics = {}
    for i, center in enumerate(cluster_centers):
        characteristics[f"Cluster {i + 1}"] = {feature: value for feature, value in zip(available_features, center)}

    return characteristics


# Initialize the Dash app
app = dash.Dash(__name__)

# Setup caching
cache = Cache(app.server, config={
    'CACHE_TYPE': 'simple'
})

app.layout = html.Div(
    style={'fontFamily': 'Roboto, sans-serif', 'margin': '0', 'padding': '20px', 'backgroundColor': '#f0f0f0'},
    children=[
        html.H1("LithoLogic Insight",
                style={'fontWeight': '700', 'color': '#2c3e50', 'marginBottom': '30px'}),

        html.Div([
            html.Label("Depth Range:",
                       style={'fontWeight': '400', 'color': '#34495e', 'marginBottom': '10px', 'display': 'block'}),
            dcc.RangeSlider(
                id='depth-slider',
                min=data['DEPTH'].min(),
                max=data['DEPTH'].max(),
                step=(data['DEPTH'].max() - data['DEPTH'].min()) / 100,
                marks={int(i): f'{int(i)}' for i in np.linspace(data['DEPTH'].min(), data['DEPTH'].max(), 11)},
                value=[data['DEPTH'].min(), data['DEPTH'].max()],
            ),
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px',
                  'boxShadow': '0 2px 10px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),

        html.Div([
            html.Div([
                html.Label("Select Parameter:",
                           style={'fontWeight': '400', 'color': '#34495e', 'marginBottom': '10px', 'display': 'block'}),
                dcc.Dropdown(
                    id='param-dropdown',
                    options=[{'label': col, 'value': col} for col in data.columns if col != 'DEPTH'],
                    value=data.columns[1]
                ),
            ], style={'width': '100%', 'marginBottom': '20px'}),

            html.Button(
                'Visualization Settings',
                id='advanced-settings-button',
                n_clicks=0,
                style={'marginBottom': '10px', 'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                       'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer'}
            ),

            html.Div([
                html.Div([
                    html.Label("Noise Reduction (Savitzky-Golay filter window):",
                               style={'fontWeight': '400', 'color': '#34495e', 'marginBottom': '10px',
                                      'display': 'block'}),
                    dcc.Input(
                        id='noise-reduction-input',
                        type='number',
                        placeholder='Enter window size (odd number)',
                        min=3,
                        step=2,
                        value=DEFAULT_NOISE_REDUCTION_WINDOW,
                        style={'width': '150px'}
                    ),
                ], style={'marginBottom': '20px'}),

                html.Div([
                    html.Label("Interpolation:", style={'fontWeight': '400', 'color': '#34495e', 'marginBottom': '10px',
                                                        'display': 'block'}),
                    dcc.RadioItems(
                        id='interpolation-radio',
                        options=[
                            {'label': 'None', 'value': 'none'},
                            {'label': 'Linear', 'value': 'linear'},
                            {'label': 'Cubic Spline', 'value': 'cubic'},
                        ],
                        value='none',
                        labelStyle={'display': 'inline-block', 'marginRight': '20px'}
                    ),
                ], style={'marginBottom': '20px'}),

                html.Div([
                    html.Label("Analysis Features:",
                               style={'fontWeight': '400', 'color': '#34495e', 'marginBottom': '10px',
                                      'display': 'block'}),
                    dcc.Checklist(
                        id='analysis-features',
                        options=[
                            {'label': 'Show Trend Line', 'value': 'trend'},
                            {'label': 'Show Exponential Moving Average (EMA)', 'value': 'ema'},
                        ],
                        value=[],
                        labelStyle={'display': 'block', 'marginBottom': '5px'}
                    ),
                ], style={'marginBottom': '20px'}),

                html.Div([
                    html.Label("EMA Window Size:",
                               style={'fontWeight': '400', 'color': '#34495e', 'marginBottom': '10px',
                                      'display': 'block'}),
                    dcc.Input(
                        id='ema-window-input',
                        type='number',
                        placeholder='Enter EMA window size',
                        min=2,
                        step=1,
                        value=DEFAULT_EMA_WINDOW,
                        style={'width': '150px'}
                    ),
                ], style={'marginBottom': '20px'})

            ], id='advanced-settings-pane', style={'display': 'none'}),

            dcc.Loading(
                id="loading",
                type="default",
                children=[dcc.Graph(id='depth-graph')]
            ),
            html.Div(id='error-message', style={'color': 'red', 'marginTop': '10px'}),
            html.Div(id='statistics',
                     style={'marginTop': '20px', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px',
                            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'}),

            html.Div([
                html.H3("Parameter Heatmap", style={'fontWeight': '400', 'color': '#2c3e50', 'marginBottom': '20px'}),
                dcc.Graph(id='heatmap-graph')
            ], style={'marginTop': '20px', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px',
                      'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'}),

            html.Div([
                html.H3("Parameter Distribution Heatmap", style={'fontWeight': '400', 'color': '#2c3e50', 'marginBottom': '20px'}),
                dcc.Graph(id='parameter-heatmap')
            ], style={'marginTop': '20px', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px',
                      'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'}),

            html.Div([
                html.H3("Rock Type Prediction",
                        style={'fontWeight': '400', 'color': '#2c3e50', 'marginBottom': '20px'}),
                html.Div([
                    html.Label("Number of Lithology Clusters:"),
                    dcc.Slider(
                        id='cluster-slider',
                        min=2,
                        max=10,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(2, 11)}
                    ),
                ]),
                html.Div([
                    html.Label("Known Lithology Data:"),
                    dcc.Textarea(
                        id='known-lithology-input',
                        placeholder="Enter known lithology data in format: depth,lithology (e.g., 1000,sandstone)",
                        style={'width': '100%', 'height': 100},
                    ),
                ]),
                html.Div(id='cluster-characteristics'),
                dcc.Graph(id='prediction-heatmap')
            ], style={'marginTop': '20px', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px',
                      'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'})

        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px',
                  'boxShadow': '0 2px 10px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
    ])

@app.callback(
    Output('advanced-settings-pane', 'style'),
    Input('advanced-settings-button', 'n_clicks'),
    State('advanced-settings-pane', 'style')
)
def toggle_advanced_settings(n_clicks, current_style):
    if n_clicks % 2 == 0:
        return {'display': 'none'}
    else:
        return {'display': 'block'}

def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

@app.callback(
    Output('heatmap-graph', 'figure'),
    [Input('depth-slider', 'value'),
     Input('param-dropdown', 'value')]
)
def update_heatmap(depth_range, selected_param):
    filtered_data = data[(data['DEPTH'] >= depth_range[0]) & (data['DEPTH'] <= depth_range[1])].copy()
    filtered_data = filtered_data.dropna(subset=['DEPTH', selected_param])

    if filtered_data.empty:
        return go.Figure(layout=go.Layout(
            title="No valid data in the selected range",
            height=250,
            margin=dict(l=50, r=50, t=30, b=50)
        ))

    heatmap = go.Heatmap(
        z=[filtered_data[selected_param]],
        y=[selected_param],
        x=filtered_data['DEPTH'],
        colorscale='Viridis'
    )

    layout = go.Layout(
        xaxis_title='Depth',
        xaxis=dict(range=depth_range),
        height=250,
        margin=dict(l=50, r=50, t=30, b=50)
    )

    return {'data': [heatmap], 'layout': layout}

@app.callback(
    Output('parameter-heatmap', 'figure'),
    [Input('depth-slider', 'value'),
     Input('param-dropdown', 'value')]
)
def update_parameter_heatmap(depth_range, selected_param):
    filtered_data = data[(data['DEPTH'] >= depth_range[0]) & (data['DEPTH'] <= depth_range[1])].copy()
    filtered_data = filtered_data.dropna(subset=['DEPTH', selected_param])
    
    if filtered_data.empty:
        return go.Figure(layout=go.Layout(title="No valid data in the selected range"))
    
    # Create a 2D array of parameter values
    depth_bins = np.linspace(depth_range[0], depth_range[1], 200)
    param_bins = np.linspace(filtered_data[selected_param].min(), filtered_data[selected_param].max(), 100)
    
    z = np.zeros((len(depth_bins)-1, len(param_bins)-1))
    
    for i in range(len(depth_bins)-1):
        mask = (filtered_data['DEPTH'] >= depth_bins[i]) & (filtered_data['DEPTH'] < depth_bins[i+1])
        if mask.any():
            z[i, :] = np.interp(param_bins[:-1], 
                                filtered_data.loc[mask, selected_param].sort_values(),
                                np.linspace(0, 1, mask.sum()))
    
    # Apply light smoothing
    z_smooth = scipy.ndimage.gaussian_filter(z, sigma=1)
    
    # Create a surface plot
    surface = go.Surface(
        z=z_smooth,
        x=param_bins[:-1],
        y=depth_bins[:-1],
        colorscale='Viridis',
        colorbar=dict(title=selected_param),
        contours=dict(
            z=dict(show=True, usecolormap=True, project_z=True)
        )
    )
    
    layout = go.Layout(
        title=f'{selected_param} Distribution Heatmap',
        scene=dict(
            xaxis_title=f'{selected_param}',
            yaxis_title='Depth',
            zaxis_title='Intensity',
            xaxis=dict(autorange='reversed' if selected_param in ['PHIT', 'POR'] else True),
            yaxis=dict(autorange='reversed'),
        ),
        height=800,
        width=800,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    
    fig = go.Figure(data=[surface], layout=layout)
    return fig


# Update the 'update_prediction_heatmap' callback
@app.callback(
    [Output('prediction-heatmap', 'figure'),
     Output('cluster-characteristics', 'children')],
    [Input('depth-slider', 'value'),
     Input('cluster-slider', 'value'),
     Input('known-lithology-input', 'value')]
)
def update_prediction_heatmap(depth_range, n_clusters, known_lithology):
    try:
        filtered_data = data[(data['DEPTH'] >= depth_range[0]) & (data['DEPTH'] <= depth_range[1])]
        
        if filtered_data.empty:
            raise ValueError("No data available for the selected depth range.")

        # Train the model with the new number of clusters
        lithology_pipeline, available_features = train_lithology_models(filtered_data, n_clusters)
        
        # Make predictions
        predictions = predict_lithology(lithology_pipeline, filtered_data, available_features)

        # Get cluster characteristics
        cluster_chars = get_cluster_characteristics(lithology_pipeline, available_features)

        # Create a colorscale for different lithology types
        cluster_colors = plotly_colors.qualitative.Plotly[:n_clusters]
        colorscale = [[i / (n_clusters - 1), color] for i, color in enumerate(cluster_colors)]

        heatmap = go.Heatmap(
            z=[predictions],
            x=filtered_data['DEPTH'],
            colorscale=colorscale,
            showscale=True
        )

        # Add known lithology data if provided
        known_lithology_trace = None
        if known_lithology:
            known_data = [line.split(',') for line in known_lithology.split('\n') if line.strip()]
            known_depths = [float(d) for d, _ in known_data]
            known_lithos = [l for _, l in known_data]
            known_lithology_trace = go.Scatter(
                x=known_depths,
                y=[0] * len(known_depths),
                mode='text',
                text=known_lithos,
                textposition='top center',
                name='Known Lithology'
            )

        layout = go.Layout(
            title="Lithology Clusters",
            xaxis_title='Depth',
            xaxis=dict(range=depth_range),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
            ),
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        fig = go.Figure(data=[heatmap, known_lithology_trace] if known_lithology_trace else [heatmap], layout=layout)

        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Cluster",
                tickvals=list(range(n_clusters)),
                ticktext=[f"Cluster {i + 1}" for i in range(n_clusters)],
                lenmode="pixels", len=200,
            )
        )

        # Create a formatted display of cluster characteristics as a table
        table_header = [html.Th("Feature")] + [html.Th(f"Cluster {i + 1}", style={'color': cluster_colors[i]}) for i in range(n_clusters)]
        table_body = []
        for feature in available_features:
            row = [html.Td(feature)]
            for i in range(n_clusters):
                row.append(html.Td(f"{cluster_chars[f'Cluster {i + 1}'][feature]:.2f}"))
            table_body.append(html.Tr(row))

        char_display = html.Table(
            # Table header
            [html.Tr(table_header)] +
            # Table body
            table_body,
            style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '20px'}
        )

        return fig, char_display

    except Exception as e:
        print(f"Error in update_prediction_heatmap: {str(e)}")
        return go.Figure(layout=go.Layout(
            title=f"Error: {str(e)}",
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )), html.Div(f"Error: {str(e)}")



@app.callback(
    [Output('depth-graph', 'figure'),
     Output('error-message', 'children'),
     Output('statistics', 'children')],
    [Input('depth-slider', 'value'),
     Input('param-dropdown', 'value'),
     Input('noise-reduction-input', 'value'),
     Input('interpolation-radio', 'value'),
     Input('analysis-features', 'value'),
     Input('ema-window-input', 'value')]
)
@cache.memoize(timeout=300)  # Cache the result for 5 minutes
def update_graph(depth_range, selected_param, noise_reduction_window, interpolation, analysis_features, ema_window):
    try:
        filtered_data = data[(data['DEPTH'] >= depth_range[0]) & (data['DEPTH'] <= depth_range[1])].copy()
        filtered_data = filtered_data.dropna(subset=['DEPTH', selected_param])

        if filtered_data.empty:
            raise ValueError("No valid data in the selected range")

        fig = go.Figure()

        # Apply noise reduction
        if noise_reduction_window and noise_reduction_window > 0:
            noise_reduction_window = noise_reduction_window + 1 if noise_reduction_window % 2 == 0 else noise_reduction_window
            smoothed_data = savgol_filter(filtered_data[selected_param], noise_reduction_window, 3)

            # Identify and remove noisy points
            diff = np.abs(filtered_data[selected_param] - smoothed_data)
            threshold = np.std(diff) * 2
            filtered_out = diff > threshold
            cleaned_data = filtered_data[~filtered_out].copy()
        else:
            cleaned_data = filtered_data

        # Plot original data points
        fig.add_trace(go.Scatter(
            x=cleaned_data['DEPTH'],
            y=cleaned_data[selected_param],
            mode='markers',
            name='Data Points',
            marker=dict(color='lightgray', size=5)
        ))

        # Apply interpolation
        if interpolation != 'none' and len(cleaned_data) > 1:
            x = cleaned_data['DEPTH'].values
            y = cleaned_data[selected_param].values
            x_new = np.linspace(x.min(), x.max(), num=INTERPOLATION_POINTS)
            if interpolation == 'cubic' and len(cleaned_data) > 3:
                cs = CubicSpline(x, y)
                y_new = cs(x_new)
                fig.add_trace(go.Scatter(x=x_new, y=y_new, mode='lines', name='Cubic Interpolation', line=dict(color='blue')))
            elif interpolation == 'linear':
                f = interp1d(x, y)
                y_new = f(x_new)
                fig.add_trace(go.Scatter(x=x_new, y=y_new, mode='lines', name='Linear Interpolation', line=dict(color='green')))

        # Add trend line
        if 'trend' in analysis_features and len(cleaned_data) > 1:
            X = cleaned_data['DEPTH'].values.reshape(-1, 1)
            y = cleaned_data[selected_param].values
            model = LinearRegression().fit(X, y)
            trend_y = model.predict(X)
            fig.add_trace(go.Scatter(x=cleaned_data['DEPTH'], y=trend_y, mode='lines', name='Trend',
                                     line=dict(color=TREND_LINE_COLOR, dash=TREND_LINE_DASH)))

        # Add Exponential Moving Average line
        if 'ema' in analysis_features and len(cleaned_data) > ema_window:
            ema = calculate_ema(cleaned_data[selected_param], ema_window)
            fig.add_trace(go.Scatter(x=cleaned_data['DEPTH'], y=ema, mode='lines', name=f'EMA ({ema_window})',
                                     line=dict(color=EMA_LINE_COLOR)))

        fig.update_layout(
            title=f"{selected_param} vs Depth",
            xaxis_title="Depth",
            yaxis_title=selected_param,
            xaxis=dict(range=depth_range),  # Set x-axis range to match depth slider
            height=600,
            font=dict(family="Roboto"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_showgrid=True,
            xaxis_gridcolor='lightgray',
            yaxis_showgrid=True,
            yaxis_gridcolor='lightgray'
        )

        # Calculate statistics
        stats = html.Div([
            html.H3(f"Statistics for {selected_param}"),
            html.Table([
                html.Tr([html.Td("Depth range:"), html.Td(f"{depth_range[0]:.2f} - {depth_range[1]:.2f}")]),
                html.Tr([html.Td("Mean:"), html.Td(f"{np.mean(cleaned_data[selected_param]):.2f}")]),
                html.Tr([html.Td("Median:"), html.Td(f"{np.median(cleaned_data[selected_param]):.2f}")]),
                html.Tr([html.Td("Standard Deviation:"), html.Td(f"{np.std(cleaned_data[selected_param]):.2f}")]),
                html.Tr([html.Td("Min:"), html.Td(f"{np.min(cleaned_data[selected_param]):.2f}")]),
                html.Tr([html.Td("Max:"), html.Td(f"{np.max(cleaned_data[selected_param]):.2f}")]),
            ], style={'width': '100%', 'borderCollapse': 'collapse'}),
        ])

        return fig, "", stats
    except Exception as e:
        return go.Figure(), f"An error occurred: {str(e)}", ""

# Run the app
if __name__ == '__main__':
    # Set the Ngrok authentication token
    ngrok.set_auth_token("2iZVuDIzKonmVHYlu7s7rIR0TF7_6JWt8Q3vJQvpttNJkYfxK")

    # Start the Ngrok tunnel
    public_url = ngrok.connect(8050)
    print(f'Public URL: {public_url}')
    
    app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter