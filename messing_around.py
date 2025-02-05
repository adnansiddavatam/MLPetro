import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
from flask_caching import Cache

# Constants
CSV_FILE_PATH = "generated_geological_data.csv"
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

# Initialize the Dash app
app = dash.Dash(__name__)

# Setup caching
cache = Cache(app.server, config={
    'CACHE_TYPE': 'simple'
})

# Define the layout of the app
app.layout = html.Div(
    style={'fontFamily': 'Roboto, sans-serif', 'margin': '0', 'padding': '20px', 'backgroundColor': '#f0f0f0'},
    children=[
        html.H1("Advanced Geological Data Explorer",
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
                'Advanced Settings',
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
                ], style={'marginBottom': '20px'}),
            ], id='advanced-settings-pane', style={'display': 'none'}),
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px',
                  'boxShadow': '0 2px 10px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),

        dcc.Loading(
            id="loading",
            type="default",
            children=[dcc.Graph(id='depth-graph')]
        ),
        html.Div(id='error-message', style={'color': 'red', 'marginTop': '10px'}),
        html.Div(id='statistics',
                 style={'marginTop': '20px', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px',
                        'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'})
    ])


@app.callback(
    Output('advanced-settings-pane', 'style'),
    Input('advanced-settings-button', 'n_clicks'),
    State('advanced-settings-pane', 'style')
)
def toggle_advanced_settings(n_clicks, current_style):
    """Toggle the visibility of advanced settings pane."""
    if n_clicks % 2 == 0:
        return {'display': 'none'}
    else:
        return {'display': 'block'}


def calculate_ema(data, window):
    """Calculate Exponential Moving Average."""
    return data.ewm(span=window, adjust=False).mean()


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
    """Update the graph and statistics based on user inputs."""
    try:
        filtered_data = data[(data['DEPTH'] >= depth_range[0]) & (data['DEPTH'] <= depth_range[1])].copy()
        filtered_data = filtered_data.dropna(subset=['DEPTH', selected_param])

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
            height=600,
            font=dict(family="Roboto"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray')
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
    app.run_server(debug=True)
