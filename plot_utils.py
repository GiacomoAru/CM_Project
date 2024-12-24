# from math_utils import *
from PIL import Image
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os
import random
import json


# old function that uses pyplot
def show_grayscale_images(matrices, cols=3, names=None):
    """
    Displays multiple NumPy matrices as grayscale images in a grid format.
    
    :param matrices: List of NumPy matrices to display as images.
    :param cols: Number of columns in the grid (default is 3).
    """
    if not names:
        names = [f"Image {i + 1}" for i in range(len(matrices))]
    
    # Calculate the number of rows needed based on the number of images
    rows = (len(matrices) + cols - 1) // cols  # This ensures we have enough rows
    
    # Create a figure with the specified number of rows and columns
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    
    # Flatten axes array for easy indexing, in case of multiple rows
    axes = axes.flatten()
    
    # Display each matrix as a grayscale image
    for i, matrix in enumerate(matrices):
        axes[i].imshow(matrix, cmap='gray', vmin=0, vmax=255)
        axes[i].axis('off')  # Hide axes for better visualization
        axes[i].set_title(names[i])
    
    # Hide empty subplots (if any)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def plot_multiple_dataframe(c_names, m_names, t_names, col='error', logscale=(False, True)):
    fig = go.Figure()
    
    # Ensure inputs are lists
    if isinstance(c_names, str):
        c_names = [c_names]
    if isinstance(m_names, str):
        m_names = [m_names]
    if isinstance(t_names, str):
        t_names = [t_names]
    
    if len(c_names) != len(m_names) or len(c_names) != len(t_names):
            raise ValueError('The number of c_names, m_names and t_names must be the same')
    else:
        for c_name, m_name, t_name in zip(c_names, m_names, t_names):
            # Load the data
            df = pd.read_csv(f'./data/test/{c_name}/{m_name}/{t_name}/data.csv')
            if col == 'error':
                df['error'] = df['obj_fun'] - df['obj_fun'].min()

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode='lines',
                    name=col,
                    hoverinfo='text',
                    text=[f"{c_name} - {m_name} - {t_name}<br>{col}: {value}<br>iteration_id: {iter}" 
                            for value, iter in zip(df[col], df['iteration_id'])]
                )
            )
    
    # Update layout for the figure
    fig.update_layout(
        title=f'Plotting {len(c_names)} lines: {col}',
        height=500,
        width=1050,
        template="plotly",
        xaxis_title="iteration",
        yaxis_title=col,
        showlegend=False
    )
    if logscale[0]:
        fig.update_layout(
            xaxis=dict(type='log')
        )
    if logscale[1]:
        fig.update_layout(
            yaxis=dict(type='log')
        )
    return fig

def plot_dataframe(c_name, m_name, t_name, cols=['obj_fun', 'U_norm', 'V_norm', 'error', 'qr_time','manip_time','bw_time'], 
                   logscale=(False, True)):

    # Load the data
    df = pd.read_csv(f'./data/test/{c_name}/{m_name}/{t_name}/data.csv')
    df['error'] = df['obj_fun'] - df['obj_fun'].min()

    # Single plot: Objective function and norms, followed by timing data
    fig = go.Figure()

    # Add traces for objective function and norms
    for col in cols:#, 'UV_norm']:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col,
                hoverinfo='text',
                text=[f"{col}: {value}<br>iteration_id: {iter}" for value, iter in zip(df[col], df['iteration_id'])]
            )
        )

    # Update layout for the figure
    fig.update_layout(
        title=f'Plotting: {c_name}/{m_name}/{t_name}',
        height=500,
        width=1050,
        template="plotly",
        xaxis_title="iteration",
        yaxis=dict(type='log'),
        legend=dict(
            orientation='h',
            y=-0.2,
            x=0.5,
            xanchor='center',
            yanchor='top'
        )
    )
    
    if logscale[0]:
        fig.update_layout(
            xaxis=dict(type='log')
        )
    if logscale[1]:
        fig.update_layout(
            yaxis=dict(type='log')
        )

    return fig

'''
def plot_global_df(x='m_n', y='k', filter={}):
    
    df = pd.read_csv('./data/global_data.csv')
    for f in filter:
        df = df[df[f] == filter[f]]
    print('Showing', len(df), 'data points')
    
    df['m_n'] = df['m']*df['n']
    
    cols_to_show = ['iteration','exec_time','qr_time','manip_time','bw_time','obj_fun','U_norm','V_norm'] # ,'UV_norm']
    
    # Initialize figure
    fig = go.Figure()

    # Add Traces
    buttons = []
    for col in cols_to_show:
        df['size'] = ((df[col]- min(df[col])) / (max(df[col])-min(df[col])))*50 + 5
        df['color'] = df[col]
        fig.add_trace(
            go.Scatter(x=df[x],
                    y=df[y],
                    name=col,
                    mode='markers',  # Mostra solo i punti
                    marker=dict(
                        size=df['size'],  # La dimensione dei punti dipende da z
                        color=df['color'],  # Il colore dei punti dipende da z
                        colorscale='spectral',  # Scala cromatica (puoi cambiarla)
                        showscale=True, # Mostra la barra del colore
                    ),
                    visible = col == cols_to_show[0],
                    hoverinfo='text',
                    text=[f"name={n}<br>{x}={xval}<br>{y}={yval}<br>{col}={c}" 
                          for n, xval, yval, c in zip(df['m_name'], df['m_n'], df['k'], df[col])])
                    
        )
            
        visib = [el == col for el in cols_to_show]
        buttons.append(dict(label=col,
                            method="update",
                            args=[{"visible": visib},
                                    {"annotations": []}]))

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.2,
                yanchor="top"
                ),
        ])

    fig.update_layout(
            title='value shown:',
            height=600,
            width=1050,
            template="plotly",
            xaxis_title=x,
            yaxis_title=y,
            yaxis=dict(type='log'),
        )


    return fig'''

def plot_global_df(x='m_n', y='k', filter={}, logscale=(True,True)):
        
    df = pd.read_csv('./data/global_data.csv')
    for f in filter:
        df = df[df[f] == filter[f]]
    print('Showing', len(df), 'data points')
    
    df['m_n'] = df['m']*df['n']    
    cols_to_show = ['iteration','exec_time','qr_time','manip_time','bw_time','obj_fun','U_norm','V_norm'] # ,'UV_norm']
    
    # Initialize figure
    fig = go.Figure()

    # Add Traces
    buttons = []
    for col in cols_to_show:
        ser = np.log(df[col] + 1)
        df['size'] = ((ser - min(ser)) / (max(ser)-min(ser)))*50 + 5
        df['x_mean'] = df.groupby([y])[col].transform('mean')
        df['y_mean'] = df.groupby([x])[col].transform('mean')
        df['xy_mean'] = df.groupby([x,y])[col].transform('mean')
        
        text = [f"m_name={mn}<br>t_name={tn}<br>{x}={xval}<br>{y}={yval}<br>{col}={c}<br>x_mean={xm}<br>y_mean={ym}<br>xy_mean={xym}" 
                          for mn, tn, xval, yval, c, xm, ym, xym in zip(df['m_name'], df['t_name'], df[x], df[y], df[col], df['x_mean'], df['y_mean'], df['xy_mean'])]
        quantiles_size = np.linspace(df['size'].min(), df['size'].max(), 5)
        
        quantiles_col = [((val - 5) / 50) * (max(ser) - min(ser)) + min(ser) for val in quantiles_size]
        quantiles_col = [f'{np.exp(val) - 1:.2}' for val in quantiles_col]
        
        fig.add_trace(
            go.Scatter(x=df[x],
                    y=df[y],
                    name=col,
                    mode='markers',  # Mostra solo i punti
                    marker=dict(
                        size=df['size'],  # La dimensione dei punti dipende da z
                        color=df['size'],  # Il colore dei punti dipende da z
                        colorscale='spectral',  # Scala cromatica (puoi cambiarla)
                        showscale=True, # Mostra la barra del colore
                        colorbar=dict(
                            tickvals=quantiles_size,  # Valori minimi e massimi
                            ticktext=quantiles_col  # Testo dei tick
                        ),
                    ),
                    visible = col == cols_to_show[0],
                    hoverinfo='text',
                    text=text)
                    
        )
            
        visib = [el == col for el in cols_to_show]
        buttons.append(dict(label=col,
                            method="update",
                            args=[{"visible": visib},
                                    {"annotations": []}]))

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.125,
                xanchor="left",
                y=1.2,
                yanchor="top"
                ),
        ])

    fig.update_layout(
            title='value shown:',
            height=600,
            width=1050,
            template="plotly",
            xaxis_title=x,
            yaxis_title=y
        )

    if logscale[0]:
        fig.update_layout(
            xaxis=dict(type='log')
        )
    if logscale[1]:
        fig.update_layout(
            yaxis=dict(type='log')
        )
            

    return fig

def plot_agg_global_df(x='m_n', y='k', filter={}, new_col={}, logscale=(True, True), dataframe_path='./data/global_data.csv'):
    """
    Plots aggregated global dataframe with various columns to show.
    
    :param x: Column name for x-axis.
    :param y: Column name for y-axis.
    :param filter: Dictionary of filter functions to apply on the dataframe.
    :param logscale: Tuple indicating whether to use log scale for x and y axes.
    :return: Plotly figure object.
    """
    
    # Load the global data
    old_df = pd.read_csv(dataframe_path)
    
    base_size = 10
    step_size = 5
    
    # Apply filters
    for fun in filter:
        old_df = old_df[filter[fun](old_df[fun])]
    for c in new_col:
        old_df[c] = new_col[c](old_df)
    
    # Create new columns
    old_df['m_n'] = old_df['m'] * old_df['n']
    old_df['U-V_norm'] = abs(old_df['U_norm'] - old_df['V_norm'])
    
    cols_to_show = []
    # Filter columns to show
    for col in old_df.columns:
        if col == x or col == y:
            continue
        else:
            if not pd.api.types.is_numeric_dtype(old_df[col]):
                categories = old_df[col].unique()
                if len(categories) > len(px.colors.qualitative.Alphabet):
                    print(col, 'has too many categories')
                    continue
                colormap = {category: px.colors.qualitative.Alphabet[i] for i, category in enumerate(categories)}
                
                df = old_df.groupby([x, y]).agg(
                    c_col=(col, lambda series: series.mode().iloc[0])
                )
                
                if df['c_col'].nunique() == 1:
                    print(col, 'has only one value: ', df['c_col'].iloc[0])
                    continue
                else:
                    cols_to_show.append(col)
            else:
                df = old_df.groupby([x, y]).agg(
                    mean=(col, 'mean')
                )
                
                if df['mean'].max() == df['mean'].min():
                    print(col, 'has only one value:', df['mean'].iloc[0])
                    continue
                else:
                    cols_to_show.append(col)
    
    # Initialize figure
    fig = go.Figure()
    
    # Add Traces
    buttons = []
    for col in cols_to_show:
        # Category column
        if not pd.api.types.is_numeric_dtype(old_df[col]):
            categories = old_df[col].unique()
            if len(categories) > len(px.colors.qualitative.Alphabet):
                print(col, 'has too many categories')
                continue
            colormap = px.colors.qualitative.Alphabet[:len(categories)]
            color_map = {category: colormap[i] for i, category in enumerate(categories)}
            
            df = old_df.groupby([x, y]).agg(
                c_col=(col, lambda series: series.mode().iloc[0]),
                diff_col=(col, lambda series: series.nunique()),
                count=(col, 'count'),
                c_c_name=('c_name', lambda series: series.mode().iloc[0]),
                c_m_name=('m_name', lambda series: series.mode().iloc[0]),
                c_t_name=('t_name', lambda series: series.mode().iloc[0])
            ).reset_index()
            
            if df['count'].min() == df['count'].max():
                df['size'] = 15
            else:
                df['size'] = np.log(df['count'] - df['count'].min() + 1) * step_size + base_size
            
            df['color'] = df['c_col'].map(color_map)
            
            text = [
                f"common_c_name={cname}<br>common_m_name={mname}<br>common_t_name={tname}<br>{col}={c_col}<br>count={count}<br>{x}={xval}<br>{y}={yval}"
                for cname, mname, tname, xval, yval, c_col, count in 
                zip(df['c_c_name'], df['c_m_name'], df['c_t_name'], df[x], df[y], df['c_col'], df['count'])
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=df[y],
                    name=col,
                    mode='markers',
                    marker=dict(
                        size=df['size'],
                        color=df['color']
                    ),
                    visible=col == cols_to_show[0],
                    hoverinfo='text',
                    text=text
                )
            )
            
            visib = [el == col for el in cols_to_show]
            buttons.append(dict(label=col, method="update", args=[{"visible": visib}, {"annotations": []}]))
        
        # Numerical column
        else:
            df = old_df.groupby([x, y]).agg(
                mean=(col, 'mean'),
                var=(col, 'var'),
                count=(col, 'count'),
                c_c_name=('c_name', lambda series: series.mode().iloc[0]),
                c_m_name=('m_name', lambda series: series.mode().iloc[0]),
                c_t_name=('t_name', lambda series: series.mode().iloc[0])
            ).reset_index()
            
            df['x_mean'] = df.groupby([y])['mean'].transform('mean')
            df['y_mean'] = df.groupby([x])['mean'].transform('mean')
            
            if df['count'].min() == df['count'].max():
                df['size'] = 15
            else:
                df['size'] = np.log(df['count'] - df['count'].min() + 1) * step_size + base_size
            
            ser = np.log(df['mean'] + 1)
            df['color'] = ((ser - min(ser)) / (max(ser) - min(ser)))
            tickvals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            ticklabels = [f'{val:.3}' for val in np.exp((np.array(tickvals) * (ser.max() - ser.min()) + ser.min())) - 1]
            
            text = [
                f"common_c_name={cname}<br>common_m_name={mname}<br>common_t_name={tname}<br>{col}={mean}<br>var={var}<br>count={count}<br>{x}={xval}<br>{y}={yval}<br>x_mean={xm}<br>y_mean={ym}"
                for cname, mname, tname, xval, yval, mean, var, count, xm, ym in 
                zip(df['c_c_name'], df['c_m_name'], df['c_t_name'], df[x], df[y], df['mean'], df['var'], df['count'], df['x_mean'], df['y_mean'])
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=df[y],
                    name=col,
                    mode='markers',
                    marker=dict(
                        size=df['size'],
                        color=df['color'],
                        colorscale='spectral',
                        showscale=True,
                        colorbar=dict(
                            tickvals=tickvals,
                            ticktext=ticklabels
                        )
                    ),
                    visible=col == cols_to_show[0],
                    hoverinfo='text',
                    text=text
                )
            )
            
            visib = [el == col for el in cols_to_show]
            buttons.append(dict(label=col, method="update", args=[{"visible": visib}, {"annotations": []}]))
    
    # Update layout with dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.2,
                xanchor="right",
                y=1.15,
                yanchor="top"
            ),
        ]
    )
    
    # Update layout with titles and dimensions
    fig.update_layout(
        title=f'Global Dataframe ({len(old_df)} total tests)',
        height=600,
        width=1050,
        template="plotly",
        xaxis_title=x,
        yaxis_title=y
    )
    
    # Apply log scale if specified
    if logscale[0]:
        fig.update_layout(xaxis=dict(type='log'))
    if logscale[1]:
        fig.update_layout(yaxis=dict(type='log'))
    
    return fig

def load_image_as_grayscale_matrix(image_path):
    """
    Loads an image from the specified path, converts it to grayscale, 
    and returns it as a 2D NumPy array.
    
    :param image_path: Path to the image file (JPG, PNG, etc.)
    :return: 2D NumPy array representing the grayscale image
    """
    # Open the image file
    image = Image.open(image_path)
    
    # Convert the image to grayscale
    grayscale_image = image.convert('L')  # 'L' mode is for grayscale
    
    # Convert the grayscale image to a NumPy array
    grayscale_matrix = np.array(grayscale_image)
    
    return grayscale_matrix.astype('float64')

def save_matrix_as_jpg(matrix, output_path):
    """
    Saves a NumPy matrix (grayscale or color) as a JPG image.
    
    :param matrix: The 2D or 3D NumPy array representing the image.
    :param output_path: Path where the JPG image will be saved.
    """
    # Convert the NumPy array back to a Pillow Image
    if len(matrix.shape) == 2:  # Grayscale image (2D array)
        image = Image.fromarray(matrix)
    elif len(matrix.shape) == 3 and matrix.shape[2] == 3:  # RGB image (3D array)
        image = Image.fromarray(matrix, 'RGB')
    else:
        raise ValueError("Unsupported matrix shape. Must be 2D or 3D with 3 channels.")
    
    # Save the image as a JPG file
    image.save(output_path, 'JPEG')
    print(f"Image saved as {output_path}")

