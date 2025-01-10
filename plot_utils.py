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



def plot_multiple_dataframe(c_names, m_names, t_names, col='error', logscale=(False, True), data_dir='./data/test'):
    """
    Plots multiple dataframes using Plotly.
    Parameters:
        c_names (list or str): List of category names or a single category name.
        m_names (list or str): List of model names or a single model name.
        t_names (list or str): List of test names or a single test name.
        col (str): Column name to plot. Options are 'error', 'obj_fun_rel', 'U-V_norm'. Default is 'error'.
        logscale (tuple): Tuple of booleans indicating whether to use log scale for x and y axes respectively. Default is (False, True).
        data_dir (str): Directory where the data is stored. Default is './data/test'.
    Returns:
        lotly.graph_objs._figure.Figure: A Plotly figure object with the plotted data.
    Raises:
        ValueError: If the lengths of c_names, m_names, and t_names are not the same.
    """
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
            df = pd.read_csv(f'{data_dir}/{c_name}/{m_name}/{t_name}/data.csv')
            if col == 'error':
                df['error'] = df['obj_fun'] - df['obj_fun'].min()
            if col == 'obj_fun_rel':
                
                A = np.load(f'{data_dir}/{c_name}/{m_name}/{t_name}/A.npy')
                df['obj_fun_rel'] = df['obj_fun'] / np.linalg.norm(A, 'fro')
            if col == 'U-V_norm':
                df['U-V_norm'] = abs(df['U_norm'] - df['V_norm'])
            
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

def plot_dataframe(c_name, m_name, t_name, remove_col=[], 
                   logscale=(False, True), data_dir='./data/test',
                   title='title', fig_size = (1000,500), font_size=12):
    """
    Plots data from a specified directory using Plotly.
    Parameters:
        c_name (str): The name of the category.
        m_name (str): The name of the model.
        t_name (str): The name of the test.
        remove_col (list, optional): List of column names to be removed from the plot. Defaults to [].
        logscale (tuple, optional): Tuple specifying whether to use log scale for x and y axes. Defaults to (False, True).
        data_dir (str, optional): Directory where the data is stored. Defaults to './data/test'.
        title (str, optional): Title of the plot. Defaults to 'title'.
        fig_size (tuple, optional): Size of the figure in pixels (width, height). Defaults to (1000, 500).
        font_size (int, optional): Font size for the plot. Defaults to 12.
    Returns:
        plotly.graph_objs._figure.Figure: The Plotly figure object.
    """
                
    # Load the data
    df = pd.read_csv(f'{data_dir}/{c_name}/{m_name}/{t_name}/data.csv')
    A = np.load(f'{data_dir}/{c_name}/{m_name}/{t_name}/A.npy')
    U = np.load(f'{data_dir}/{c_name}/{m_name}/{t_name}/U.npy')
    V = np.load(f'{data_dir}/{c_name}/{m_name}/{t_name}/V.npy')
    
    A_norm = np.linalg.norm(A, 'fro')
    UV_norm = np.linalg.norm(U @ V.T, 'fro')
    
    # Single plot: Objective function and norms, followed by timing data
    fig = go.Figure()
    
    # Add traces for objective function and norms
    for col in df.columns:
        if col in remove_col:
            continue
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

    # Add the A norm as a horizontal line trace (this will appear in the legend)
    if not ('A_norm' in remove_col):
        fig.add_trace(
            go.Scatter(
                x=[df.index.min(), df.index.max()],
                y=[A_norm, A_norm],
                mode='lines',
                name=f"A_norm",
                line=dict(
                    color='red',
                    width=2,
                    dash='dash'
                ),
                showlegend=True
            )
        )
    if not ('UV_norm' in remove_col):
        fig.add_trace(
            go.Scatter(
                x=[df.index.min(), df.index.max()],
                y=[UV_norm, UV_norm],
                mode='lines',
                name=f"UV_norm",
                line=dict(
                    color='cyan',
                    width=2,
                    dash='dash'
                ),
                showlegend=True
            )
        )

    # Update layout for the figure
    fig.update_layout(
        title=title,
        height=fig_size[1],
        width=fig_size[0],
        template="plotly",
        xaxis_title="iteration",
        yaxis=dict(type='log'),
        legend=dict(
            orientation='h',
            y=-0.2,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        font=dict(size=font_size)
    )
    
    # Log scale for axes, if specified
    if logscale[0]:
        fig.update_layout(
            xaxis=dict(type='log')
        )
    if logscale[1]:
        fig.update_layout(
            yaxis=dict(type='log')
        )

    return fig



def plot_agg_global_df(x='m_n', y='k', remove_col=['m', 'n'],
                       dataframe_path='./data/global_data.csv', df=None,
                       filter={}, new_col={}, remove_outliers=0, 
                       logscale=(True, True), fig_size=(1100, 600),
                       title='Title', font_size=12):
    """
    Plots an aggregated global dataframe with various customization options.
    Parameters:
        x (str): Column name to be used for the x-axis.
        y (str): Column name to be used for the y-axis.
        remove_col (list): List of columns to be removed from the dataframe.
        dataframe_path (str): Path to the CSV file containing the global data.
        df (pd.DataFrame, optional): DataFrame to be used instead of loading from a CSV file.
        filter (dict): Dictionary of functions to filter the dataframe.
        new_col (dict): Dictionary of functions to create new columns in the dataframe.
        remove_outliers (int): Number of outliers to remove from the top and bottom of the numerical columns.
        logscale (tuple): Tuple indicating whether to apply log scale to the x and y axes.
        fig_size (tuple): Tuple specifying the width and height of the figure.
        title (str): Title of the plot.
        font_size (int): Font size for the plot text.
    Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure object with the aggregated data plot.
    """
    # Load the global data
    if df is None:
        old_df = pd.read_csv(dataframe_path)
    else:
        old_df = df.copy()
        
    base_size = 16
    step_size = 4
    color_scale = px.colors.qualitative.Set1
    color_scale_2 = px.colors.qualitative.Alphabet
    
    # Apply filters
    for fun in filter:
        try:
            old_df = old_df[filter[fun](old_df[fun])]
        except:
            continue
    for c in new_col:
        old_df[c] = new_col[c](old_df)
    for fun in filter:
        try:
            old_df = old_df[filter[fun](old_df[fun])]
        except:
            continue

    
    cols_to_show = []
    # Filter columns to show
    for col in old_df.columns:   
        
        if col == x or col == y or col in remove_col:
            continue
        else:
            if not pd.api.types.is_numeric_dtype(old_df[col]):
                
                categories = old_df[col].unique()
                if len(categories) > len(color_scale_2):
                    print(col, 'has too many categories')
                    continue
                
                df = old_df.groupby([x, y]).agg(
                    c_col=(col, lambda series: series.mode().iloc[0])
                )
                
                if df['c_col'].nunique() == 1:
                    print(col, 'has only one value: ', df['c_col'].iloc[0])
                    continue
                else:
                    cols_to_show.append(col)
            else:
                
                if old_df[col].nunique() == 1:
                    print(col, 'has only one value:', old_df[col].iloc[0])
                    continue

                
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
            if len(categories) <= len(color_scale):
                colormap = color_scale[:len(categories)]
            else:
                colormap = color_scale_2[:len(categories)]
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
            df = old_df.drop(old_df[col].nlargest(remove_outliers).index.union(old_df[col].nsmallest(remove_outliers).index))
            
            df = df.groupby([x, y]).agg(
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
            
            # Applicazione del logaritmo e normalizzazione
            ser = np.log1p(df['mean'])  # np.log1p(x) = log(1 + x), più stabile per valori vicini a 0
            ser_norm = (ser - ser.min()) / (ser.max() - ser.min())  # Normalizzazione tra 0 e 1
            df['color'] = ser_norm

            # Tick labels per distinguere meglio i valori tra 0 e 1
            tickvals = np.linspace(0, 1, 6)  # Valori per i tick nella scala normalizzata
            ticklabels = [f'{val:.3f}' for val in (np.expm1(tickvals * (ser.max() - ser.min()) + ser.min()))]
            
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
    
    # Update layout with titles, dimensions, and font size
    fig.update_layout(
        title=title + f' ({len(old_df)} total runs)',
        height=fig_size[1],
        width=fig_size[0],
        template="plotly",
        xaxis_title=x,
        yaxis_title=y,
        font=dict(size=font_size)
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

