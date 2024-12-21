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
    
def resize_image(image, new_height, new_width):
    """
    Resize a grayscale image represented as a float64 matrix.

    Parameters:
        image (numpy.ndarray): Input matrix representing the grayscale image.
        new_height (int): Desired height of the output image.
        new_width (int): Desired width of the output image.

    Returns:
        numpy.ndarray: Resized image as a float64 matrix.
    """
    # Original dimensions
    orig_height, orig_width = image.shape
    
    # Create an output matrix with the desired dimensions
    resized_image = np.zeros((new_height, new_width), dtype=np.float64)
    
    # Calculate scaling factors
    scale_x = orig_width / new_width
    scale_y = orig_height / new_height

    for i in range(new_height):
        for j in range(new_width):
            # Map the pixel in the output image back to the input image
            x = j * scale_x
            y = i * scale_y
            
            # Find the coordinates of the surrounding pixels
            x0 = int(np.floor(x))
            x1 = min(x0 + 1, orig_width - 1)
            y0 = int(np.floor(y))
            y1 = min(y0 + 1, orig_height - 1)
            
            # Interpolation weights
            wx = x - x0
            wy = y - y0
            
            # Bilinear interpolation
            top = (1 - wx) * image[y0, x0] + wx * image[y0, x1]
            bottom = (1 - wx) * image[y1, x0] + wx * image[y1, x1]
            resized_image[i, j] = (1 - wy) * top + wy * bottom
    
    return resized_image


def plot_dataframe(c_name, m_name, t_name, plot_time=False):

    # Load the data
    df = pd.read_csv(f'./data/test/{c_name}/{m_name}/{t_name}/data.csv')
    df['error'] = df['obj_fun'] - df['obj_fun'].min()
    
    if plot_time:
        # Create a subplot with 2 rows and 1 column
        fig = make_subplots(rows=2, cols=1, subplot_titles=["Objective Function and Norms", "Timing Data"])

        # First plot: Objective function and norms
        for col in ['obj_fun', 'U_norm', 'V_norm', 'error']:#, 'UV_norm']:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode='lines',
                    name=col,
                    hoverinfo='text',
                    text=[f"{col}: {value}<br>iteration_id: {iter}" for value, iter in zip(df[col], df['iteration_id'])]
                ),
                row=1, col=1
            )

        # Second plot: Timing data
        for col in ['qr_time', 'manip_time', 'bw_time']:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode='lines',
                    name=col,
                    hoverinfo='text',
                    text=[f"{col}: {value}<br>iteration_id: {iter}" for value, iter in zip(df[col], df['iteration_id'])]
                ),
                row=2, col=1
            )

        # Update layout for the entire figure
        fig.update_layout(
            height=800,  # Adjust the height to fit the two plots
            width=1050,
            template="plotly",
            xaxis_title="Iteration",
            yaxis=dict(type='log'),
            legend=dict(
                orientation='h',
                y=-0.2,
                x=0.5,
                xanchor='center',
                yanchor='top'
            )
        )

        return fig

    else:
        # Single plot: Objective function and norms, followed by timing data
        fig = go.Figure()

        # Add traces for objective function and norms
        for col in ['obj_fun', 'U_norm', 'V_norm', 'error', 'qr_time','manip_time','bw_time',]:#, 'UV_norm']:
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
            height=500,
            width=1050,
            template="plotly",
            xaxis_title="Iteration",
            yaxis=dict(type='log'),
            legend=dict(
                orientation='h',
                y=-0.2,
                x=0.5,
                xanchor='center',
                yanchor='top'
            )
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
 
def compute_global_stats_df():
    main_folder = Path("./data/test")
    global_df = {
        'c_name':[], 'm_name':[], 't_name':[], 
        'init_method':[], 'epsilon':[],
        'pc_name':[], 'date':[],
        'm':[], 'n':[], 'k':[], 
        'iteration':[], 'exec_time':[], 
        'qr_time':[], 'manip_time':[], 'bw_time':[],
        'obj_fun':[], # 'UV_norm':[],
        'U_norm':[], 'V_norm':[]
    }
    
    # Iterate over all directories and subdirectories
    for subfolder in main_folder.rglob('*'):
        # Check if the path is a directory and is at the third level
        if subfolder.is_dir() and len(subfolder.relative_to(main_folder).parts) == 3:
            dummy_df = pd.read_csv((subfolder / 'data.csv').absolute())
            with open(subfolder / 'input_values.json', 'r') as f:
                input_values = json.loads(f.read())
            with open(subfolder / 'pc_info.json', 'r') as f:
                pc_info = json.loads(f.read())
                
            global_df['c_name'].append(input_values['c_name'])
            global_df['m_name'].append(input_values['m_name'])
            global_df['t_name'].append(input_values['t_name'])

            global_df['init_method'].append(input_values['init_method'])
            global_df['epsilon'].append(input_values['epsilon'])
            
            global_df['pc_name'].append(pc_info['pc_name'])
            global_df['date'].append(input_values['date'])
            
            global_df['iteration'].append(dummy_df['iteration_id'].values[-1])
            
            global_df['qr_time'].append(np.sum(dummy_df['qr_time'].values))
            global_df['manip_time'].append(np.sum(dummy_df['manip_time'].values))
            global_df['bw_time'].append(np.sum(dummy_df['bw_time'].values))
            global_df['exec_time'].append(global_df['qr_time'][-1] + global_df['manip_time'][-1] + global_df['bw_time'][-1])
    
            global_df['obj_fun'].append(dummy_df['obj_fun'].values[-1])
            # global_df['UV_norm'].append(dummy_df['UV_norm'].values[-1])
            global_df['U_norm'].append(dummy_df['U_norm'].values[-1])
            global_df['V_norm'].append(dummy_df['V_norm'].values[-1])
            
            A = np.load((subfolder / 'A.npy').absolute())
            U = np.load((subfolder / 'U.npy').absolute())
            
            global_df['m'].append(A.shape[0])
            global_df['n'].append(A.shape[1])
            global_df['k'].append(U.shape[1])
    
    global_df = pd.DataFrame(global_df)       
    global_df.to_csv('./data/global_data.csv', index=False)
    
def keep_n_files(folder, n):
    # Convert folder to a Path object
    folder_path = Path(folder)
    
    # Get a list of all files in the folder (excluding directories)
    files = [f for f in folder_path.iterdir() if f.is_file()]

    # If the folder contains less than or equal to n files, no need to delete anything
    if len(files) <= n:
        print(f"The folder contains {len(files)} files, which is less than or equal to {n}. No files need to be removed.")
        return

    # Select n random files to keep
    files_to_keep = random.sample(files, n)

    # Remove all files except the ones selected
    for file in files:
        if file not in files_to_keep:
            file.unlink()  # Delete the file
            print(f"File removed: {file.name}")

    print(f"{n} random files have been kept. The others have been deleted.")
    
def load_matrices(c_name, m_name, t_name, matrices={'U', 'V', 'A'}):
    ret = {}
    for el in matrices:
        ret[el] = np.load(f'./data/test/{c_name}/{m_name}/{t_name}/{el}.npy')
    return ret