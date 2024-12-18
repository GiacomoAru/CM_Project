# from math_utils import *
from PIL import Image
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
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

'''
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
    plt.show()'''
    
#TODO: test this
def show_grayscale_images(matrices, cols=3, names=None):
    """
    Displays multiple NumPy matrices as grayscale images in a grid format using Plotly Express.
    
    :param matrices: List of NumPy matrices to display as images.
    :param cols: Number of columns in the grid (default is 3).
    :param names: List of titles for each image (default is None, and titles will be auto-generated).
    """
    if not names:
        names = [f"Image {i + 1}" for i in range(len(matrices))]
    
    # Prepare a list of data dictionaries for Plotly
    data = []
    for idx, (matrix, name) in enumerate(zip(matrices, names)):
        data.append({
            "Image": name,
            "z": matrix
        })
    
    # Create a DataFrame for easier handling in Plotly
    df = pd.DataFrame(data)
    
    # Create a subplot figure with the specified number of columns
    fig = px.imshow(
        [np.array(matrix, dtype=np.uint8) for matrix in matrices],
        facet_col=0,
        facet_col_wrap=cols,
        labels=dict(color="Intensity"),
        color_continuous_scale="gray"
    )
    
    # Add titles to the subplots
    for i, name in enumerate(names):
        fig.layout.annotations[i]["text"] = name
    
    # Update layout to adjust spacing and aesthetics
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=(len(matrices) // cols + 1) * 200,
        width=cols * 200,
        coloraxis_showscale=False
    )
    
    # Show the figure
    fig.show()

 
 

def plot_dataframe(test_name, plot_time=False):
    """
    Plot a DataFrame from a CSV file located in './data/test/' + test_name + '/data.csv'.

    :param test_name: Name of the test (used to locate the CSV file).
    :param plot_time: If True, create a subplot with time-related columns. Otherwise, plot objective function and norms.
    :return: A Plotly figure.
    """
    # Load the data
    df = pd.read_csv(f'./data/test/{test_name}/data.csv')

    if plot_time:
        # Create a subplot with 2 rows and 1 column
        fig = make_subplots(rows=2, cols=1, subplot_titles=["Objective Function and Norms", "Timing Data"])

        # First plot: Objective function and norms
        for col in ['obj_fun', 'U_norm', 'V_norm']:#, 'UV_norm']:
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
        for col in ['obj_fun', 'U_norm', 'V_norm']:#, 'UV_norm']:
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

def plot_global_df(x='m_n', y='k', filter={}):
    
    df = pd.read_csv('./data/global_data.csv')
    for f in filter:
        df = df[df[f] == filter[f]]
        
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
                    text=[f"name={n}<br>m*n={mn}<br>k={k}<br>{col}={c}" for n, mn, k, c in zip(df['test_name'], df['m_n'], df['k'], df[col])])
                    
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
            #yaxis=dict(type='log'),
        )


    return fig

def compute_global_stats_df():
    main_folder = Path("./data/test")
    global_df = {
        'test_class':[], 'test_name':[], 'init_method':[], 
        'pc_name':[], 'date':[],
        'm':[], 'n':[], 'k':[], 
        'iteration':[], 'exec_time':[], 
        'qr_time':[], 'manip_time':[], 'bw_time':[],
        'obj_fun':[], # 'UV_norm':[],
        'U_norm':[], 'V_norm':[]
    }
    
    for subfolder in main_folder.iterdir():
        if subfolder.is_dir(): 
            dummy_df = pd.read_csv((subfolder / 'data.csv').absolute())
            with open(subfolder / 'input_values.json', 'r') as f:
                input_values = json.loads(f.read())
            with open(subfolder / 'pc_info.json', 'r') as f:
                pc_info = json.loads(f.read())
                
            global_df['test_class'].append(input_values['test_class'])
            global_df['test_name'].append(input_values['test_name'])
            global_df['init_method'].append(input_values['init_method'])
            
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
    
def load_matrices(test_name, matrices={'U', 'V', 'A'}):
    ret = {}
    for el in matrices:
        ret[el] = np.load(f'./data/test/{test_name}/' + el + '.npy')
    return ret