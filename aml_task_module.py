# -*- coding: utf-8 -*-
"""
AML_Task_Module
"""

# Data wranlging
import kagglehub
import pandas as pd
import numpy as np
import networkx as nx
import math
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.offline as py
# Modeling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
from sklearn.base import clone
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score
import xgboost as xgb


def plot_ego_graph(G, classes_df, seed_idx=1, radius=2):
    """
    Plot the ego graph around an illicit transaction.

    Parameters:
    - G: networkx Graph
    - classes_df: DataFrame containing 'txId' and 'class' columns
    - seed_idx: Index of illicit transaction to use as seed (default: 1)
    - radius: Neighborhood radius around the seed node (default: 2)
    """

    # Get an illicit tx_id
    illicit_ids = classes_df[classes_df['class'] == '1']['txId'].values
    seed_id = illicit_ids[seed_idx]

    # Get a small neighborhood around it
    ego = nx.ego_graph(G, seed_id, radius=radius)

    # Spring layout for ego graph
    pos = nx.spring_layout(ego, seed=42)

    # Edge coordinates
    edge_x = []
    edge_y = []
    for edge in ego.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    # Node coordinates and colors
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    for node in ego.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
        node_colors.append('red' if node == seed_id else 'skyblue')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=node_colors,
            size=10,
            line_width=2
        )
    )
    node_trace.text = node_text

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=f"{radius}-Hop Neighborhood of Illicit Transaction {seed_id}",
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[dict(
                        showarrow=True,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002
                    )],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                 )
    fig.show()


def plot_class_distribution(classes_df, print_counts=True):
    """
    Plot the distribution of transaction classes.

    Parameters:
    - classes: DataFrame containing a 'class' column
    - print_counts: Whether to print the class counts (default: True)
    """

    if print_counts:
        print(classes_df['class'].value_counts())

    # Set the seaborn style
    sns.set(style='whitegrid')

    # Count classes
    class_counts = classes_df['class'].value_counts()

    # Create the plot
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')

    # Add title and axis labels
    plt.title('Transaction Class Distribution')
    plt.xlabel('Class (0 = Unknown, 1 = Illicit, 2 = Licit)')
    plt.ylabel('Count')

    # Show the plot
    plt.show()


def plot_transactions_over_time(features):
    """
    Plot the number of transactions over time by label category.

    Parameters:
    - features: DataFrame containing 'time_step', 'class', and 'txId' columns
    """

    # Set Seaborn whiteboard style
    sns.set_style('whitegrid')

    # Group transactions by time_step and class
    total_txs = features.groupby('time_step').count()
    illicit_txs = features[features['class'] == '1'].groupby('time_step').count()
    licit_txs = features[features['class'] == '2'].groupby('time_step').count()
    unknown_txs = features[features['class'] == '0'].groupby('time_step').count()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.title('Number of Transactions by Time Step', fontsize=16)
    plt.plot(total_txs['txId'], color='gray', label='Total', linewidth=2)
    plt.plot(illicit_txs['txId'], color='orangered', label='Illicit', linewidth=2)
    plt.plot(licit_txs['txId'], color='olivedrab', label='Licit', linewidth=2)
    plt.plot(unknown_txs['txId'], color='steelblue', label='Unknown', linewidth=2)

    # Label axes
    plt.xlabel('Time Step')
    plt.ylabel('Number of Transactions')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


def plot_transaction_labels_per_timestep(features):
    """
    Plot stacked bar chart of transaction labels per time step.

    Parameters:
    - features: DataFrame containing 'time_step' and 'class' columns
    """

    # Group transactions by time_step and class
    grouped_class = (
        features[['time_step', 'class']]
        .groupby(['time_step', 'class'])
        .size()
        .to_frame(name='count')
        .reset_index()
    )

    timesteps = list(range(1, 50))

    fig = go.Figure(data=[
        go.Bar(
            name="Unknown (unlabelled)",
            x=timesteps,
            y=grouped_class[grouped_class['class'] == '0']['count'],
            marker=dict(color='steelblue', line=dict(color='steelblue', width=1))
        ),
        go.Bar(
            name="Licit (non-fraud)",
            x=timesteps,
            y=grouped_class[grouped_class['class'] == '2']['count'],
            marker=dict(color='olivedrab', line=dict(color='olivedrab', width=1))
        ),
        go.Bar(
            name="Illicit (fraud)",
            x=timesteps,
            y=grouped_class[grouped_class['class'] == '1']['count'],
            marker=dict(color='orangered', line=dict(color='orangered', width=1))
        )
    ])

    fig.update_layout(
        barmode='stack',
        xaxis_title="Time Step",
        yaxis_title="Number of Transactions",
        title="Transaction Labels per Time Step",
        titlefont_size=16,
        legend_title_text='Transaction Label',
        margin=dict(b=20, l=40, r=20, t=60)
    )

    fig.show()


def plot_degree_distribution(G):
    """
    Plot the degree distribution of a transaction graph.

    Parameters:
    - G: NetworkX graph
    """
    degrees = list(dict(G.degree()).values())

    # Set Seaborn whitegrid style
    sns.set(style='whitegrid')

    plt.figure(figsize=(6, 4))
    plt.hist(degrees, bins=100, log=True, color='navy')
    plt.title('Transaction Graph Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Log Count')
    plt.tight_layout()
    plt.show()


def plot_degree_distribution_by_class(G, classes):
    """
    Plot degree distributions by class (Illicit, Licit, Unknown) for the transaction graph.

    Parameters:
    - G: NetworkX graph
    - classes: DataFrame with 'txId' and 'class' columns
    """
    # Compute node degrees
    node_degrees = dict(G.degree())

    # Merge degrees with class labels
    class_degrees = classes.copy()
    class_degrees['degree'] = class_degrees['txId'].map(node_degrees)
    class_degrees = class_degrees.dropna(subset=['degree'])

    # Keep only valid class labels
    class_degrees = class_degrees[class_degrees['class'].isin(['0', '1', '2'])]

    # Convert types
    class_degrees['class'] = class_degrees['class'].astype(int)
    class_degrees['degree'] = class_degrees['degree'].astype(int)

    # Prepare degree lists
    degree_illicit = class_degrees[class_degrees['class'] == 1]['degree'].tolist()
    degree_licit = class_degrees[class_degrees['class'] == 2]['degree'].tolist()
    degree_unknown = class_degrees[class_degrees['class'] == 0]['degree'].tolist()

    # Set style
    sns.set(style='whitegrid')

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    axes[0].hist(degree_illicit, bins=100, log=True, color='orangered')
    axes[0].set_title('Illicit Transactions')
    axes[0].set_xlabel('Degree')
    axes[0].set_ylabel('Log Count')

    axes[1].hist(degree_licit, bins=100, log=True, color='olivedrab')
    axes[1].set_title('Licit Transactions')
    axes[1].set_xlabel('Degree')

    axes[2].hist(degree_unknown, bins=100, log=True, color='steelblue')
    axes[2].set_title('Unknown Transactions')
    axes[2].set_xlabel('Degree')

    fig.suptitle('Transaction Graph Degree Distribution by Class', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_average_degree_per_class(G, classes):
    """
    Plot the average node degree per transaction class.

    Parameters:
    - G: NetworkX graph
    - classes: DataFrame with 'txId' and 'class' columns
    """
    # Set style
    sns.set(style='whitegrid')

    # Compute node degrees
    node_degrees = dict(G.degree())

    # Merge degrees with class labels
    class_degrees = classes.copy()
    class_degrees['degree'] = class_degrees['txId'].map(node_degrees)
    class_degrees = class_degrees.dropna(subset=['degree'])

    # Remove invalid class labels (keep only 0, 1, 2)
    class_degrees = class_degrees[class_degrees['class'].isin(['0', '1', '2'])]

    # Average degree per class
    avg_degree_per_class = class_degrees.groupby('class')['degree'].mean()

    # Plot
    plt.figure(figsize=(6, 4))
    avg_degree_per_class.plot(kind='bar', color=['orangered', 'olivedrab', 'steelblue'])
    plt.title('Average Degree per Class', fontsize=14)
    plt.xlabel('Transaction Class')
    plt.ylabel('Average Degree')
    plt.xticks([0, 1, 2], ['Illicit', 'Licit', 'Unknown'], rotation=0)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_time_step_transactions(features, edges, time_step=29):
    """
    Plot the transaction graph for all transactions in a given time step.

    Parameters:
    - features: DataFrame containing transaction features (must have 'txId', 'time_step', 'class')
    - edges: DataFrame containing transaction edges ('txId1', 'txId2')
    - time_step: int, the time step to visualize (default=29)
    """

    # Filter transactions and edges for the given time step
    all_ids = features[features['time_step'] == time_step]['txId']
    short_edges = edges[edges['txId1'].isin(all_ids)]

    # Build graph
    graph = nx.from_pandas_edgelist(short_edges, source='txId1', target='txId2', create_using=nx.DiGraph())

    # Layout
    pos = nx.spring_layout(graph, seed=42)

    # Edge trace
    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='blue'),
        hoverinfo='none',
        mode='lines'
    )

    # Node trace
    node_x, node_y, node_text = [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))

    node_classes = features[features['txId'].isin(graph.nodes())]['class'].astype(int)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=node_classes,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Transaction Type',
                xanchor='left',
                titleside='right',
                tickmode='array',
                tickvals=[0, 1, 2],
                ticktext=['Unknown', 'Illicit', 'Licit']
            ),
            line_width=2
        ),
        text=node_text
    )

    # Figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"All Transactions in Time Step {time_step}",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                showarrow=True,
                xref="paper", yref="paper",
                x=0.005, y=-0.002
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    fig.show()


def plot_illicit_transactions_at_time_step(features, edges, time_step=29):
    """
    Plot the graph of illicit transactions for a given time step.

    Parameters:
    - features: DataFrame with transaction features (must have 'txId', 'time_step', 'class')
    - edges: DataFrame with transaction edges ('txId1', 'txId2')
    - time_step: int, time step to visualize (default=29)
    """

    # Filter illicit transaction IDs at the given time step
    illicit_ids = features[(features['time_step'] == time_step) & (features['class'] == '1')]['txId']
    short_edges = edges[edges['txId1'].isin(illicit_ids.tolist())]

    # Build graph
    graph = nx.from_pandas_edgelist(short_edges, source='txId1', target='txId2', create_using=nx.DiGraph())

    if graph.number_of_nodes() == 0:
        print(f"No illicit transactions found at time step {time_step}.")
        return

    # Layout
    pos = nx.spring_layout(graph, seed=42)

    # Edge trace
    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='blue'),
        hoverinfo='none',
        mode='lines'
    )

    # Node trace
    node_x, node_y, node_text = [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))

    node_classes = features[features['txId'].isin(graph.nodes())]['class'].astype(int)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=node_classes,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Transaction Type',
                xanchor='left',
                titleside='right',
                tickmode='array',
                tickvals=[0, 1, 2],
                ticktext=['Unknown', 'Illicit', 'Licit']
            ),
            line_width=2
        ),
        text=node_text
    )

    # Figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Illicit Transactions at Time Step {time_step}",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                showarrow=True,
                xref="paper", yref="paper",
                x=0.005, y=-0.002
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    fig.show()


def plot_heatmap_and_high_corr_pairs(features, corr_threshold=0.9):
    """
    Plots a heatmap of feature correlations and finds pairs of highly correlated features.

    Parameters:
    - features: DataFrame containing the feature data.
    - corr_threshold: float, correlation threshold for identifying high correlations (default=0.9).

    Returns:
    - high_corr_pairs: List of tuples containing highly correlated feature pairs.
    """

    # 1. Plot heatmap of correlations
    corr = features.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', annot=False, fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

    # 2. Find highly correlated pairs
    high_corr_pairs = [
        (col1, col2) for col1 in corr.columns for col2 in corr.columns
        if col1 != col2 and abs(corr.loc[col1, col2]) > corr_threshold
    ]

    return high_corr_pairs


def drop_highly_correlated_features(df, threshold=0.9):
    corr_matrix = df.corr().abs()  # Absolute correlation
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df_reduced = df.drop(columns=to_drop)

    return df_reduced, to_drop


def build_transaction_graph_and_compute_features(train_data, test_data, edges, alpha=0.85, max_iter=1000):
    """
    Build transaction graphs for training and testing data, compute graph properties, and return feature dataframes.

    Parameters:
    - train_data: DataFrame, training dataset (timesteps < 39).
    - test_data: DataFrame, testing dataset (timesteps >= 39).
    - edges: DataFrame, edges in the graph.
    - alpha: float, damping factor for PageRank (default=0.85).
    - max_iter: int, maximum iterations for eigenvector centrality (default=1000).

    Returns:
    - graph_features_train: DataFrame, graph properties for training data.
    - graph_features_test: DataFrame, graph properties for testing data.
    """

    # Build train and test graphs using the respective edges
    train_edges = edges[edges['txId1'].isin(train_data['txId']) & edges['txId2'].isin(train_data['txId'])]
    test_edges = edges[edges['txId1'].isin(test_data['txId']) & edges['txId2'].isin(test_data['txId'])]

    G_train = nx.from_pandas_edgelist(train_edges, source='txId1', target='txId2', create_using=nx.DiGraph())
    G_test = nx.from_pandas_edgelist(test_edges, source='txId1', target='txId2', create_using=nx.DiGraph())

    # Function to compute graph properties
    def compute_graph_properties(G):
        degree_dict = dict(G.degree())
        in_degree_dict = dict(G.in_degree())
        out_degree_dict = dict(G.out_degree())
        clustering_dict = nx.clustering(G.to_undirected())  # Clustering needs undirected graph
        pagerank_dict = nx.pagerank(G, alpha=alpha)
        eigenvector_dict = nx.eigenvector_centrality(G, max_iter=max_iter)

        return degree_dict, in_degree_dict, out_degree_dict, clustering_dict, pagerank_dict, eigenvector_dict

    # Compute properties for train and test graphs
    degree_dict_train, in_degree_dict_train, out_degree_dict_train, clustering_dict_train, pagerank_dict_train, eigenvector_dict_train = compute_graph_properties(G_train)
    degree_dict_test, in_degree_dict_test, out_degree_dict_test, clustering_dict_test, pagerank_dict_test, eigenvector_dict_test = compute_graph_properties(G_test)

    # Convert to DataFrame for easy manipulation
    graph_features_train = pd.DataFrame({
        'txId': list(degree_dict_train.keys()),
        'degree': list(degree_dict_train.values()),
        'in_degree': list(in_degree_dict_train.values()),
        'out_degree': list(out_degree_dict_train.values()),
        'clustering_coef': [clustering_dict_train.get(node, 0) for node in degree_dict_train.keys()],
        'pagerank': [pagerank_dict_train.get(node, 0) for node in degree_dict_train.keys()],
        'eigenvector_centrality': [eigenvector_dict_train.get(node, 0) for node in degree_dict_train.keys()]
    })

    graph_features_test = pd.DataFrame({
        'txId': list(degree_dict_test.keys()),
        'degree': list(degree_dict_test.values()),
        'in_degree': list(in_degree_dict_test.values()),
        'out_degree': list(out_degree_dict_test.values()),
        'clustering_coef': [clustering_dict_test.get(node, 0) for node in degree_dict_test.keys()],
        'pagerank': [pagerank_dict_train.get(node, 0) for node in degree_dict_test.keys()],  # Use train pagerank as a fallback
        'eigenvector_centrality': [eigenvector_dict_test.get(node, 0) for node in degree_dict_test.keys()]
    })

    return graph_features_train, graph_features_test


def plot_confusion_matrix(cm, title='Confusion Matrix', labels=['Licit', 'Illicit']):
    """
    Plots a confusion matrix using seaborn heatmap.

    Parameters:
    - cm: Confusion matrix (2x2 numpy array or similar).
    - title: The title for the plot (default 'Confusion Matrix').
    - labels: List of labels for the x and y axes (default ['Licit', 'Illicit']).
    """
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()
