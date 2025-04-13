import numpy as np
import torch
import pandas as pd
from typing import Dict, Generator, Tuple, List
from sklearn.preprocessing import OneHotEncoder, Normalizer
from torch_geometric.data import HeteroData
import networkx as nx
from matplotlib import pyplot as plt
import torch

def create_sparse_identity_tensor(df, column_name):
    """
    Create a sparse identity tensor based on unique values of a specified column in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing the column of interest.
        column_name (str): Name of the column to process.

    Returns:
        torch.sparse_coo_tensor: Sparse identity tensor.
    """
    num_unique_nodes = df[column_name].dropna().nunique()
    indices = torch.arange(num_unique_nodes, dtype=torch.long).repeat(2, 1)
    values = torch.ones(num_unique_nodes, dtype=torch.float32)
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_unique_nodes, num_unique_nodes),
        dtype=torch.float32
    )
    
    
def numpy_to_sparse_tensor(matrix: np.ndarray) -> torch.Tensor:
    """
    Converts a NumPy array to a PyTorch sparse tensor in COO format.

    Args:
        matrix (np.ndarray): The input NumPy array.

    Returns:
        torch.Tensor: A PyTorch sparse tensor.
    """
    dense_tensor = torch.from_numpy(matrix).float()
    sparse_tensor = dense_tensor.to_sparse()
    return sparse_tensor
    
    
def generate_node_features(
    df: pd.DataFrame,
    col_mapping: Dict[str, str],
    event_id_col: str,
    activity_col: str,
    hour_cols: List[str],
    tertile_cols: List[str],
    time_since_first_day_col: str,
    days_from_lc_start_col: str
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """
    Generates feature matrices for different node types based on the provided column mapping.

    For each node type in `col_mapping`, this function yields a tuple containing the node type
    and its corresponding feature matrix. For most node types, the feature matrix is a sparse
    identity tensor created using `create_sparse_identity_tensor`. For the "event" node type,
    a custom feature matrix is created using `create_event_feature_matrix`, which is then
    converted to a PyTorch tensor.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        col_mapping (Dict[str, str]): A dictionary mapping node types to their corresponding
            columns in the DataFrame.
        event_id_col (str): The column name for event IDs, used when node_type is "event".
        activity_col (str): The column name for activity names, used for one-hot encoding
            when node_type is "event".
        hour_cols (List[str]): A list of column names for hour features, used when
            node_type is "event".
        tertile_cols (List[str]): A list of column names for tertile features, used when
            node_type is "event".
        time_since_first_day_col (str): The column name for time since the first day, used
            when node_type is "event".
        days_from_lc_start_col (str): The column name for days from the lifecycle start,
            used when node_type is "event".

    Yields:
        Tuple[str, torch.Tensor]: A tuple containing the node type and its feature matrix.

    Notes:
        - The DataFrame must contain the columns specified in `event_id_col`, `activity_col`,
          `hour_cols`, `tertile_cols`, `time_since_first_day_col`, and `days_from_lc_start_col`
          when "event" is in `col_mapping`.
        - The feature matrix for "event" nodes is converted from a NumPy array to a PyTorch tensor.
    """
    for node_type, col in col_mapping.items():
        if node_type == "event":
            feature_matrix = create_event_feature_matrix(
                df=df,
                event_id_col=event_id_col,
                activity_col=activity_col,
                hour_cols=hour_cols,
                tertile_cols=tertile_cols,
                time_since_first_day_col=time_since_first_day_col,
                days_from_lc_start_col=days_from_lc_start_col
            )
            normalizer = Normalizer(norm='l2')
            # Normalize each event node's feature vector independently
            feature_matrix = normalizer.transform(feature_matrix)
            
            feature_matrix = numpy_to_sparse_tensor(feature_matrix)  # Convert to sparse torch.Tensor
        else:
            feature_matrix = create_sparse_identity_tensor(df, col)
        yield node_type, feature_matrix

def create_event_feature_matrix(
    df: pd.DataFrame,
    event_id_col: str,
    activity_col: str,
    hour_cols: List[str],
    tertile_cols: List[str],
    time_since_first_day_col: str,
    days_from_lc_start_col: str
) -> np.ndarray:
    """
    Creates a node feature matrix for events where each row is a feature vector
    containing concatenated features:
    a. One-hot encoding of the activity column.
    b. Hour columns (e.g., hour_0 to hour_23).
    c. Tertile columns (e.g., tertile_12am-8am, tertile_8am-4pm, tertile_4pm-12am).
    d. Time since first day column.
    e. Days from LC start column.

    Args:
        df (pd.DataFrame): Input dataframe containing event data.
        event_id_col (str): Name of the column containing event IDs.
        activity_col (str): Name of the column containing activity names.
        hour_cols (List[str]): List of column names for hour features.
        tertile_cols (List[str]): List of column names for tertile features.
        time_since_first_day_col (str): Name of the column for time since first day.
        days_from_lc_start_col (str): Name of the column for days from LC start.

    Returns:
        np.ndarray: A numpy array with shape (n_events, n_features), where n_events is the
                    number of unique events.

    Note:
        - The list of all possible activities is computed inside the function using
          df[activity_col].unique(). Ensure that the input dataframe contains all
          possible activities if consistent encoding across datasets is required.
    """
    # Compute all possible activities from the dataframe
    all_possible_activities = df[activity_col].unique()

    # Create the one-hot encoder for activities
    activity_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore',
        categories=[all_possible_activities]
    )

    # Get unique events based on the first occurrence of each event ID
    unique_events = df.drop_duplicates(event_id_col)

    # One-hot encode the activity column for unique events
    activity_encoded = activity_encoder.fit_transform(unique_events[[activity_col]])

    # Extract other features directly from unique_events as numpy arrays
    hour_features = unique_events[hour_cols].values
    tertile_features = unique_events[tertile_cols].values
    time_features = unique_events[[time_since_first_day_col, days_from_lc_start_col]].values

    # Concatenate all features horizontally into a single numpy array
    feature_matrix = np.hstack([
        activity_encoded,
        hour_features,
        tertile_features,
        time_features
    ])

    # Print explanatory shapes for debugging
    print(f"Number of unique events processed: {len(unique_events)}")
    print(f"Shape of activity one-hot encoded features: {activity_encoded.shape}")
    print(f"Shape of hour features: {hour_features.shape}")
    print(f"Shape of tertile features: {tertile_features.shape}")
    print(f"Shape of time features: {time_features.shape}")
    print(f"Shape of full event feature matrix: {feature_matrix.shape}")

    return feature_matrix
    
def create_heterodata_nodes(node_features_dict: dict) -> HeteroData:
    """
    Populate a PyG HeteroData object with x-attributes from a given dictionary
    of node features. Removes any existing 'x' node type.

    Args:
        node_features_dict (dict): A dictionary of node-type -> feature array.
            e.g., {
                "event": ...,
                "employee": ...,
                "order": ...,
                "tu": ...,
                ...
            }

    Returns:
        HeteroData: The same HeteroData object, updated with .x for each node type.
    """
    data = HeteroData()
    # Remove top-level "x" store if it exists
    if 'x' in data:
        del data['x']

    # For each node type in the dictionary, set data[node_type].x
    for node_type, feature_tensor in node_features_dict.items():
        data[node_type].x = feature_tensor

    return data


def create_node_index_mappings(df: pd.DataFrame, node_types: list) -> dict:
    """
    For each node type (i.e., a column name in the DataFrame), drop NaN values,
    retrieve unique elements (possibly exploding lists if your utility function does so),
    then build a {index -> node_value} mapping.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns for each node type.
        node_types (list): A list of column names, each representing a node type.

    Returns:
        dict: A dictionary where:
              - keys are node type strings (e.g. "TU")
              - values are dictionaries of {index -> unique_node_value}
    """
    node_to_index_dict = {}

    for node_type in node_types:
        # Drop NaN values in this column
        column_series = df[node_type].dropna()

        # Extract unique elements (exploding if needed)
        unique_nodes = column_series.unique()

        # Build an index->node mapping
        # Example: {0: <first node>, 1: <second node>, ...}
        mapping_dict = {idx: unique_nodes[idx] for idx in range(len(unique_nodes))}

        node_to_index_dict[node_type] = mapping_dict

    return node_to_index_dict


def process_edge_type(
    data,
    edge_type,
    edge_class,
    df,
    col_src,
    col_dst,
    col_batch,
    col_labels,
    src_dict,
    dst_dict,
    train_batches,
    val_batches,
    test_batches,
    directed
):
    """
    Processes a single edge type based on its class and adds edges and masks to the HeteroData object.
    """
    if edge_class == 'E2E':
        # E2E logic remains unchanged...
        edges = []
        labels = []
        train_mask = []
        val_mask = []
        test_mask = []

        # For each unique batch, create edges between consecutive events
        for batch, group in df.drop_duplicates(subset=[col_batch, col_src]).groupby(col_batch):
            events = group[col_src].values
            log_days_to_finish = group[col_labels].values  # edge labels from col_labels

            source = events[:-1]
            target = events[1:]
            edge_labels = log_days_to_finish[:-1]

            batch_edges = np.vstack((source, target))
            edges.append(batch_edges)
            labels.append(edge_labels)

            # Assign masks based on which split the batch belongs to
            if batch in train_batches:
                train_mask.extend([True] * len(source))
                val_mask.extend([False] * len(source))
                test_mask.extend([False] * len(source))
            elif batch in val_batches:
                train_mask.extend([False] * len(source))
                val_mask.extend([True] * len(source))
                test_mask.extend([False] * len(source))
            else:  # test batches
                train_mask.extend([False] * len(source))
                val_mask.extend([False] * len(source))
                test_mask.extend([True] * len(source))

        if edges:
            edge_matrix = np.hstack(edges)         # shape = (2, total_num_edges)
            edge_labels = np.hstack(labels)          # shape = (total_num_edges,)
        else:
            edge_matrix = np.empty((2, 0), dtype=np.int64)
            edge_labels = np.empty(0, dtype=np.float64)
            train_mask = []
            val_mask = []
            test_mask = []

        # Convert to torch tensors and assign to forward edge
        data[edge_type].edge_index = torch.tensor(edge_matrix, dtype=torch.long)
        data[edge_type].edge_labels = torch.tensor(edge_labels, dtype=torch.float)
        data[edge_type].train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data[edge_type].val_mask = torch.tensor(val_mask, dtype=torch.bool)
        data[edge_type].test_mask = torch.tensor(test_mask, dtype=torch.bool)

        # If the edge is undirected, also create the reverse edge
        if not directed:
            reverse_edge_type = (edge_type[2], edge_type[1], edge_type[0])
            reversed_edge_matrix = edge_matrix[[1, 0], :]
            data[reverse_edge_type].edge_index = torch.tensor(reversed_edge_matrix, dtype=torch.long)
            data[reverse_edge_type].edge_labels = torch.tensor(edge_labels, dtype=torch.float)
            data[reverse_edge_type].train_mask = torch.tensor(train_mask, dtype=torch.bool)
            data[reverse_edge_type].val_mask = torch.tensor(val_mask, dtype=torch.bool)
            data[reverse_edge_type].test_mask = torch.tensor(test_mask, dtype=torch.bool)

    else:
        # O2O branch remains unchanged
        if edge_class == 'O2O':
            src_reverse_dict = {v: k for k, v in src_dict.items()}
            dst_reverse_dict = {v: k for k, v in dst_dict.items()}

            filtered_df = df[[col_src, col_dst, col_batch]].dropna().drop_duplicates()
            filtered_df.columns.values[2] = "BATCH"
            col_batch = "BATCH"

            filtered_df['SRC_IDX'] = filtered_df[col_src].map(src_reverse_dict)
            filtered_df['DST_IDX'] = filtered_df[col_dst].map(dst_reverse_dict)

            if filtered_df[['SRC_IDX', 'DST_IDX']].isna().any().any():
                print(f"Warning: Some nodes not found in the dictionaries for edge type {edge_type}!")

        # Updated branch for O2E
        elif edge_class == 'O2E':
            # For O2E, we switched between source and destination:
            # The destination is used directly (col_src) and the source is mapped using col_dst.
            dst_reverse_dict = {v: k for k, v in dst_dict.items()}
            filtered_df = df[[col_src, col_dst, col_batch]].dropna().drop_duplicates()
            filtered_df.columns.values[2] = "BATCH"
            col_batch = "BATCH"

            filtered_df['SRC_IDX'] = filtered_df[col_dst].map(dst_reverse_dict)
            filtered_df['DST_IDX'] = filtered_df[col_src]
            if filtered_df['SRC_IDX'].isna().any():
                print(f"Warning: Some source nodes not found in dictionary for edge type {edge_type}!")

        # Build the edge matrix: shape = (2, num_edges)
        edge_matrix = np.array([filtered_df['SRC_IDX'].values, filtered_df['DST_IDX'].values])

        # Assign split membership: train / val / test
        filtered_df['set_type'] = np.select(
            [
                filtered_df[col_batch].isin(train_batches),
                filtered_df[col_batch].isin(val_batches),
                filtered_df[col_batch].isin(test_batches)
            ],
            ['train', 'val', 'test'],
            default=None
        )

        train_mask = torch.tensor((filtered_df['set_type'] == 'train').values, dtype=torch.bool)
        val_mask   = torch.tensor((filtered_df['set_type'] == 'val').values,   dtype=torch.bool)
        test_mask  = torch.tensor((filtered_df['set_type'] == 'test').values,  dtype=torch.bool)

        # Store forward edge in HeteroData
        data[edge_type].edge_index = torch.tensor(edge_matrix, dtype=torch.long)
        data[edge_type].train_mask = train_mask
        data[edge_type].val_mask = val_mask
        data[edge_type].test_mask = test_mask

        # If the edge is undirected, create the reverse edge
        if not directed:
            reverse_edge_type = (edge_type[2], edge_type[1], edge_type[0])
            reversed_edge_matrix = edge_matrix[[1, 0], :]
            data[reverse_edge_type].edge_index = torch.tensor(reversed_edge_matrix, dtype=torch.long)
            data[reverse_edge_type].train_mask = train_mask
            data[reverse_edge_type].val_mask = val_mask
            data[reverse_edge_type].test_mask = test_mask
            
            
def create_hetero_masks_and_edges(
    df: pd.DataFrame,
    data: HeteroData,
    edge_types: list,
    node_to_index_dict: dict,
    ratio_dict: dict,
    col_batch: str = 'LIFECYCLE_BATCH',
    col_labels: str = 'log_days_to_finish'
) -> HeteroData:
    """
    1) Splits the values in 'col_batch' into train, val, test sets based on ratio_dict.
    2) Iterates over edge_types, calling process_edge_type for each one.
    3) Returns the updated HeteroData object with edge_index and train/val/test masks.

    Args:
        df (pd.DataFrame): The DataFrame containing your graph-related columns.
        data (HeteroData): The PyG HeteroData object to populate with edges.
        edge_types (list): A list of dicts describing each edge's metadata. Example element:
            {
                'edge_type': ('tu', 'O2O', 'product'),  # (src_node_type, qualifier, dst_node_type)
                'col_src': 'TU',
                'col_dst': 'PRODUCT',
                'edge_class': 'O2O',    # or 'O2E' or 'E2E'
                'directed': False       # True or False
            }
        node_to_index_dict (dict): Maps column names -> {index -> node_value} dicts
                                   (produced by create_node_index_mappings).
        ratio_dict (dict): Specifies the train/val/test ratio. e.g. {"train": 0.7, "val": 0.1, "test": 0.2}.
        col_batch (str): The column used for splitting into train/val/test sets. Defaults to 'LIFECYCLE_BATCH'.
        col_labels (str): The column used for E2E edge labels. Defaults to 'log_days_to_finish'.

    Returns:
        HeteroData: The updated HeteroData with edges and masks added.
    """

    # ---------------------
    # Step 1: Split Batches
    # ---------------------
    unique_batches = df[col_batch].dropna().unique()
    total_batches = len(unique_batches)

    # Sort or shuffle if needed:
    # np.random.shuffle(unique_batches)  # only if you want a random split, for example

    train_end = int(total_batches * ratio_dict["train"])
    val_end   = train_end + int(total_batches * ratio_dict["val"])
    # test_end = val_end + int(total_batches * ratio_dict["test"])  # would be total_batches in most cases

    train_batches = set(unique_batches[:train_end])
    val_batches   = set(unique_batches[train_end:val_end])
    test_batches  = set(unique_batches[val_end:])

    print(f"Total Batches: {total_batches}")
    print(f"Train Batches:      {len(train_batches)}")
    print(f"Validation Batches: {len(val_batches)}")
    print(f"Test Batches:       {len(test_batches)}")

    # --------------------------------------------------------
    # Step 2: For Each Edge Type, Process and Add to HeteroData
    # --------------------------------------------------------
    for edge in edge_types:
        edge_type = edge['edge_type']    # (node_src, edge_qualifier, node_dst)
        col_src   = edge['col_src']
        col_dst   = edge['col_dst']
        edge_class = edge['edge_class']
        directed = edge['directed']

        src_dict = node_to_index_dict.get(col_src, {})
        dst_dict = node_to_index_dict.get(col_dst, {})

        # Print or log to see what edges you're creating
        print(f"Creating edges for {edge_type} from columns {col_src} -> {col_dst} [class={edge_class}]")

        process_edge_type(
            data=data,
            edge_type=edge_type,
            edge_class=edge_class,
            df=df,
            col_src=col_src,
            col_dst=col_dst,
            col_batch=col_batch,
            col_labels=col_labels,
            src_dict=src_dict,
            dst_dict=dst_dict,
            train_batches=train_batches,
            val_batches=val_batches,
            test_batches=test_batches,
            directed=directed
        )

    # Optional final validation
    data.validate(raise_on_error=True)
    return data

    
def visualize_heterodata_meta_graph(data, figsize=(14, 14), seed=9, k=3.5, custom_colors=None):
    """
    Visualizes a meta-level graph of node types and edge types from a PyG HeteroData object.
    Each node in the NetworkX plot is a node type, and each directed edge is an edge type.

    Args:
        data (HeteroData): A PyG HeteroData object. We assume data.node_types and data.edge_types
                           are available (PyG >= 2.0).
        figsize (tuple): Figure size for the matplotlib plot.
        seed (int): Random seed passed to NetworkX's spring_layout for reproducibility.
        k (float): The optimal distance between nodes in the layout. Smaller means tighter clusters.
        custom_colors (dict, optional): A dictionary mapping node types to colors.
                                        e.g. {"event": "red", "order": "green"}.
                                        If not provided, a default color cycle is used.

    Returns:
        None. Displays a matplotlib figure showing the meta-graph.
    """

    # 1) Gather all node types and edge types from the HeteroData object
    node_types = list(data.node_types)  # e.g. ['event', 'employee', 'order', ...]
    edge_types = list(data.edge_types)  # e.g. [('event','E2O','order_step'), ...]

    # 2) Initialize a directed graph at the meta-level
    G = nx.DiGraph()

    # 3) Add each node type as a single node in the graph
    for ntype in node_types:
        G.add_node(ntype)

    # 4) Add an edge for each (src_type, relation, dst_type)
    #    We'll store the "relation" in the edge attribute for labeling
    for (src_type, relation, dst_type) in edge_types:
        # relation is typically something like "O2O", "E2O", or "E2E"
        G.add_edge(src_type, dst_type, relation=relation)

    # 5) Define or generate a color for each node type
    if custom_colors is not None:
        # Use user-provided colors for node types that appear; random fallback otherwise
        node_color_map = {}
        for ntype in node_types:
            if ntype in custom_colors:
                node_color_map[ntype] = custom_colors[ntype]
            else:
                # fallback random color
                node_color_map[ntype] = "#%06x" % random.randint(0, 0xFFFFFF)
    else:
        # Default color cycle for up to 12 node types; repeat if more
        default_cycle = [
            "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a",
            "#b15928", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f",
            "#cab2d6", "#ffff99"
        ]
        node_color_map = {}
        for i, ntype in enumerate(node_types):
            node_color_map[ntype] = default_cycle[i % len(default_cycle)]

    # Build a color list for NetworkX
    node_colors = [node_color_map[ntype] for ntype in G.nodes()]

    # 6) Compute layout positions for the meta-nodes
    pos = nx.spring_layout(G, seed=seed, k=k)

    # 7) Draw the figure
    plt.figure(figsize=figsize)

    # Draw nodes (with assigned colors)
    nx.draw_networkx_nodes(
        G, pos, node_size=700, node_color=node_colors
    )

    # Draw edges with arrows
    nx.draw_networkx_edges(
        G, pos, arrowstyle='-|>', arrowsize=20, edge_color='gray'
    )

    # Draw node labels
    nx.draw_networkx_labels(
        G, pos, font_size=10, font_color='black'
    )

    # Draw edge labels (the "relation" attribute)
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=8
    )

    # Final formatting
    plt.title("Meta-Graph of HeteroData Node Types and Edge Types")
    plt.axis('off')
    # plt.show()
    plot_path = "meta-graph.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path