import numpy as np

# --- Node Creation Function ---

def create_physical_nodes(num_nodes: int) -> dict:
    """
    Generates a dictionary representing physical nodes with random resource capacities.

    Args:
        num_nodes: The number of physical nodes to create.

    Returns:
        A dictionary containing numpy arrays for cluster IDs and their
        available/total CPU, RAM, and Disk resources.
    """
    # Define possible resource options and their probabilities (equal probability here)
    cpu_options = [100, 200, 400]
    ram_options = [60, 120, 180]
    dsk_options = [100, 200, 400]

    # Randomly select resource capacities for each node
    cpu = np.random.choice(cpu_options, size=num_nodes, p=[1/3, 1/3, 1/3])
    dsk = np.random.choice(dsk_options, size=num_nodes, p=[1/3, 1/3, 1/3])
    ram = np.random.choice(ram_options, size=num_nodes, p=[1/3, 1/3, 1/3])

    # Store node data in a dictionary
    nodes_data = {
        'cluster_id': np.arange(1, num_nodes + 1),
        'available_cpu': cpu,
        'available_ram': ram,
        'available_dsk': dsk,
        'total_cpu': cpu,  # Initially, available resources are equal to total resources
        'total_ram': ram,
        'total_dsk': dsk,
    }
    return nodes_data

# --- Base Data Definitions for Microservices and Relations ---

# Base resource and request arrays for microservices
BASE_REPLICAS_STANDARD = np.array([5, 10, 9, 4, 10, 2, 8, 23, 9, 25, 5, 8, 18, 10, 12, 6, 5])
BASE_REPLICAS_MORE_05 = np.array([8, 15, 14, 6, 15, 3, 12, 35, 14, 38, 8, 12, 27, 15, 18, 9, 8])
BASE_REQUESTS_STANDARD = np.array([10, 8, 8, 5, 8, 4, 4, 4, 5, 4, 8, 4, 5, 4, 5, 4, 6])
BASE_REQUESTS_ROUNDED = np.array([15, 12, 12, 8, 12, 6, 6, 6, 8, 6, 12, 6, 8, 6, 8, 6, 9])

# Scaled consumption matrices for microservice dependencies (CONS)
# Each represents consumption patterns for different scaling levels.
# Using dtype=object to allow for variable-length nested lists/arrays.
CONS_PLUS_17 = np.array([
    [19, 21, 26], [21, 29], [30], [32, 33], [32], [], [19, 31], [31],
    [22, 28], [22, 26, 28], [19], [25], [19, 25, 33, 34], [], [33], [31], [29]
], dtype=object)

CONS_PLUS_34 = np.array([
    [36, 38, 43], [38, 46], [47], [49, 50], [49], [], [36, 48], [48],
    [39, 45], [39, 43, 45], [36], [42], [36, 42, 50, 51], [], [50], [48], [46]
], dtype=object)

CONS_PLUS_51 = np.array([
    [53, 55, 60], [55, 63], [64], [66, 67], [66], [], [53, 65], [65],
    [56, 62], [56, 60, 62], [53], [59], [53, 59, 67, 68], [], [67], [65], [63]
], dtype=object)

CONS_PLUS_68 = np.array([
    [70, 72, 77], [72, 80], [81], [83, 84], [83], [], [70, 82], [82],
    [73, 79], [73, 77, 79], [70], [76], [70, 76, 84, 85], [], [84], [82], [80]
], dtype=object)

# Base request and data transfer volumes between microservices
BASE_REQUESTS_IJ = [
    50, 70, 8, 30, 100, 30, 20, 10, 20, 10, 15, 60,
    30, 8, 30, 20, 10, 15, 20, 20, 20, 25, 20, 20,
    45, 20, 45, 8, 30, 8, 15, 15
]

BASE_DATA_TRANS_IJ = [
    0, 0, 0, 0, 0, 0, 4.6, 3.1, 4.0, 3.5, 5.9, 1.8,
    5.6, 5.7, 5.3, 4.8, 4.1, 4.2, 3.6, 4.7, 3.4, 4.4,
    4.9, 3.2, 6.4, 4.5, 6.1, 5.5, 2.4, 5.2, 4.3, 6.2, 0 # Added missing 0 at the end
]

# Scaled requests and data transfer (1.5x of base)
REQUESTS_IJ_1_5X = [
    75.0, 105.0, 12.0, 45.0, 150.0, 45.0, 30.0, 15.0,
    30.0, 15.0, 22.5, 90.0, 45.0, 12.0, 45.0, 30.0,
    15.0, 23.0, 30.0, 30.0, 30.0, 37.5, 30.0, 30.0,
    67.5, 30.0, 68.0, 12.0, 45.0, 12.0, 23.0, 23.0
]

DATA_TRANS_IJ_1_5X = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.9, 4.65, 6.0, 5.25, 8.85, 2.7,
    8.4, 8.55, 7.95, 7.2, 6.15, 6.3, 5.4, 7.05, 5.1, 6.6,
    7.35, 4.8, 9.6, 6.75, 9.15, 8.25, 3.6, 7.8, 6.45, 9.3, 0.0
]


# Base Microservices Data
BASE_MICROSERVICES_DATA = {
    'ms_i': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
    'CONS': np.array([
        [2, 4, 9], [4, 12], [13], [15, 16], [15], [], [2, 14], [14],
        [5, 11], [5, 9, 11], [2], [8], [2, 8, 16, 17], [], [16], [14], [12]
    ], dtype=object),
    'cpu': np.array([2.1, 0.5, 3.1, 4.7, 1.8, 2.5, 6.2, 0.8, 3.9, 0.2, 2.8, 5.3, 0.6, 6.1, 1.2, 5.4, 3.7]),
    'dsk': np.array([1.4, 3.2, 1.6, 0.2, 3.1, 5.1, 0.6, 6.2, 2.3, 4.8, 2.6, 0.9, 4.8, 2.5, 4.2, 1.6, 2.2]),
    'ram': np.array([2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 4, 2, 4]),
    'replicas': BASE_REPLICAS_STANDARD,
    'requests': BASE_REQUESTS_STANDARD,
}


# Base Relations Data (ms_pair, requests_i_j, data_trans_i_j)
BASE_RELATIONS_DATA = {
    'ms_pair': [
        (1, 1), (3, 3), (6, 6), (7, 7), (10, 10), (13, 13),
        (1, 2), (1, 4), (1, 9), (2, 4), (2, 12), (3, 13),
        (4, 15), (4, 16), (5, 15), (7, 2), (7, 14), (8, 14),
        (9, 5), (9, 11), (10, 5), (10, 9), (10, 11), (11, 2),
        (12, 8), (13, 2), (13, 8), (13, 16), (13, 17), (15, 16),
        (16, 14), (17, 12),
    ],
    'requests_i_j': BASE_REQUESTS_IJ,
    'data_trans_i_j': BASE_DATA_TRANS_IJ
}


# Additional MS pairs for higher scaling levels in relations data
MS_PAIRS_PLUS_17 = [
    (18, 18), (20, 20), (23, 23), (24, 24), (27, 27), (30, 30),
    (18, 19), (18, 21), (18, 26), (19, 21), (19, 29), (20, 30),
    (21, 32), (21, 33), (22, 32), (24, 19), (24, 31), (25, 31),
    (26, 22), (26, 28), (27, 22), (27, 26), (27, 28), (28, 19),
    (29, 25), (30, 19), (30, 25), (30, 33), (30, 34), (32, 33),
    (33, 31), (34, 29),
]

MS_PAIRS_PLUS_34 = [
    (35, 35), (37, 37), (40, 40), (41, 41), (44, 44), (47, 47),
    (35, 36), (35, 38), (35, 43), (36, 38), (36, 46), (37, 47),
    (38, 49), (38, 50), (39, 49), (41, 36), (41, 48), (42, 48),
    (43, 39), (43, 45), (44, 39), (44, 43), (44, 45), (45, 36),
    (46, 42), (47, 36), (47, 42), (47, 50), (47, 51), (49, 50),
    (50, 48), (51, 46),
]

MS_PAIRS_PLUS_51 = [
    (52, 52), (54, 54), (57, 57), (58, 58), (61, 61), (64, 64),
    (52, 53), (52, 55), (52, 60), (53, 55), (53, 63), (54, 64),
    (55, 66), (55, 67), (56, 66), (58, 53), (58, 65), (59, 65),
    (60, 56), (60, 62), (61, 56), (61, 60), (61, 62), (62, 53),
    (63, 59), (64, 53), (64, 59), (64, 67), (64, 68), (66, 67),
    (67, 65), (68, 63),
]

MS_PAIRS_PLUS_68 = [
    (69, 69), (71, 71), (74, 74), (75, 75), (78, 78), (81, 81),
    (69, 70), (69, 72), (69, 77), (70, 72), (70, 80), (71, 81),
    (72, 83), (72, 84), (73, 83), (75, 70), (75, 82), (76, 82),
    (77, 73), (77, 79), (78, 73), (78, 77), (78, 79), (79, 70),
    (80, 76), (81, 70), (81, 76), (81, 84), (81, 85), (83, 84),
    (84, 82), (85, 80),
]


# --- Helper Functions for Data Scaling ---

def create_scaled_microservices_data(
    base_data: dict,
    scale_factor: int,
    cons_appendices: list = None,
    replicas_array: np.ndarray = None,
    requests_array: np.ndarray = None
) -> dict:
    """
    Creates scaled microservices data based on a base set.

    Args:
        base_data: The base microservices data dictionary.
        scale_factor: The integer factor to scale the number of microservices.
                      This scales ms_i, cpu, dsk, and ram by repeating base data.
        cons_appendices: An optional list of CONS arrays to concatenate for scaled versions.
                         Each element in the list will be appended sequentially.
        replicas_array: An optional numpy array to use for 'replicas'. If None,
                        BASE_REPLICAS_STANDARD is repeated by scale_factor.
        requests_array: An optional numpy array to use for 'requests'. If None,
                        BASE_REQUESTS_STANDARD is repeated by scale_factor.

    Returns:
        A dictionary containing the scaled microservices data.
    """
    # Calculate the new total number of microservices
    new_ms_count = scale_factor * len(base_data['ms_i'])
    ms_i_scaled = np.arange(1, new_ms_count + 1)

    # Concatenate CONS arrays. Start with base_data['CONS'] and add appendices.
    cons_scaled = base_data['CONS']
    if cons_appendices:
        for cons_appendix in cons_appendices:
            cons_scaled = np.concatenate([cons_scaled, cons_appendix])

    # Concatenate resource arrays by repeating the base data 'scale_factor' times
    cpu_scaled = np.concatenate([base_data['cpu']] * scale_factor)
    dsk_scaled = np.concatenate([base_data['dsk']] * scale_factor)
    ram_scaled = np.concatenate([base_data['ram']] * scale_factor)

    # Use provided replicas/requests arrays if they exist, otherwise repeat base ones
    replicas_final = replicas_array if replicas_array is not None else np.concatenate([base_data['replicas']] * scale_factor)
    requests_final = requests_array if requests_array is not None else np.concatenate([base_data['requests']] * scale_factor)

    return {
        'ms_i': ms_i_scaled,
        'CONS': cons_scaled,
        'cpu': cpu_scaled,
        'dsk': dsk_scaled,
        'ram': ram_scaled,
        'replicas': replicas_final,
        'requests': requests_final,
    }

def create_scaled_relations_data(
    base_relations: dict,
    ms_pair_appendices: list = None,
    requests_data_list: list = None,
    data_trans_data_list: list = None
) -> dict:
    """
    Creates scaled microservice relations data.

    Args:
        base_relations: The base relations data dictionary.
        ms_pair_appendices: An optional list of additional ms_pair lists to concatenate.
                            Each element in the list will be extended sequentially.
        requests_data_list: An optional list of requests_i_j lists to concatenate.
                            Each element in the list will be extended sequentially.
        data_trans_data_list: An optional list of data_trans_i_j lists to concatenate.
                              Each element in the list will be extended sequentially.

    Returns:
        A dictionary containing the scaled relations data.
    """
    # Initialize with base data lists (make copies to avoid modifying originals)
    ms_pair_scaled = list(base_relations['ms_pair'])
    requests_scaled = list(base_relations['requests_i_j'])
    data_trans_scaled = list(base_relations['data_trans_i_j'])

    # Extend with provided appendices if they exist
    if ms_pair_appendices:
        for appendix in ms_pair_appendices:
            ms_pair_scaled.extend(appendix)

    if requests_data_list:
        for data_list in requests_data_list:
            requests_scaled.extend(data_list)

    if data_trans_data_list:
        for data_list in data_trans_data_list:
            data_trans_scaled.extend(data_list)

    return {
        'ms_pair': ms_pair_scaled,
        'requests_i_j': requests_scaled,
        'data_trans_i_j': data_trans_scaled
    }


# --- Scaled Microservices Data Definitions ---
# Using the helper function to generate different scaling scenarios

MICROSERVICES_DATA_1_5X = create_scaled_microservices_data(
    BASE_MICROSERVICES_DATA, 1, # Scale factor 1 as we're only changing replicas/requests
    replicas_array=BASE_REPLICAS_MORE_05,
    requests_array=BASE_REQUESTS_ROUNDED
)

MICROSERVICES_DATA_2X = create_scaled_microservices_data(
    BASE_MICROSERVICES_DATA, 2, # Scales ms_i, cpu, dsk, ram by 2x
    cons_appendices=[CONS_PLUS_17] # Adds the next set of CONS dependencies
)

MICROSERVICES_DATA_2_5X = create_scaled_microservices_data(
    BASE_MICROSERVICES_DATA, 2,
    cons_appendices=[CONS_PLUS_17],
    replicas_array=np.concatenate([BASE_REPLICAS_MORE_05, BASE_REPLICAS_STANDARD]),
    requests_array=np.concatenate([BASE_REQUESTS_ROUNDED, BASE_REQUESTS_STANDARD])
)

MICROSERVICES_DATA_3X = create_scaled_microservices_data(
    BASE_MICROSERVICES_DATA, 3,
    cons_appendices=[CONS_PLUS_17, CONS_PLUS_34]
)

MICROSERVICES_DATA_3_5X = create_scaled_microservices_data(
    BASE_MICROSERVICES_DATA, 3,
    cons_appendices=[CONS_PLUS_17, CONS_PLUS_34],
    replicas_array=np.concatenate([BASE_REPLICAS_MORE_05, BASE_REPLICAS_STANDARD, BASE_REPLICAS_STANDARD]),
    requests_array=np.concatenate([BASE_REQUESTS_ROUNDED, BASE_REQUESTS_STANDARD, BASE_REQUESTS_STANDARD])
)

MICROSERVICES_DATA_4X = create_scaled_microservices_data(
    BASE_MICROSERVICES_DATA, 4,
    cons_appendices=[CONS_PLUS_17, CONS_PLUS_34, CONS_PLUS_51]
)

MICROSERVICES_DATA_4_5X = create_scaled_microservices_data(
    BASE_MICROSERVICES_DATA, 4,
    cons_appendices=[CONS_PLUS_17, CONS_PLUS_34, CONS_PLUS_51],
    replicas_array=np.concatenate([BASE_REPLICAS_MORE_05, BASE_REPLICAS_STANDARD, BASE_REPLICAS_STANDARD, BASE_REPLICAS_STANDARD]),
    requests_array=np.concatenate([BASE_REQUESTS_ROUNDED, BASE_REQUESTS_STANDARD, BASE_REQUESTS_STANDARD, BASE_REQUESTS_STANDARD])
)

MICROSERVICES_DATA_5X = create_scaled_microservices_data(
    BASE_MICROSERVICES_DATA, 5,
    cons_appendices=[CONS_PLUS_17, CONS_PLUS_34, CONS_PLUS_51, CONS_PLUS_68]
)


# --- Scaled Relations Data Definitions ---
# Using the helper function and building upon previous scaled data where appropriate
# Note: For relations data, `ms_pair` needs to be manually appended with new tuples.

RELATIONS_DATA_1_5X = create_scaled_relations_data(
    BASE_RELATIONS_DATA,
    requests_data_list=[REQUESTS_IJ_1_5X],
    data_trans_data_list=[DATA_TRANS_IJ_1_5X]
)

RELATIONS_DATA_2X = create_scaled_relations_data(
    BASE_RELATIONS_DATA,
    ms_pair_appendices=[MS_PAIRS_PLUS_17],
    requests_data_list=[BASE_REQUESTS_IJ], # Append base requests for the new set of MS
    data_trans_data_list=[BASE_DATA_TRANS_IJ] # Append base data transfer for the new set of MS
)

RELATIONS_DATA_2_5X = create_scaled_relations_data(
    BASE_RELATIONS_DATA,
    ms_pair_appendices=[MS_PAIRS_PLUS_17],
    requests_data_list=[REQUESTS_IJ_1_5X], # Append 1.5x requests for the new set of MS
    data_trans_data_list=[DATA_TRANS_IJ_1_5X] # Append 1.5x data transfer for the new set of MS
)

# For 3X and higher, we concatenate the *previous* full set of relations data
# and then add the new 'plus' data.

RELATIONS_DATA_3X = create_scaled_relations_data(
    RELATIONS_DATA_2X, # Start from the 2X data
    ms_pair_appendices=[MS_PAIRS_PLUS_34],
    requests_data_list=[BASE_REQUESTS_IJ],
    data_trans_data_list=[BASE_DATA_TRANS_IJ]
)

RELATIONS_DATA_3_5X = create_scaled_relations_data(
    RELATIONS_DATA_2X, # Start from the 2X data
    ms_pair_appendices=[MS_PAIRS_PLUS_34],
    requests_data_list=[REQUESTS_IJ_1_5X], # Use 1.5x requests for the new set
    data_trans_data_list=[DATA_TRANS_IJ_1_5X] # Use 1.5x data transfer for the new set
)

RELATIONS_DATA_4X = create_scaled_relations_data(
    RELATIONS_DATA_3X, # Start from the 3X data
    ms_pair_appendices=[MS_PAIRS_PLUS_51],
    requests_data_list=[BASE_REQUESTS_IJ],
    data_trans_data_list=[BASE_DATA_TRANS_IJ]
)

RELATIONS_DATA_4_5X = create_scaled_relations_data(
    RELATIONS_DATA_3X, # Start from the 3X data
    ms_pair_appendices=[MS_PAIRS_PLUS_51],
    requests_data_list=[REQUESTS_IJ_1_5X],
    data_trans_data_list=[DATA_TRANS_IJ_1_5X]
)

RELATIONS_DATA_5X = create_scaled_relations_data(
    RELATIONS_DATA_4X, # Start from the 4X data
    ms_pair_appendices=[MS_PAIRS_PLUS_68],
    requests_data_list=[BASE_REQUESTS_IJ],
    data_trans_data_list=[BASE_DATA_TRANS_IJ]
)
