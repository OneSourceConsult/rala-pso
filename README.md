# RALA-PSO

## Overview

This repository implements a Particle Swarm Optimization (PSO) algorithm for resource allocation and scheduling in distributed systems. The solution optimizes the placement of microservices across clusters while considering resource constraints (CPU, RAM, Disk) and microservice dependencies.

## Key Features

- **Particle Swarm Optimization**: Implements a modified PSO algorithm with adaptive inertia weight and learning factors
- **Resource Management**: Considers CPU, RAM, and Disk constraints when scheduling microservices
- **Constraint Handling**: Supports microservice co-location constraints (CONS)
- **Adaptive Parameters**: Dynamically adjusts weights and learning factors during optimization
- **Scalable Design**: Works with varying numbers of clusters and microservices

## Components

1. **PSO Algorithm (`algorithm.py`)**
   - Core optimization logic
   - Fitness function calculation
   - Swarm initialization and iteration
   - Resource feasibility checks

2. **Particle Implementation (`particle.py`)**
   - Individual particle behavior
   - Position and velocity updates
   - Fitness evaluation

3. **Data Generation (`data.py`)**
   - Physical node creation with random resources
   - Microservice definitions with resource requirements
   - Pre-defined scaling configurations (1.5x to 5x)
   - Helper functions for data scaling

## Parameters

The algorithm uses the following configurable parameters (set in `algorithm.py`):

```python
params = {
    "max_cpu": 0.95,    # Maximum CPU usage threshold
    "max_ram": 0.90,    # Maximum RAM usage threshold
    "w_max": 0.7,       # Maximum inertia weight
    "w_min": 0.5,       # Minimum inertia weight
    "c1_min": 0.01,     # Minimum cognitive learning factor
    "c1_max": 2.5,      # Maximum cognitive learning factor
    "c2_min": 0.01,     # Minimum social learning factor
    "c2_max": 2.5,      # Maximum social learning factor
    "weights": [0.4, 0.5, 0.10],  # Fitness function weights
    "theta": 3.4,       # Weight control factor
    "max_itr": 120,     # Maximum iterations per run
    "tmax": 100         # Maximum time for parameter calculation
}
```

## Usage

1. **Initialize physical nodes:**

   ```python
   from data import create_physical_nodes
   clusters = create_physical_nodes(num_nodes=10)
   ```

2. **Select microservice configuration:**

   ```python
   from data import MICROSERVICES_DATA_2X
   components = MICROSERVICES_DATA_2X
   ```

3. **Run the scheduler:**

   ```python
   pso = PSO()
   solution, fitness = pso.scheduler(clusters, components)
   ```

## Data Scaling

The `data.py` module includes several pre-configured scaling levels:

- BASE_MICROSERVICES_DATA (1x)
- MICROSERVICES_DATA_1_5X
- MICROSERVICES_DATA_2X
- MICROSERVICES_DATA_3X
- MICROSERVICES_DATA_4X
- MICROSERVICES_DATA_5X

Each scaling level includes:

- Increased number of microservices
- Additional dependency constraints
- Scaled resource requirements

## Dependencies

- Python 3.x
- NumPy

## How to cite

If you use this repository in academic work, research, or technical reports, please cite it using one of the formats below.

### MLA

OneSource Consult. RALA-PSO. 2025,
https://github.com/OneSourceConsult/rala-pso

### APA (7th edition)

OneSource Consult. (2025). RALA-PSO.
https://github.com/OneSourceConsult/rala-pso

### ISO 690

OneSource Consult. **RALA-PSO**. 2025. Available at:  
https://github.com/OneSourceConsult/rala-pso.  
Accessed: Jul. 30, 2025.

### BibTeX

```bibtex
@misc{rala_pso_2025,
  author       = {{OneSource Consult}},
  title        = {{RALA-PSO}},
  year         = {2025},
  howpublished = {\url{https://github.com/OneSourceConsult/rala-pso}},
  note         = {Accessed: Jul. 30, 2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.


