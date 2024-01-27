# NiceFlow Generative Model

## Introduction
The NiceFlow Generative Model is a machine learning project focusing on generative models using NICE (Non-linear Independent Components Estimation) architecture. This project is structured to experiment with different configurations and datasets.

## Project Structure
The project includes several key directories:
- `data`: Contains datasets used in the model.
- `docs`: Documentation and visualizations of the model outputs.
- `models`: Model definitions and saved model states.
- `plots`: Directory for generated plots (currently empty).
- `samples`: Directory for sample outputs (currently empty).
- `nice.py`: The main script defining the NICE model.
- `train.py`: Script for training the model.

## Documentation and Visualizations
The `docs` directory includes visualizations for two types of transformations - `additive` and `affine`, applied to two datasets: `MNIST` and `Fashion-MNIST`. 

### Transformation Comparisons

#### MNIST
| Transformation Type | Image 1 | Image 2 |
| -------------------- | ------- | ------- |
| **Additive** | ![MNIST Additive 1](docs/additive/mnist/mnist_batch128_coupling4_coupling_typeadditive_mid1000_hidden5_.pt.png) | ![MNIST Additive 2](docs/additive/mnist/mnist_batch128_coupling4_coupling_typeadditive_mid1000_hidden5_.ptepoch46.png) |
| **Affine** | ![MNIST Affine 1](docs/affine/mnist/mnist_batch128_coupling4_coupling_typeaffine_mid1000_hidden5_.pt.png) | ![MNIST Affine 2](docs/affine/mnist/mnist_batch128_coupling4_coupling_typeaffine_mid1000_hidden5_.ptepoch46.png) |

#### Fashion-MNIST
| Transformation Type | Image 1 | Image 2 |
| -------------------- | ------- | ------- |
| **Additive** | ![Fashion-MNIST Additive 1](docs/additive/fashion-mnist/fashion-mnist_batch128_coupling4_coupling_typeadditive_mid1000_hidden5_.pt.png) | ![Fashion-MNIST Additive 2](docs/additive/fashion-mnist/fashion-mnist_batch128_coupling4_coupling_typeadditive_mid1000_hidden5_.ptepoch42.png) |
| **Affine** | ![Fashion-MNIST Affine 1](docs/affine/fashion-mnist/fashion-mnist_batch128_coupling4_coupling_typeaffine_mid1000_hidden5_.pt.png) | ![Fashion-MNIST Affine 2](docs/affine/fashion-mnist/fashion-mnist_batch128_coupling4_coupling_typeaffine_mid1000_hidden5_.ptepoch46.png) |
