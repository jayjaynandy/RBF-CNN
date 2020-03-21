# Approximate Manifold Defense Against Multiple Adversarial Perturbations

A shorter version of this paper has been accepted in [NeurIPS-2019 Workshop on Machine Learning with Guarantees](https://sites.google.com/view/mlwithguarantees/accepted-papers) and the full version 
Link to be added. (Full paper, IJCNN 2020)

In this work, we propose RBF-CNN, an approximate manifold-based defense framework against adversarial attacks for image classification. Unlike previous approaches, we demonstrate that our manifold based defense model is scalable to complex data manifold of natural images. 

To the best of our knowledge, RBF-CNN is the first generative model-based defense framework to achieve robustness for any Lp perturbations along with offering the flexibility of allowing for the trade-off adjustment of robustness vs accuracy at deployment.
Experiments on MNIST and CIFAR-10 have shown that our model achieves robustness any minor adversarial perturbations w.r.t L1, L2 and Linf norms.

In this repository, we provide the training code for our defense model. The training of RBF-CNN is a two-step process:

## Training of the RBF layer using non-parametric EM algorithm
Execute train_rbf_cifar.py and train_rbf_mnist.py for CIFAR-10 and MNIST respectively (Required library: numba).
We have already provide the trained RBF layer used for our model in `./layer_cifar/` and `./layer_mnist/` respectively.

## Training of the CNN network in presence of the trainined RBF layer
The code for rCNN+ is provided here. Execute `train_cnn_cifar.py` and `train_cnn_mnist.py` for CIFAR-10 and MNIST respectively.
To train the rCNN models, remove the data augmentation during training. 

Please note that, our RBF-CNN models become robust once it achieves `~100% training accuracy`.
Weights for the trained models are provided in `./trained_weights/`.

## Testing
Please follow the code: `test_cifar.py` and `test_mnist.py`
