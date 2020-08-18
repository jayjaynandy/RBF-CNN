# Approximate Manifold Defense Against Multiple Adversarial Perturbations

A shorter version of this paper has been accepted in [NeurIPS 2019 Workshop on Machine Learning with Guarantees](https://sites.google.com/view/mlwithguarantees/accepted-papers) 
[[pdf]](https://drive.google.com/file/d/1I2WKHg-s7wJgG21apg3FhxaYzzFl4vgt/view), 
[[poster]](https://drive.google.com/file/d/1Wp-kKsc0927ZXo5lS8f2GPnmSpIWdRlN/view) and the full version of this paper is accepted at IJCNN-2020 [[Arxiv Link]](https://arxiv.org/abs/2004.02183).
The video presentation of our paper is provided in this [youtube link](https://www.youtube.com/watch?v=oKBu90fuTgI).
An updated version that removes numba dependency to train the RBF layer can be founc [here](https://github.com/jayjaynandy/RBF_CNN-v2).

In this work, we propose RBF-CNN, an approximate manifold-based defense framework against adversarial attacks for image classification. Unlike previous approaches, we demonstrate that our manifold based defense model is scalable to complex data manifold of natural images. 

To the best of our knowledge, RBF-CNN is the first generative model-based defense framework to achieve robustness for any Lp perturbations along with offering the flexibility of allowing for the trade-off adjustment of robustness vs accuracy at deployment.
Our experiments on MNIST and CIFAR-10 demonstrate that RBF-CNN models provide robustness any minor adversarial perturbations w.r.t L1, L2, and Linf norms. Further, we have applied the certification technique proposed for [randomized smoothing](https://arxiv.org/abs/1902.02918) to demonstrate that our RBF-CNN models also attain provable robustness for L2 norms.

In this repository, we provide the training code for our defense model. The training of RBF-CNN is a two-step process:

## Step 1: Training of the RBF layer
Execute train_rbfLayer.py (`required library: numba`). We have already provide one copy of the trained RBF layer used for our model in `./rbf_layer/`.

## Step 2: Training of the CNN network in presence of the trainined RBF layer
CNN model can be trained only after training the RBF layer.

To train rCNN model, execute: `train_rCNN.py`.

To train rCNN+ model, execute: `train_rCNN+.py`.


Please note that, our RBF-CNN models become robust once it achieves `~100% training accuracy`.
Weights for the trained models are provided in `./trained_weights/`.

## Testing
Please follow the code: `test_script.py`.

## Citation

If our code or our results are useful in your reasearch, please consider citing:

```[bibtex]
@inproceedings{rbfcnn_ijcnn20,
  author={Jay Nandy and Wynne Hsu and Mong{-}Li Lee},
  title={Approximate Manifold Defense Against Multiple Adversarial Perturbations},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  year={2020},
}
```
