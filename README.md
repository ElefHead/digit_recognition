# MNIST Digit Recognition
Fully connected neural networks for handwritten digit classification using [MNIST data](http://yann.lecun.com/exdb/mnist/)

## Description  
*Layers* : 3 ( 200, 200, 10 )  
*Optimization* : AMSGrad  
*Weight Initialization* : He-normal  
*Regularization* : L2  
*Activation* : ( elu, elu, softmax )  
*Cost Function* : Categorical cross-entropy  
*Total epochs* : 100  
*Batch size* : 250  

## Performance
Here are some performance stats using various optimizers: 
  
|Optimization    | Training Accuracy | Testing Accuracy |
|:--------------:|------------------:|-----------------:|
|Momentum        | 0.586             | 0.584            |
|RMSProp         | 0.9870            | 0.9679           |
|AMSGrad         | 0.9975            | 0.9743           |
|ADAM(TensorFlow)| 0.9967            | 0.9735           |

## Notes

* The latest commit has only AMSGrad. RMSProp and Momentum are present in previous commits.  
* The tensorflow code was for me to see the kind of performance I should be replicating. It uses AdamOptimization and not AMSGrad.
* The performance stats aren't an average over x attempts, its just what I got running them once. Thus, they're not absolute or conclusive of anything and are subject to
randoms.  This point is specifically directed at my AMSGrad implementation vs Tensorflow's ADAM.


---  

This was a fun exercise for me to see the impact of various optimizers and needless to say, I was absolutely blown away by
the boost in performance caused by RMSProp and AMSGrad.
As another fun exercise, I have updated the repository with the tenserflow eager implementation of equivalent ADAM code.
