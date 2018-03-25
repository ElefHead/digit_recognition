# MNIST Digit Recognition
Fully connected neural network (tensorflow as well as pure numpy codes) for digit classification using MNIST data

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
  
|Optimization   | Training Accuracy | Testing Accuracy |  
|:-------------:|------------------:|-----------------:|  
|Momentum       | 0.586             | 0.584            |  
|RMSProp        | 0.987             | 0.9679           |  
|AMSGrad        | 0.9908            | 0.9743           |

## Notes

* All the above metrics were obtained on my numpy implementation.  
* The latest commit has only AMSGrad, RMSProp and Momentum are present in previous commits.  
* The tensorflow code was for me to the kind of performance I should be getting. It doesn't make use of He-initialization or
L2-regularization. I will update the tensorflow code to replicate all that is happening in the numpy code.  

---  

This was a fun exercise for me to see the impact of various optimizers and needless to say, I was absolutely blown away by
the boost in performance caused by RMSProp and AMSGrad.  
