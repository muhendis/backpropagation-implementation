import torch
from collections import OrderedDict
from activations import BaseActivationFunction, SigmoidActivationFunction, ReLUActivationFunction, IdentityActivationFunction
from collections import OrderedDict

########################################################################################################################################################################

class CustomLinearLayer:
    def __init__(self, in_features, out_features):
        self.in_features=in_features
        self.out_features=out_features        
        self.weight = torch.randn(out_features, in_features, requires_grad=True)#default
        self.bias = torch.randn(out_features, requires_grad=True)#default
        self.grad_weight = torch.zeros(out_features, in_features) #placeholder for grad weight
        self.grad_bias = torch.zeros(out_features) #placeholder for grad bias
        self.input=None#placeholder,for storing x value
        
    def forward(self, x):
        self.input = x
        return self.input @ self.weight.t() + self.bias

    def backward(self, grad_output):
        # grad_output is the gradient of the loss with respect to the output of the linear layer
        
        # Calculate the gradients of the weight (dL/dW)
        # This is the dot product of the input transposed (x^T) and the gradient of the loss with respect to the output of the layer (dL/dy)  
        # grad_weight is computed using torch.mm for matrix multiplication between the transposed gradient output and the input to the layer.
        self.grad_weight.data = grad_output.t() @ self.input  # Gradient with respect to the weight
        
        # Calculate the gradients of the biases (dL/db)
        # This is the sum of the gradients for each example in the batch
        # grad_bias is computed by summing grad_output across the batch dimension.
        self.grad_bias.data = grad_output.sum(0)  # Gradient with respect to the bias

        # Calculate the gradient of the loss with respect to the input of the layer (dL/dx)
        # This is needed for layers that come before this one in the network
        # This will be used by the previous layer in the network for its backward pass.
        grad_input = grad_output @ self.weight

        # Update weight and biases here if doing so manually, or return gradients for use in an optimizer
        # self.weight.data -= learning_rate * grad_weight
        # self.bias.data -= learning_rate * grad_bias
        
        return grad_input# grad_input

########################################################################################################################################################################

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """

        self.ActivationFunction_dict={
            "sigmoid": SigmoidActivationFunction,
            "relu":ReLUActivationFunction,
            "identity": IdentityActivationFunction}

        
        self.f_function = self.ActivationFunction_dict[f_function]()
        self.g_function = self.ActivationFunction_dict[g_function]()


        self.layers=OrderedDict([
            ('linear1', CustomLinearLayer(in_features=linear_1_in_features, out_features=linear_1_out_features)),
            ('activation1',self.f_function),
            ('linear2', CustomLinearLayer(in_features=linear_2_in_features, out_features=linear_2_out_features)),
            ('activation2',self.g_function),

        ])  
        
        self.parameters = dict(
            W1 = self.layers['linear1'].weight,
            b1 = self.layers['linear1'].bias,
            W2 = self.layers['linear2'].weight,
            b2 = self.layers['linear2'].bias,
        )
        self.grads = dict(
            dJdW1 = self.layers['linear1'].grad_weight,
            dJdb1 = self.layers['linear1'].grad_bias,
            dJdW2 = self.layers['linear2'].grad_weight,
            dJdb2 = self.layers['linear2'].grad_bias,
        )
      
        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, linear_1_in_features)
        returns:
            y_hat: output tensor of shape (batch_size, linear_2_out_features)
        """
        self.forward_results = []
        input = x
        for name, layer in self.layers.items():
            input = layer.forward(input)  
            self.forward_results.append((name, input))
        return self.forward_results[-1][1]
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # Start with the gradient of the loss
        grad = dJdy_hat
        # Iterate over the layers in reverse order
        for name, layer in reversed(self.layers.items()):
            if isinstance(layer, CustomLinearLayer):
                grad = layer.backward(grad)
            if isinstance(layer, BaseActivationFunction):
                grad = grad*layer.backward() 
                
   
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].data=self.grads[grad].zero_().data
        self.cache = dict()

    def __show_forward_result__(self):
        print(self.forward_results)
        
    def __repr__(self):
        lines = ['MLP(']
        for name, layer in self.layers.items():
            # For custom layers
            if isinstance(layer, CustomLinearLayer):
                lines.append(f'  ({name}): {layer.__class__.__name__}('
                             f'in_features={layer.in_features}, out_features={layer.out_features})')
            # For activation functions or other types of layers
            else:
                lines.append(f'  ({name}): {layer}')
        lines.append(')')
        return '\n'.join(lines)

########################################################################################################################################################################

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor of shape (batch_size, linear_2_out_features)
        y_hat: the prediction tensor of shape (batch_size, linear_2_out_features)

    Return:
        J: scalar loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # The torch.F.mse_loss function's default behavior is to use the reduction mode 'mean.'
    # Therefore, all the code was written with respect to that.
    #scalar loss
    loss_for_each_y_outputs=((y-y_hat)**2)
    loss_for_each_observation=loss_for_each_y_outputs.mean(axis=1)
    J=loss_for_each_observation.mean(axis=0) 

    # the gradient tensor of y_hat
    # (1/(y.shape[0]*y.shape[1])) is added  for the reduction mode 'mean.'
    dJdy_hat = (2*(y-y_hat)*(-1)) * (1/(y.shape[0]*y.shape[1]))


    return J,dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor of shape (batch_size, linear_2_out_features)
        y: the label tensor of shape (batch_size, linear_2_out_features)
        
    Return:
        J: scalar loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    
    #Small value to ensure numerical stability
    epsilon=1e-12
    # Clamp the input to prevent log(0)
    y_hat_clamped = torch.clamp(y_hat, epsilon, 1 - epsilon)
    #now, it is time to calculate
    loss_for_each_y_outputs = - (y * torch.log(y_hat_clamped) + (1 - y) * torch.log(1 - y_hat_clamped))
    loss_for_each_observation = loss_for_each_y_outputs.mean(axis=1)
    J=loss_for_each_observation.mean(axis=0) 

    # the gradient tensor of y_hat
    # (1/(y.shape[0]*y.shape[1])) is added  for the reduction mode 'mean.'
    
    # Derivation (dJdy_hat) ):
    #--> (-(y/ y_hat_clamped)) + ((1 - y) / (1 - y_hat_clamped)) but;
    #--> since y_hat is output of sigmoid(z), derivation must have applied to sigmoid  

    dJdy_hat = ((-(y/ (y_hat_clamped))) + ((1 - y) / (1 - (y_hat_clamped)))) * (1/(y.shape[0]*y.shape[1])) 
    #dJdy_hat = (y_hat - y) * (1/(y.shape[0]*y.shape[1])) 

    return J,dJdy_hat



def cross_entropy_loss(y, y_hat):
    """
    Args:
        y: the correct class indices tensor of shape (batch_size,)
        y_hat: the prediction tensor of shape (batch_size, linear_2_out_features)

    Return:
        J: scalar loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """

    # 1.  Apply softmax to your neural network's raw outputs to get probabilities. (y_hat)
    # first, for numerical stability 
    #Before applying the exponential function in the softmax function, the maximum value of the inputs is subtracted from each input. 
    #This operation does not change the output of the softmax function but prevents large values from being exponentiated,
    #which helps in avoiding overflow.   
    max_y_hat = torch.max(y_hat, dim=1, keepdim=True).values
    exps = torch.exp(y_hat - max_y_hat)
    softmax = exps / torch.sum(exps, dim=1, keepdim=True)

    # 2. Calculate the log of those probabilities
    log_probs = torch.log(softmax)

    # 3. Select the log probability of the correct class for each example in the batch.
    batch_size = y.shape[0]
    actual_log_probs = log_probs[range(batch_size), y]

    # 4. Compute the mean of the negative log probabilities (cross-entropy loss)
    J = -actual_log_probs.mean()

    #--------------------------  Compute gradients------------- (mathematic word de)
    # Calculate the gradient of the loss
    # Initialize gradient tensor with probabilities
    dJdy_hat = softmax.clone()
    # Subtract 1 from the true class probabilities (where y_j = 1)
    dJdy_hat[range(batch_size), y] -= 1
    # Average the gradients over the batch
    dJdy_hat /= batch_size
    return J, dJdy_hat



