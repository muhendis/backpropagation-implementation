import torch
from abc import ABC, abstractmethod

class BaseActivationFunction(ABC):
    """
    Abstract base class for activation functions in neural networks.
    """

    def __init__(self):
        self.input_tensor = None
        self.output = None

    @abstractmethod
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        returns a tensor which is the output of a function.
        """
        raise NotImplementedError


    @abstractmethod
    def backward(self,grad: torch.Tensor) -> torch.Tensor:
        """
        returns a tensor which is the derivative of a function.
        """
        raise NotImplementedError



#############################################################################################################################################################################
class ActivationFunctionError(Exception):
    """Custom exception for errors related to activation functions."""

#############################################################################################################################################################################
class SigmoidActivationFunction(BaseActivationFunction):
    """
    Sigmoid activation function implementation.
    """

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        #  the input tensor is clamped to a range of [-15, 15]. 
        # This range is generally sufficient to avoid the overflow and underflow problems while still maintaining the characteristics of the sigmoid function
        self.input_tensor=input_tensor
        self.input_tensor = torch.clamp(self.input_tensor, min=-15, max=15)
        self.output = 1 / (1 + torch.exp(-self.input_tensor))
        return self.output

    def backward(self) -> torch.Tensor:
        if self.output is None:
            raise ActivationFunctionError("Forward pass must be called before backward.")
        return self.output * (1 - self.output)

class ReLUActivationFunction(BaseActivationFunction):
    """
    ReLU activation function implementation.
    """

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.input_tensor=input_tensor
        self.output = torch.maximum(self.input_tensor, torch.tensor(0.0))
        return self.output

    def backward(self) -> torch.Tensor:
        if self.output is None:
            raise ActivationFunctionError("Forward pass must be called before backward.")
        return (self.input_tensor > 0).float()

class IdentityActivationFunction(BaseActivationFunction):
    """
    Identity activation function implementation.
    """

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.input_tensor=input_tensor
        self.output = self.input_tensor
        return self.output

    def backward(self) -> torch.Tensor:
        if self.output is None:
            raise ActivationFunctionError("Forward pass must be called before backward.")
        return torch.ones_like(self.input_tensor)
