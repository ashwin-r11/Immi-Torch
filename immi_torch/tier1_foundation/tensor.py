"""started....
date:03/01/26     :{dd/mm/yy}

Module 01: Tensor - The Foundation of Everything

//////////////////////////////////////////////////////////////////////////////////////
useful know how's:
Tensor Initialization Process:

Input Data → Validation → NumPy Array → Tensor Wrapper → Ready for Operations
   [1,2,3] →    types   →  np.array   →    shape=(3,)  →     + - * / @ ...
     ↓             ↓          ↓             ↓
  List/Array    Type Check   Memory      Attributes Set
               (optional)    Allocation

Memory Allocation Example:
Input: [[1, 2, 3], [4, 5, 6]]
         ↓
NumPy allocates: [1][2][3][4][5][6] in contiguous memory
         ↓
Tensor wraps with: shape=(2,3), size=6, dtype=int64

//////////////////////////////////////////////////////////////////////////////////////
Tensor Class Structure:
┌─────────────────────────────────┐
│ Core Attributes:                │
│ • data: np.array (the numbers)  │
│ • shape: tuple (dimensions)     │
│ • size: int (total elements)    │
│ • dtype: type (float32)         │
├─────────────────────────────────┤
│ Arithmetic Operations:          │
│ • __add__, __sub__, __mul__     │
│ • __truediv__, matmul()         │
├─────────────────────────────────┤
│ Shape Operations:               │
│ • reshape(), transpose()        │
│ • sum(), mean(), max()          │
│ • __getitem__ (indexing)        │
├─────────────────────────────────┤
│ Utility Methods:                │
│ • __repr__(), __str__()         │
│ • numpy(), memory_footprint()   │
└─────────────────────────────────┘
"""


#code part!                                                             ~written by @ashwin-r11 on jan5_26
import numpy as np

#| export
class Tensor:
   
      def __init__(self,data):
         """Create a new tensor from data.
         """
        
         self.data = np.array(data, dtype=np.float32)
         self.shape = self.data.shape
         self.size = self.data.size
         self.dtype = self.data.dtype
         
      def __repr__(self):
         """String representation of tensor for debugging.
         """
         return f"data = Tensor(data={self.data} ; shape = {self.shape})"
        
      def __str__(self):
          """human readable string representation
          """
          return f"Tensor({self.data})"
      
      def numpy(self):
        """Return the underlying NumPy array.
        """
        return self.data
     
      def memory_footprint(self):
        """Calculate exact memory usage in bytes.
        """
        return self.data.nbytes
     
      #--------------------------------------------------------------------------
      #Tensor Operations Fns
      
      #overwriting add_operation
      def __add__(self,other):
         if isinstance(other,Tensor):
            other=other.data
         return Tensor(self.data+other)                                 #other(NOT! other.data) here signifies any other scalar type! or even tensor too! since we unwrapped it in before line!!!
      
      #overwriting sub_operation
      def __sub__(self,other):
         """Subtract two tensors element-wise.
         """
         if isinstance(other,Tensor):
            other=other.data
         return Tensor(self.data-other)
      
      #overwriting mul_operation-----------(not! matmul just element wise mul!)
      def __mul__(self,other):
         """Multiply two tensors element-wise (NOT matrix multiplication).
         """ 
         if isinstance(other,Tensor):
            other=other.data
         return Tensor(self.data*other)
      
      #overwriting trueDiv_operation
      def __truediv__(self, other):
        """Divide two tensors element-wise.
        """
        if isinstance(other,Tensor):
            other=other.data
        return Tensor(self.data/other)
      
      #matmul fn
      def matmul(self, other):
        """Matrix multiplication of two tensors.
        """
        
        if not isinstance(other, Tensor):
            raise TypeError(f"Expected Tensor for matrix multiplication, got {type(other)}")
        if self.shape == () or other.shape == ():
            return Tensor(self.data * other.data)
        if len(self.shape) == 0 or len(other.shape) == 0:
            return Tensor(self.data * other.data)
        if len(self.shape) >= 2 and len(other.shape) >= 2:
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"Cannot perform matrix multiplication: {self.shape} @ {other.shape}. "
                    f"Inner dimensions must match: {self.shape[-1]} ≠ {other.shape[-2]}"
                )

        # Educational implementation: explicit loops to show what matrix multiplication does
        # This is intentionally slower than np.matmul to demonstrate the value of vectorization
        # In Module 17 (Acceleration), will  use optimized BLAS operations

        a = self.data
        b = other.data

        # Handle 2D matrices with explicit loops (educational)
        if len(a.shape) == 2 and len(b.shape) == 2:
            M, K = a.shape
            K2, N = b.shape
            result_data = np.zeros((M, N), dtype=a.dtype)

            # Explicit nested loops - students can see exactly what's happening!
            # Each output element is a dot product of a row from A and a column from B
            for i in range(M):
                for j in range(N):
                    # Dot product of row i from A with column j from B
                    result_data[i, j] = np.dot(a[i, :], b[:, j])
        else:
            # For batched operations (3D+), use np.matmul for correctness
            # Students will understand this once they grasp the 2D case
            result_data = np.matmul(a, b)

        return Tensor(result_data)

            
      def __matmul__(self,other):                                       #to activate @ symbol to matmul fn
         return self.matmul(other)
      
      def __getitem__(self, key):
        """Enable indexing and slicing operations on Tensors.
        """
        
        result_data = self.data[key]
        if not isinstance(result_data,np.ndarray):
           wrapped_results = np.ndarray(result_data)
        return Tensor(wrapped_results)           
      
      #reshape fn
      #need to revisit!!
      def reshape(self, *shape):
        """Reshape tensor to new dimensions.
        """
        
        if len(shape)==1 and isinstance(shape[0],(tuple,list)):         #Handle both reshape(2, 3) and reshape((2, 3)) calling styles
           new_shape = tuple(shape[0])
        else:
           new_shape = shape
         
        if -1 in new_shape:                                             #If -1 in shape, infer that dimension from total size
           if new_shape.count(-1)>1:
              raise ValueError("more than one -1 !! can't find appropriate dimension to reshape!") 
           known_size = 1
           unknown_idx = new_shape.index(-1)
           for i, dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim
           unknown_dim = self.size // known_size
           new_shape = list(new_shape)
           new_shape[unknown_idx] = unknown_dim
           new_shape = tuple(new_shape)
        if np.prod(new_shape) != self.size:                             #Validate total elements match
            target_size = int(np.prod(new_shape))
            raise ValueError(
                f"Total elements must match: {self.size} ≠ {target_size}"
            )
        reshaped_data = np.reshape(self.data, new_shape)                #to create new view
        return Tensor(reshaped_data)
     
      #transpose fn
      def transpose(self, dim0=None, dim1=None):
        """Transpose tensor dimensions.
        """
         
        if dim0 is None and dim1 is None:
            if len(self.shape) < 2:
                return Tensor(self.data.copy())                         #For 1D tensors: return copy
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed_data = np.transpose(self.data, axes)         #If no dims specified: swap last two dimensions
        else:
            if dim0 is None or dim1 is None:
                raise ValueError("Both dim0 and dim1 must be specified")#If both dims specified: swap those specific dimensions
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            transposed_data = np.transpose(self.data, axes)
        return Tensor(transposed_data)
      
      
      #sum_along_axis fn
      def sum(self, axis=None, keepdims=False):
        """Sum tensor along specified axis. 
        """   
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)
     
      def mean(self, axis=None, keepdims=False):
         """mean tensor along specified axis. 
         """ 
         result = np.mean(self.data, axis=axis,keepdims=keepdims)
         return Tensor(result)
      
      def max(self, axis=None, keepdims=False):
        """Find maximum values along specified axis.
        """
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)
     
      #--------------------------------------------------------------------------
       
      
      