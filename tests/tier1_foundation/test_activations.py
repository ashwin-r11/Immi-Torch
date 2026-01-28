"""Tests for `immi_torch.tier1_foundation.activations`"""
import pytest
import numpy as np
from immi_torch.tier1_foundation import activations
from immi_torch.tier1_foundation import Tensor, ReLU, Sigmoid, Tanh, GELU, Softmax

def test_import_activations():
    assert activations is not None

# %% [markdown]
"""
### 🧪 Unit Test: Sigmoid

This test validates sigmoid activation behavior.

**What we're testing**: Sigmoid squashes values to (0, 1) range
**Why it matters**: Essential for binary classification outputs and gates
**Expected**: All outputs in (0, 1), sigmoid(0) = 0.5, monotonic increasing
"""

# %% nbgrader={"grade": true, "grade_id": "test-sigmoid", "locked": true, "points": 10}
def test_unit_sigmoid():
    """🧪 Test Sigmoid implementation."""
    print("🧪 Unit Test: Sigmoid...")
    
    sigmoid = Sigmoid()
    
    # Test basic output range
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    result = sigmoid.forward(x)
    
    # All values should be in (0, 1)
    assert np.all(result.data > 0), "All sigmoid values should be positive"
    assert np.all(result.data < 1), "All sigmoid values should be less than 1"
    
    # sigmoid(0) should be 0.5
    assert np.isclose(result.data[2], 0.5), f"sigmoid(0) should be 0.5, got {result.data[2]}"
    
    # Should be monotonically increasing
    assert np.all(np.diff(result.data) > 0), "Sigmoid should be monotonically increasing"
    
    # Test numerical stability with large values
    x_large = Tensor(np.array([-1000.0, 1000.0]))
    result_large = sigmoid.forward(x_large)
    assert not np.any(np.isnan(result_large.data)), "Sigmoid should not produce NaN"
    assert not np.any(np.isinf(result_large.data)), "Sigmoid should not produce infinity"
    
    print("✅ Sigmoid works correctly!")


# %% [markdown]
"""
### 🧪 Unit Test: ReLU

This test validates ReLU activation behavior.

**What we're testing**: ReLU zeros out negative values, preserves positives
**Why it matters**: Most popular hidden layer activation
**Expected**: max(0, x) behavior
"""

# %% nbgrader={"grade": true, "grade_id": "test-relu", "locked": true, "points": 10}
def test_unit_relu():
    """🧪 Test ReLU implementation."""
    print("🧪 Unit Test: ReLU...")
    
    relu = ReLU()
    
    # Test basic behavior
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    result = relu.forward(x)
    
    # Negative values should become 0
    assert result.data[0] == 0, "ReLU should zero out negative values"
    assert result.data[1] == 0, "ReLU should zero out negative values"
    
    # Zero should stay zero
    assert result.data[2] == 0, "ReLU(0) should be 0"
    
    # Positive values should be unchanged
    assert result.data[3] == 1.0, "ReLU should preserve positive values"
    assert result.data[4] == 2.0, "ReLU should preserve positive values"
    
    # Test with 2D tensor
    x_2d = Tensor(np.array([[-1, 2], [3, -4]]))
    result_2d = relu.forward(x_2d)
    expected = np.array([[0, 2], [3, 0]])
    assert np.allclose(result_2d.data, expected), "ReLU should work with 2D tensors"
    
    print("✅ ReLU works correctly!")


# %% [markdown]
"""
### 🧪 Unit Test: Tanh

This test validates tanh activation behavior.

**What we're testing**: Tanh squashes values to (-1, 1) range
**Why it matters**: Better gradient flow than sigmoid, zero-centered
**Expected**: All outputs in (-1, 1), tanh(0) = 0
"""

# %% nbgrader={"grade": true, "grade_id": "test-tanh", "locked": true, "points": 10}
def test_unit_tanh():
    """🧪 Test Tanh implementation."""
    print("🧪 Unit Test: Tanh...")
    
    tanh = Tanh()
    
    # Test basic output range
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    result = tanh.forward(x)
    
    # All values should be in (-1, 1)
    assert np.all(result.data > -1), "All tanh values should be > -1"
    assert np.all(result.data < 1), "All tanh values should be < 1"
    
    # tanh(0) should be 0
    assert np.isclose(result.data[2], 0.0), f"tanh(0) should be 0, got {result.data[2]}"
    
    # Should be monotonically increasing
    assert np.all(np.diff(result.data) > 0), "Tanh should be monotonically increasing"
    
    # Should be symmetric around origin: tanh(-x) = -tanh(x)
    assert np.isclose(result.data[0], -result.data[4]), "Tanh should be symmetric"
    
    print("✅ Tanh works correctly!")


# %% [markdown]
"""
### 🧪 Unit Test: GELU

This test validates GELU activation behavior.

**What we're testing**: GELU provides smooth approximation of ReLU
**Why it matters**: Used in modern transformers like GPT and BERT
**Expected**: Smooth transition near zero, ReLU-like for large positives
"""

# %% nbgrader={"grade": true, "grade_id": "test-gelu", "locked": true, "points": 10}
def test_unit_gelu():
    """🧪 Test GELU implementation."""
    print("🧪 Unit Test: GELU...")
    
    gelu = GELU()
    
    # Test basic behavior
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    result = gelu.forward(x)
    
    # GELU(0) should be 0
    assert np.isclose(result.data[2], 0.0), f"GELU(0) should be 0, got {result.data[2]}"
    
    # For large positive values, GELU ≈ x (identity-like)
    x_large_pos = Tensor(np.array([5.0]))
    result_large = gelu.forward(x_large_pos)
    assert np.isclose(result_large.data[0], 5.0, rtol=0.1), "GELU should approach identity for large positive values"
    
    # For large negative values, GELU ≈ 0
    x_large_neg = Tensor(np.array([-5.0]))
    result_large_neg = gelu.forward(x_large_neg)
    assert np.isclose(result_large_neg.data[0], 0.0, atol=0.1), "GELU should approach 0 for large negative values"
    
    print("✅ GELU works correctly!")


# %% [markdown]
"""
### 🧪 Unit Test: Softmax

This test validates softmax activation behavior.

**What we're testing**: Softmax creates valid probability distributions
**Why it matters**: Essential for multi-class classification outputs
**Expected**: Outputs sum to 1.0, all values in (0, 1), largest input gets highest probability
"""

# %% nbgrader={"grade": true, "grade_id": "test-softmax", "locked": true, "points": 10}
def test_unit_softmax():
    """🧪 Test Softmax implementation."""
    print("🧪 Unit Test: Softmax...")

    softmax = Softmax()

    # Test basic probability properties
    x = Tensor([1, 2, 3])
    result = softmax.forward(x)

    # Should sum to 1
    assert np.allclose(np.sum(result.data), 1.0), f"Softmax should sum to 1, got {np.sum(result.data)}"

    # All values should be positive
    assert np.all(result.data > 0), "All softmax values should be positive"

    # All values should be less than 1
    assert np.all(result.data < 1), "All softmax values should be less than 1"

    # Largest input should get largest output
    max_input_idx = np.argmax(x.data)
    max_output_idx = np.argmax(result.data)
    assert max_input_idx == max_output_idx, "Largest input should get largest softmax output"

    # Test numerical stability with large numbers
    x = Tensor([1000, 1001, 1002])  # Would overflow without max subtraction
    result = softmax.forward(x)
    assert np.allclose(np.sum(result.data), 1.0), "Softmax should handle large numbers"
    assert not np.any(np.isnan(result.data)), "Softmax should not produce NaN"
    assert not np.any(np.isinf(result.data)), "Softmax should not produce infinity"

    # Test with 2D tensor (batch dimension)
    x = Tensor([[1, 2], [3, 4]])
    result = softmax.forward(x, dim=-1)  # Softmax along last dimension
    assert result.shape == (2, 2), "Softmax should preserve input shape"
    # Each row should sum to 1
    row_sums = np.sum(result.data, axis=-1)
    assert np.allclose(row_sums, [1.0, 1.0]), "Each row should sum to 1"

    print("✅ Softmax works correctly!")

if __name__ == "__main__":
    test_unit_softmax()

# %% [markdown]
"""
## 🔧 Integration: Bringing It Together

Now let's test how all our activation functions work together and understand their different behaviors.
"""


# %% [markdown]
"""
### Understanding the Output Patterns

From the demonstration above, notice how each activation serves a different purpose:

**Sigmoid**: Squashes everything to (0, 1) - good for probabilities
**ReLU**: Zeros negatives, keeps positives - creates sparsity
**Tanh**: Like sigmoid but centered at zero (-1, 1) - better gradient flow
**GELU**: Smooth ReLU-like behavior - modern choice for transformers
**Softmax**: Converts to probability distribution - sum equals 1

These different behaviors make each activation suitable for different parts of neural networks.
"""

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "module-test", "locked": true, "points": 20}

def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_sigmoid()
    test_unit_relu()
    test_unit_tanh()
    test_unit_gelu()
    test_unit_softmax()

    print("\nRunning integration scenarios...")

    # Test 1: All activations preserve tensor properties
    print("🧪 Integration Test: Tensor property preservation...")
    test_data = Tensor([[1, -1], [2, -2]])  # 2D tensor

    activations = [Sigmoid(), ReLU(), Tanh(), GELU()]
    for activation in activations:
        result = activation.forward(test_data)
        assert result.shape == test_data.shape, f"Shape not preserved by {activation.__class__.__name__}"
        assert isinstance(result, Tensor), f"Output not Tensor from {activation.__class__.__name__}"

    print("✅ All activations preserve tensor properties!")

    # Test 2: Softmax works with different dimensions
    print("🧪 Integration Test: Softmax dimension handling...")
    data_3d = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])  # (2, 2, 3)
    softmax = Softmax()

    # Test different dimensions
    result_last = softmax(data_3d, dim=-1)
    assert result_last.shape == (2, 2, 3), "Softmax should preserve shape"

    # Check that last dimension sums to 1
    last_dim_sums = np.sum(result_last.data, axis=-1)
    assert np.allclose(last_dim_sums, 1.0), "Last dimension should sum to 1"

    print("✅ Softmax handles different dimensions correctly!")

    # Test 3: Activation chaining (simulating neural network)
    print("🧪 Integration Test: Activation chaining...")

    # Simulate: Input → Linear → ReLU → Linear → Softmax (like a simple network)
    x = Tensor([[-1, 0, 1, 2]])  # Batch of 1, 4 features

    # Apply ReLU (hidden layer activation)
    relu = ReLU()
    hidden = relu.forward(x)

    # Apply Softmax (output layer activation)
    softmax = Softmax()
    output = softmax.forward(hidden)

    # Verify the chain
    assert hidden.data[0, 0] == 0, "ReLU should zero negative input"
    assert np.allclose(np.sum(output.data), 1.0), "Final output should be probability distribution"

    print("✅ Activation chaining works correctly!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 02")

# Run comprehensive module test
if __name__ == "__main__":
    test_module()


# %% [markdown]
"""
## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of activation functions and their systems implications:

### 1. Computational Cost Comparison
**Question**: ReLU is the most popular activation function in hidden layers. Given what you implemented, why is ReLU computationally cheaper than Sigmoid or GELU?

**Consider**:
- What mathematical operations does ReLU require? (hint: just max(0, x))
- What operations does Sigmoid require? (hint: exponentials)
- If you have a hidden layer with 1 million neurons, how many exp() calls does each activation require?

**Real-world context**: In production models with billions of parameters, even small per-element costs add up. ReLU's simplicity makes it 3-4x faster than Sigmoid.

---

### 2. Numerical Stability
**Question**: Look at your Softmax implementation. Why did we subtract the maximum value before computing exponentials?

**Consider**:
- What happens when you compute exp(1000)?
- What about exp(1000) / (exp(1000) + exp(1001))?
- Does subtracting a constant from all inputs change the final softmax output?

**Mathematical insight**: exp(x - max) / sum(exp(x - max)) = exp(x) / sum(exp(x)) because the constant cancels.

---

### 3. Sparsity and Efficiency
**Question**: ReLU creates "sparsity" by zeroing negative values. Why might having many zero activations be beneficial for neural networks?

**Consider**:
- Memory: Do zeros need to be stored differently than non-zeros?
- Computation: What happens when you multiply by zero?
- Learning: If 50% of neurons are "off" for a given input, what does that mean for the representation?

**Think about**:
- Sparse matrix representations and their memory benefits
- How GPUs handle sparse operations
- Whether sparsity helps or hurts different types of neural networks

---

### 4. Activation Selection for Different Layers
**Question**: Why do we typically use different activations for hidden layers vs. output layers?

**Consider the requirements**:
- Hidden layers: Need to preserve gradients, be efficient, add nonlinearity
- Binary classification output: Need values in (0, 1) representing probability
- Multi-class classification output: Need probability distribution (sum = 1)

**Match the activation to the use case**:
- ReLU for hidden layers (why?)
- Sigmoid for binary output (why?)
- Softmax for multi-class output (why?)

---

### 5. The "Dying ReLU" Problem
**Question**: A neuron's output is always zero if its input is always negative. What situations might cause this, and why is it a problem?

**Consider**:
- If weights are initialized poorly, could all inputs to a neuron be negative?
- Once a ReLU neuron "dies" (always outputs 0), can it recover during training?
- How does the gradient flow through a ReLU that outputs 0?

**Solutions used in practice**:
- LeakyReLU: f(x) = max(0.01*x, x) - small slope for negatives
- PReLU: Learnable slope for negative values
- GELU: Smooth approximation with no sharp corner

---

### Bonus Challenge: Memory Analysis

**Scenario**: You're running inference on a model with a hidden layer of size (batch=32, features=4096) using different activations.

**Calculate for each activation**:
1. How many bytes of output memory are needed? (assume float32)
2. How many temporary buffers does Softmax need vs ReLU?
3. If you switch from float32 to float16, what's the memory savings?

**Key insight**: Activation functions are memory-light (output same size as input), but the choice affects computational speed and numerical precision significantly.
"""


# %% [markdown]
"""
## 📊 Systems Analysis: Activation Computation Costs

Let's understand ONE key systems concept: **computational cost differences between activations**.

This analysis reveals why ReLU dominates hidden layers while more expensive activations are reserved for specific use cases.
"""

# %%
def analyze_activation_performance():
    """Demonstrate computational cost differences between activation functions."""
    print("Analyzing Activation Computation Costs...")
    print("=" * 60)

    import time

    # Create test data (realistic hidden layer size)
    size = 1000000  # 1 million elements (like a large hidden layer)
    test_data = Tensor(np.random.randn(size).astype(np.float32))

    print(f"\nTesting with {size:,} elements (simulating large hidden layer)")
    print("-" * 60)

    # Initialize activations
    relu = ReLU()
    sigmoid = Sigmoid()
    tanh = Tanh()
    gelu = GELU()

    # Warm up
    _ = relu(test_data)
    _ = sigmoid(test_data)

    # Time each activation (multiple runs for accuracy)
    n_runs = 10

    # ReLU timing
    start = time.time()
    for _ in range(n_runs):
        _ = relu(test_data)
    relu_time = (time.time() - start) / n_runs * 1000

    # Sigmoid timing
    start = time.time()
    for _ in range(n_runs):
        _ = sigmoid(test_data)
    sigmoid_time = (time.time() - start) / n_runs * 1000

    # Tanh timing
    start = time.time()
    for _ in range(n_runs):
        _ = tanh(test_data)
    tanh_time = (time.time() - start) / n_runs * 1000

    # GELU timing
    start = time.time()
    for _ in range(n_runs):
        _ = gelu(test_data)
    gelu_time = (time.time() - start) / n_runs * 1000

    print("\n🧪 Activation Performance Results:")
    print(f"   ReLU:    {relu_time:.2f}ms (baseline)")
    print(f"   Sigmoid: {sigmoid_time:.2f}ms ({sigmoid_time/relu_time:.1f}x slower)")
    print(f"   Tanh:    {tanh_time:.2f}ms ({tanh_time/relu_time:.1f}x slower)")
    print(f"   GELU:    {gelu_time:.2f}ms ({gelu_time/relu_time:.1f}x slower)")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("   1. ReLU is fastest: Just max(0, x) - no exponentials")
    print("   2. Sigmoid/Tanh require exp() - expensive operation")
    print("   3. GELU uses sigmoid internally - inherits its cost")
    print("   4. For hidden layers: ReLU's speed advantage adds up!")

    print("\nREAL-WORLD IMPLICATIONS:")
    print("   - ResNet uses ReLU: billions of activations per forward pass")
    print("   - GPT uses GELU: worth the cost for better gradients")
    print("   - Sigmoid/Tanh: reserved for output layers or gates")
    print("=" * 60)

# Run the analysis
if __name__ == "__main__":
    analyze_activation_performance()

# %% [markdown]
"""
## ⭐ Aha Moment: Activations Add Intelligence

**What you built:** Five activation functions that introduce nonlinearity to neural networks.

**Why it matters:** Without activations, stacking layers would just be matrix multiplication -
a linear operation. ReLU's simple "zero out negatives" rule is what allows networks to learn
complex patterns like recognizing faces or understanding language.

Your activations are ready to be combined with Linear layers in Module 03!
"""

# %%
def demo_activations():
    """See how activations transform data."""
    print("AHA MOMENT: Activations Add Intelligence")
    print("=" * 45)

    # Test input with positive and negative values
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    print(f"Input:   {x.data}")

    # ReLU - zeros out negatives
    relu = ReLU()
    relu_out = relu(x)
    print(f"ReLU:    {relu_out.data}")
    print("         Negatives become 0, positives unchanged!")

    # Sigmoid - squashes to (0, 1)
    sigmoid = Sigmoid()
    sigmoid_out = sigmoid(x)
    print(f"\nSigmoid: {np.round(sigmoid_out.data, 2)}")
    print("         All values squashed to (0, 1) range!")

    # Softmax - probability distribution
    softmax = Softmax()
    softmax_out = softmax(x)
    print(f"\nSoftmax: {np.round(softmax_out.data, 3)}")
    print(f"         Sum = {softmax_out.data.sum():.1f} (valid probability distribution!)")

    print("\n" + "=" * 45)
    print("Activations add nonlinearity - the key to deep learning!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_activations()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: Activations

Congratulations! You've built the intelligence engine of neural networks!

### Key Accomplishments
- **Built 5 core activation functions** with distinct behaviors and use cases
- **Implemented forward passes** for Sigmoid, ReLU, Tanh, GELU, and Softmax
- **Discovered computational cost differences** between activations (ReLU fastest)
- **Handled numerical stability** with clipping and max subtraction techniques
- **All tests pass** (validated by `test_module()`)

"""