import numpy as np
from autograd import Tensor
from painn import CompletePaiNN
from functools import partial
import cProfile
import pstats
import io
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def profile(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        logger.debug(f"Profile for {func.__name__}:\n{s.getvalue()}")
        return result
    return wrapper

@profile
def test_complete_painn():
    logger.info("Testing CompletePaiNN")
    num_features = 64
    num_layers = 3
    num_rbf = 20
    cutoff = 5.0
    num_scalar_outputs = 1  # e.g., energy
    num_vector_outputs = 1  # e.g., forces

    complete_painn = CompletePaiNN(num_features, num_layers, num_rbf, cutoff, 
                                   num_scalar_outputs, num_vector_outputs)

    num_nodes = 10
    num_edges = 15
    s = Tensor(np.random.randn(num_nodes, num_features))
    v = Tensor(np.random.randn(num_nodes, num_features, 3))
    edge_index = Tensor(np.random.randint(0, num_nodes, size=(2, num_edges)))
    edge_attr = Tensor(np.random.rand(num_edges, 3))

    scalar_pred, vector_pred = complete_painn(s, v, edge_index, edge_attr)
    
    logger.info(f"CompletePaiNN scalar output shape: {scalar_pred.shape}")
    logger.info(f"CompletePaiNN vector output shape: {vector_pred.shape}")
    
    assert scalar_pred.shape == (num_nodes, num_scalar_outputs), "Scalar output shape mismatch"
    assert vector_pred.shape == (num_nodes, num_vector_outputs, 3), "Vector output shape mismatch"

if __name__ == "__main__":
    test_complete_painn()
    logger.info("All tests passed!")
