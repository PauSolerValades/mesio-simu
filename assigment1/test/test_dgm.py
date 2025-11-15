"""
This modules contains testing 
functionality for the dgm.py module
"""

from src.dgm import WeibullDistribution

def test_reproducibility():
    """
    Our Weibull class should ensure reproducibility in the
    sense that each time the class is instantiated, the random
    state should be reset and fixed to the provided seed.
    This way, if I create two instances of the class, and sample
    for the same size, I should get EXACTLY the same sequence
    of numbers.
    """
    alpha = 2.0
    beta = 0.8
    seed = 1234
    wb_1 = WeibullDistribution(alpha=alpha, beta=beta, seed=seed)

    sample_1 = wb_1.sample(100)
    sample_1 = tuple(sample_1) # Make the result immutable

    wb_2 = WeibullDistribution(alpha=alpha, beta=beta, seed=seed)

    sample_2 = wb_2.sample(100)
    sample_2 = tuple(sample_2) 

    assert sample_1 == sample_2 # Compare two samples position by position

