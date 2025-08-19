from .optimizers.componentwise_bayesian_optimization import ComponentwiseOptunaOptimizer

# Create aliases for backward compatibility and different naming conventions
ComponentwiseOptimizer = ComponentwiseOptunaOptimizer
ComponentwiseOptunaOptimizer = ComponentwiseOptunaOptimizer

__all__ = [
    'ComponentwiseBayesianOptimizer',
    'ComponentwiseOptimizer', 
    'ComponentwiseOptunaOptimizer'
]