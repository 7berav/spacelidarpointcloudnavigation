import numpy as np
from scipy.optimize import fsolve
from sympy import symbols, expand, lambdify

# Assume terms are generated externally
# def generate_homogeneous_terms(order): ...
# def calculate_polynomial_matrix(points, terms): ...
# def build_polynomial_expression(terms, beta): ...

def calculate_polynomial_matrix(points, symb_terms):
    x, y, z = symbols('x y z')
    funcs = [lambdify((x, y, z), term, 'numpy') for term in symb_terms]
    A = np.zeros((points.shape[0], len(symb_terms)))
    for i, func in enumerate(funcs):
        A[:, i] = func(points[:, 0], points[:, 1], points[:, 2])
    return A

def build_polynomial_expression(symb_terms, beta):
    return sum(b * t for b, t in zip(beta, symb_terms))

def fit_closed_surface(points, terms, num_iterations=60):
    """
    points: (N, 3) numpy array
    terms: list of (i, j, k) tuples representing homogeneous polynomial terms
    """
    x, y, z = symbols('x y z')
    center_shift = np.mean(points, axis=0)
    points_shift = points - center_shift

    # Step 1: Initial least squares fit
    A = calculate_polynomial_matrix(points_shift, terms)
    target = np.ones((A.shape[0], 1))
    beta = np.linalg.lstsq(A, target, rcond=None)[0].flatten()

    # Step 2: Symbolic polynomial expression
    f_expr = build_polynomial_expression(terms, beta)

    # Step 3: Gradient-based center shift refinement
    a, b, c = symbols('a b c')
    f_shift = f_expr.subs({x: x + a, y: y + b, z: z + c})
    f_expanded = expand(f_shift - 1)
    coeffs = f_expanded.as_coefficients_dict()
    nonhomogeneous_terms = [term for term in coeffs if term.as_poly().total_degree() != 6 and term != 1]
    f_obj = sum([coeffs[t]**2 for t in nonhomogeneous_terms])
    grad = [f_obj.diff(var) for var in (a, b, c)]
    grad_func = lambdify((a, b, c), grad, modules='numpy')

    def shift_objective(v): return grad_func(*v)
    shift_solution = fsolve(shift_objective, [0, 0, 0])
    points_shift = points_shift - shift_solution

    # Step 4: Iterative regression shift refinement
    for i in range(num_iterations):
        A = calculate_polynomial_matrix(points_shift, terms)
        beta = np.linalg.lstsq(A, np.ones((A.shape[0], 1)), rcond=None)[0].flatten()
        f_expr = build_polynomial_expression(terms, beta)

        f_func = lambdify((x, y, z), f_expr, modules='numpy')
        grad_fx = lambdify((x, y, z), f_expr.diff(x), modules='numpy')
        grad_fy = lambdify((x, y, z), f_expr.diff(y), modules='numpy')
        grad_fz = lambdify((x, y, z), f_expr.diff(z), modules='numpy')

        values = f_func(points_shift[:, 0], points_shift[:, 1], points_shift[:, 2])
        grads = np.stack([
            grad_fx(points_shift[:, 0], points_shift[:, 1], points_shift[:, 2]),
            grad_fy(points_shift[:, 0], points_shift[:, 1], points_shift[:, 2]),
            grad_fz(points_shift[:, 0], points_shift[:, 1], points_shift[:, 2])
        ], axis=1)

        numerators = (values - 1).reshape(-1, 1) * grads
        denominators = np.sum(grads**2, axis=1, keepdims=True) + 1e-12
        shifts = numerators / denominators
        shift_vector = -np.mean(shifts, axis=0)
        points_shift = points_shift + shift_vector  # move points towards f=1 level set

    return beta, f_expr, center_shift + shift_solution
