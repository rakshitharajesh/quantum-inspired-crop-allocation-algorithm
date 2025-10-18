import pandas as pd
import numpy as np
from datetime import datetime
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# Importing functions from other scripts
from expected_price import compute_season_avg_and_trend
from yield_calculation import yield_for_crop


# ======================================================
# STEP 1: Profit Score Calculation
# ======================================================
def derive_profit_score(crop_name, state, area_ha=10):
    mandi = pd.read_csv("data/mandi_prices.csv")
    yield_df = pd.read_csv("data/yield_data.csv")

    price_result = compute_season_avg_and_trend(mandi, crop_name)
    if price_result['expected_price'] is None:
        raise ValueError("No price data available for this crop")

    expected_price = price_result['expected_price']
    seasonal_avg = price_result['seasonal_avg']

    yield_q_he = yield_for_crop(yield_df, crop_name, state)
    if np.isnan(yield_q_he) or yield_q_he <= 0:
        raise ValueError("No yield data for this crop/state")

    production_q = yield_q_he * area_ha
    profit_score = production_q * expected_price

    print(f"Crop: {crop_name}, State: {state}")
    print(f"Expected Price: ₹{expected_price:.2f}, Yield: {yield_q_he:.2f} q/ha")
    print(f"Production: {production_q:.2f} q, Profit Score: {profit_score:.2f}")

    return {
        "crop": crop_name,
        "state": state,
        "expected_price": expected_price,
        "yield_q_he": yield_q_he,
        "production_q": production_q,
        "profit_score": profit_score
    }


# ======================================================
# STEP 2: Build QUBO with Demand Penalties
# ======================================================
def build_qubo_with_penalties(profit_scores, demand_data, A=2000, B=1000, profit_threshold=1000):
    """
    QUBO formulation that allows some fields to remain unassigned.
    - A: soft penalty for multiple crops per field
    - B: demand penalty per crop
    - profit_threshold: minimum profit required to assign a field
    """
    from qiskit_optimization import QuadraticProgram
    qp = QuadraticProgram()

    # ---- Create binary variables x_{f,c} ----
    variables = []
    for f, ps in enumerate(profit_scores):
        crop = ps["crop"]
        var_name = f"x_{f}_{crop}"
        qp.binary_var(var_name)
        variables.append((f, crop, var_name))

    linear = {}
    quadratic = {}

    # ---- Base objective: Maximize profit (soft assignment) ----
    for f, crop, var in variables:
        ps = profit_scores[f]
        profit = ps["profit_score"]
        if profit < profit_threshold:
            # Penalize very low profit fields to discourage assignment
            linear[var] = linear.get(var, 0) + 1000  # positive => energy increases
        else:
            linear[var] = linear.get(var, 0) - profit  # maximize profit

    # ---- Soft field assignment penalty ----
    # Smaller A allows fields to remain unassigned
    fields = list(range(len(profit_scores)))
    for f in fields:
        same_field_vars = [v for (ff, c, v) in variables if ff == f]
        for v1 in same_field_vars:
            linear[v1] = linear.get(v1, 0) + A  # small soft penalty
            for v2 in same_field_vars:
                if v1 < v2:
                    quadratic[(v1, v2)] = quadratic.get((v1, v2), 0) + 2 * A

    # ---- Demand penalty per crop (soft) ----
    crops = list({ps["crop"] for ps in profit_scores})
    for crop in crops:
        D_c = demand_data.get(crop, 0)
        crop_vars = [(f, v) for (f, c, v) in variables if c == crop]
        for (f1, v1) in crop_vars:
            w1 = profit_scores[f1]["production_q"]
            linear[v1] = linear.get(v1, 0) + B * (w1**2 - 2 * D_c * w1)
            quadratic[(v1, v1)] = quadratic.get((v1, v1), 0) + B * (w1**2)
            for (f2, v2) in crop_vars:
                if v1 < v2:
                    w2 = profit_scores[f2]["production_q"]
                    quadratic[(v1, v2)] = quadratic.get((v1, v2), 0) + 2 * B * w1 * w2

    # ---- Build QUBO ----
    qp.minimize(linear=linear, quadratic=quadratic)
    return qp

from qiskit_optimization.converters import QuadraticProgramToQubo
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix

def simplify_qubo(qp):
    """
    Converts and normalizes a QuadraticProgram to a numerically stable QUBO.
    Handles sparse matrices from Qiskit >= 1.0 correctly.
    """
    conv = QuadraticProgramToQubo()
    qubo = conv.convert(qp)

    # --- Extract linear coefficients ---
    linear = qubo.objective.linear.coefficients  # numpy array already

    # --- Extract quadratic coefficients (handle sparse case) ---
    Q = qubo.objective.quadratic.to_array(symmetric=True)

    # If Q is still a sparse matrix (like dok_matrix), convert to dense ndarray
    if not isinstance(Q, np.ndarray):
        Q = Q.toarray()

    # --- Normalize for numerical stability ---
    max_coeff = max(np.max(np.abs(Q)), np.max(np.abs(linear)), 1)
    Q = Q / max_coeff
    linear = linear / max_coeff

    # --- Build dicts for QUBO ---
    linear_dict = {i: float(v) for i, v in enumerate(linear)}
    quadratic_dict = {
        (i, j): float(Q[i, j])
        for i in range(len(Q))
        for j in range(i, len(Q))
        if abs(Q[i, j]) > 1e-8
    }

    # --- Recreate normalized QUBO ---
    qubo.minimize(linear=linear_dict, quadratic=quadratic_dict)

    return qubo

# ======================================================
# STEP 3: Solve with QAOA
# ======================================================
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import BaseSamplerV2
import numpy as np

def run_qaoa_(qp):
    conv = QuadraticProgramToQubo()
    qubo = conv.convert(qp)

    print("\n=== QUBO Formulation ===")
    print(qubo.prettyprint())

    try:
        # ---- Step 1: Set up quantum backend ----
        estimator = EstimatorV2()  # uses Aer simulator backend

        # ---- Step 2: Set up optimizer (COBYLA is derivative-free) ----
        optimizer = COBYLA(maxiter=100, tol=1e-3)

        # ---- Step 3: Initialize QAOA with depth p=2 (number of layers) ----
        qaoa = QAOA(estimator, optimizer=optimizer, reps=2)

        # ---- Step 4: Wrap QAOA in a MinimumEigenOptimizer ----
        qaoa_solver = MinimumEigenOptimizer(qaoa)

        # ---- Step 5: Solve the QUBO problem ----
        result = qaoa_solver.solve(qubo)

        # ---- Step 6: Display results ----
        print("\n✅ QAOA optimization complete!")
        print("Optimal bitstring (assignments):", result.x)
        print("Objective value (energy):", result.fval)

        print("\n=== Assignment Summary ===")
        for i, ps in enumerate(profit_scores):
            assigned = "Yes" if result.x[i] == 1 else "No"
            print(f"Field {i}: {ps['crop']} in {ps['state']} — Assigned? {assigned}")

        return result

    except Exception as e:
        print("❌ QAOA execution failed:\n", e)

# =================================================
# MAIN EXECUTION
# ======================================================
if __name__ == "__main__":
    farmer_fields = [
    {"crop": "Wheat", "state": "Punjab", "area_ha": 8},
    {"crop": "Rice", "state": "West Bengal", "area_ha": 10},
    {"crop": "Potato", "state": "Uttar Pradesh", "area_ha": 6},
    {"crop": "Wheat", "state": "Haryana", "area_ha": 7},
    {"crop": "Rice", "state": "Bihar", "area_ha": 9},
]

    profit_scores = []
    for field in farmer_fields:
        try:
            ps = derive_profit_score(field['crop'], field['state'], field['area_ha'])
            profit_scores.append(ps)
        except Exception as e:
            print(f"Error: {e}")

    if not profit_scores:
        print("No valid data found for any crop")
        exit()

    # Example demand data (in normalized units)
    demand_data = {
        "Wheat": 50000,
        "Rice": -1000,
        "Potato": 30000
    }

    qp = build_qubo_with_penalties(profit_scores, demand_data)
    run_qaoa_(qp)
    

