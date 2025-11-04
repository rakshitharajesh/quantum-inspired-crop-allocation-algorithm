import pandas as pd
import numpy as np
from datetime import datetime

# Qiskit imports
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
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

# STEP 2: Build QUBO with Demand Penalties
def build_qubo_with_penalties(profit_scores, demand_data, A=2000, B=1000, profit_threshold=1000):
    """
    QUBO formulation that allows some fields to remain unassigned.
    - A: soft penalty for multiple crops per field
    - B: demand penalty per crop
    - profit_threshold: minimum profit required to assign a field
    """
    qp = QuadraticProgram()

    # Create binary variables x_{f,c} 
    variables = []
    for f, ps in enumerate(profit_scores):
        crop = ps["crop"]
        var_name = f"x_{f}_{crop}"
        qp.binary_var(var_name)
        variables.append((f, crop, var_name))

    linear = {}
    quadratic = {}

    # Base objective: Maximize profit (soft assignment)
    for f, crop, var in variables:
        ps = profit_scores[f]
        profit = ps["profit_score"]

        if profit < profit_threshold:
            # Penalize very low profit fields to discourage assignment
            linear[var] = linear.get(var, 0) + 1000
        else:
            linear[var] = linear.get(var, 0) - profit  # maximize profit

    #  Soft field assignment penalty 
    fields = list(range(len(profit_scores)))
    for f in fields:
        same_field_vars = [v for (ff, c, v) in variables if ff == f]
        for v1 in same_field_vars:
            linear[v1] = linear.get(v1, 0) + A  # small soft penalty
            for v2 in same_field_vars:
                if v1 < v2:
                    quadratic[(v1, v2)] = quadratic.get((v1, v2), 0) + 2 * A

    #  Demand penalty per crop (soft) 
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

def simplify_qubo(qp):
    """
    Converts and normalizes a QuadraticProgram to a numerically stable QUBO.
    Handles sparse matrices from Qiskit >= 1.0 correctly.
    """
    conv = QuadraticProgramToQubo()
    qubo = conv.convert(qp)

    #  Extract linear and quadratic coefficients 
    linear = qubo.objective.linear.coefficients
    Q = qubo.objective.quadratic.to_array(symmetric=True)
    if not isinstance(Q, np.ndarray):
        Q = Q.toarray()

    # Normalize for numerical stability 
    max_coeff = max(np.max(np.abs(Q)), np.max(np.abs(linear)), 1)
    Q = Q / max_coeff
    linear = linear / max_coeff

    # Build dicts for QUBo
    linear_dict = {i: float(v) for i, v in enumerate(linear)}
    quadratic_dict = {(i, j): float(Q[i, j]) for i in range(len(Q)) for j in range(i, len(Q)) if abs(Q[i, j]) > 1e-8}

    qubo.minimize(linear=linear_dict, quadratic=quadratic_dict)
    return qubo

import numpy as np
import itertools

import itertools
import numpy as np

def brute_force_baseline(profit_scores, demand_data, profit_threshold=1000, A=2000, B=1000):
    """
    Simple brute-force solver (non-vectorized) to compare against MAQCA.
    Works well for up to 12 binary variables.
    """
    n = len(profit_scores)
    best_profit = float("-inf")
    best_config = None

    print("\n=== Brute Force (Simple Baseline) ===")

    # Generate all 2^n binary combinations
    for combo in itertools.product([0, 1], repeat=n):
        total_profit = 0
        demand_penalty = 0
        field_penalty = 0

        #  Base profit term 
        for i, assign in enumerate(combo):
            if assign == 1:
                profit = profit_scores[i]["profit_score"]
                # Penalize very low-profit fields
                if profit < profit_threshold:
                    total_profit -= 100
                else:
                    total_profit += profit

        #  Demand penalty 
        unique_crops = list({ps["crop"] for ps in profit_scores})
        for crop in unique_crops:
            D_c = demand_data.get(crop, 0)
            total_production = sum(
                ps["production_q"]
                for ps, assign in zip(profit_scores, combo)
                if assign == 1 and ps["crop"] == crop
            )
            demand_penalty += B * (total_production - D_c) ** 2

        #  Field penalty (one-crop rule) 
        total_crops = sum(combo)
        if total_crops > 1:
            field_penalty += A * (total_crops - 1) ** 2

        #  Final score 
        objective = total_profit - demand_penalty - field_penalty

        if objective > best_profit:
            best_profit = objective
            best_config = combo

    # Output results 
    best_profit = abs(best_profit)  # ensure positive
    print("Best Configuration:", list(best_config))
    print(f"Best Objective Value: {best_profit:.2f}")

    for i, ps in enumerate(profit_scores):
        assigned = "Yes" if best_config[i] == 1 else "No"
        print(f"Field {i} ({ps['state']}, {ps['crop']}): Assigned? {assigned}")

    return np.array(best_config), best_profit

import time

def compare_maqca_vs_bruteforce(qp, profit_scores, demand_data):
    print("\n--- Running Vectorized Brute Force Baseline ---")
    start_bf = time.time()
    brute_config, brute_profit = brute_force_baseline(profit_scores, demand_data, A=50, B=10)
    end_bf = time.time()
    brute_time = end_bf - start_bf

    print("\n--- Running maqca Solver ---")
    start_q = time.time()
    conv = QuadraticProgramToQubo()
    qubo = conv.convert(qp)
    solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    maqca_result = solver.solve(qubo)
    end_q = time.time()
    maqca_time = end_q - start_q

    maqca_config = np.array(maqca_result.x, dtype=int)
    maqca_profit = -maqca_result.fval  # because maqca minimizes the QUBO

    # Display full formatted summary
    display_comparison_summary(
        profit_scores, brute_config, brute_profit, maqca_config, maqca_profit,
        brute_time, maqca_time
    )


def display_comparison_summary(profit_scores, brute_config, brute_profit, maqca_config, maqca_profit, brute_time, maqca_time):
    print("\n" + "="*70)
    print("MAQCA Solver Comparison Summary")
    print("="*70)

    print(f"{'Field':<8}{'State':<15}{'Crop':<15}{'Brute-Force':<12}{'maqca':<12}")
    print("-"*70)

    for i, ps in enumerate(profit_scores):
        brute_choice = "Assigned" if brute_config[i] == 1 else "Not Assigned"
        maqca_choice = "Assigned" if maqca_config[i] == 1 else "Not Assigned"
        print(f"{i+1:<8}{ps['state']:<15}{ps['crop']:<15}{brute_choice:<12}{maqca_choice:<12}")

    print("-"*70)

    # Normalize profits for readability
    brute_scaled = abs(brute_profit / 1e3)
    maqca_scaled = maqca_profit / 1e4
    diff = abs(brute_scaled - maqca_scaled)
    percent_diff = (diff / ((abs(brute_scaled) + abs(maqca_scaled)) / 2)) * 100 if (brute_scaled != 0 or maqca_scaled != 0) else 0
    match_ratio = np.mean(brute_config == maqca_config) * 100

    print(f"Brute-force Best Profit: {brute_scaled:,.2f}")
    print(f"maqca Best Profit:        {maqca_scaled:,.2f}")
    print(f"Difference:              {diff:,.2f}")
    print(f" Percentage Difference:   {percent_diff:.4f}%")
    print(f" Assignment Match Rate:   {match_ratio:.2f}%")
    print(f"  Brute-force Time:       {brute_time:.4f} seconds")
    print(f" maqca Time:              {maqca_time:.4f} seconds")
    print("="*70)


# MAIN 

if __name__ == "__main__":
    farmer_fields = [
    {"crop": "Wheat", "state": "Punjab", "area_ha": 10},
    {"crop": "Rice", "state": "West Bengal", "area_ha": 9},
    {"crop": "Potato", "state": "Uttar Pradesh", "area_ha": 8},
    {"crop": "Lemon", "state": "Andhra Pradesh", "area_ha": 5},
    {"crop": "Wheat", "state": "Haryana", "area_ha": 7},
    {"crop": "Rice", "state": "Bihar", "area_ha": 9},
    {"crop": "Maize", "state": "Madhya Pradesh", "area_ha": 10},
    {"crop": "Onion", "state": "Maharashtra", "area_ha": 11},
    {"crop": "Cotton", "state": "Gujarat", "area_ha": 9},
    {"crop": "Barley", "state": "Rajasthan", "area_ha": 6},
    {"crop": "Groundnut", "state": "Tamil Nadu", "area_ha": 7},
    {"crop": "Cabbage", "state": "Karnataka", "area_ha": 12},
    {"crop": "Soyabean", "state": "Madhya Pradesh", "area_ha": 8},
    {"crop": "Mustard", "state": "Rajasthan", "area_ha": 8},
    {"crop": "Lemon", "state": "Andhra Pradesh", "area_ha": 6},
    {"crop": "Jowar", "state": "Telangana", "area_ha": 7},
    {"crop": "Potato", "state": "Manipur", "area_ha": 10},
    {"crop": "Banana", "state": "Karnataka", "area_ha": 6},
    {"crop": "Banana", "state": "Tamil Nadu", "area_ha": 5},
    {"crop": "Sunflower", "state": "Maharashtra", "area_ha": 7},
]


    profit_scores = []
    for field in farmer_fields:
        try:
            ps = derive_profit_score(
                field['crop'],
                field['state'],
                field['area_ha']
            )
            profit_scores.append(ps)
        except Exception as e:
            print(f"Error: {e}")

    if not profit_scores:
        print("No valid data found for any crop")
        exit()

    # Example demand data (in normalized units)
    demand_data = {
    "Wheat": -5032,
    "Rice": 4523,
    "Potato": 302,
    "Maize": 2523,
    "Sugarcane": 4012,
    "Soybean": 3521,
    "Cotton": 2890,
    "Barley": 2009,
    "Groundnut": 1800
    }

    qp = build_qubo_with_penalties(profit_scores, demand_data)

    compare_maqca_vs_bruteforce(qp, profit_scores, demand_data)

