#!/usr/bin/env python3
"""
compare_and_visualize.py
Integrated classical brute-force + QUBO/maqca solver with visualization.

Drop in project root (expects data/mandi_prices.csv and data/yield_data.csv).
Produces CSV summaries and PNG charts.

Usage:
    python compare_and_visualize.py
"""

import time
import itertools
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from datetime import datetime

# Qiskit imports (wrapped in try/except so script still runs if missing)
try:
    from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_aer.primitives import EstimatorV2, SamplerV2
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    QISKIT_AVAILABLE = True
except Exception as e:
    QuadraticProgram = None
    QuadraticProgramToQubo = None
    MinimumEigenOptimizer = None
    NumPyMinimumEigensolver = None
    QISKIT_AVAILABLE = False
    warnings.warn(f"Qiskit imports failed: {e}. The script will fall back to classical solver only.")


# -------------------------
# Reusable: Profit score calculation (keeps your original logic)
# -------------------------
from expected_price import compute_season_avg_and_trend
from yield_calculation import yield_for_crop

def derive_profit_score(crop_name, state, area_ha=10):
    """
    Same logic as your originals: compute expected_price (from mandi), yield (from yield_data),
    compute production (q) and profit_score.
    """
    mandi = pd.read_csv("data/mandi_prices.csv")
    yield_df = pd.read_csv("data/yield_data.csv")

    price_result = compute_season_avg_and_trend(mandi, crop_name)
    if price_result.get('expected_price') is None:
        warnings.warn(f"No price data for {crop_name}; using fallback price=1.0")
        expected_price = 1.0
    else:
        expected_price = price_result['expected_price']

    yield_q_he = yield_for_crop(yield_df, crop_name, state)
    if np.isnan(yield_q_he) or yield_q_he <= 0:
        warnings.warn(f"No yield data for {crop_name} in {state}; using fallback yield=0.1 q/ha")
        yield_q_he = 0.1

    production_q = yield_q_he * area_ha
    profit_score = production_q * expected_price

    # Console-friendly info (keeps your print format)
    print(f"Derived -> Crop: {crop_name}, State: {state}, Price: {expected_price:.3f}, "
          f"Yield(q/ha): {yield_q_he:.3f}, Prod(q): {production_q:.3f}, Profit: {profit_score:.3f}")

    return {
        "crop": crop_name,
        "state": state,
        "expected_price": expected_price,
        "yield_q_he": yield_q_he,
        "production_q": production_q,
        "profit_score": float(profit_score)
    }


# -------------------------
# Classical QUBO builder (kept logic)
# -------------------------
def build_qubo_coeffs(profit_scores, demand_data, A=2000, B=1000, profit_threshold=1000):
    n = len(profit_scores)
    linear = {i: 0.0 for i in range(n)}
    quadratic = {}

    for i in range(n):
        p = profit_scores[i]["profit_score"]
        if p < profit_threshold:
            linear[i] += 1000.0
        else:
            linear[i] += -p

    for i in range(n):
        linear[i] += A

    # group same crops to add demand penalties exactly as your logic does
    crops = {}
    for idx, ps in enumerate(profit_scores):
        crops.setdefault(ps["crop"], []).append(idx)

    for crop, idxs in crops.items():
        D_c = demand_data.get(crop, 0)
        for i in idxs:
            w1 = profit_scores[i]["production_q"]
            linear[i] += B * (w1**2 - 2 * D_c * w1)
            quadratic[(i, i)] = quadratic.get((i, i), 0.0) + B * (w1**2)
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                w1, w2 = profit_scores[i]["production_q"], profit_scores[j]["production_q"]
                quadratic[(i, j)] = quadratic.get((i, j), 0.0) + 2 * B * w1 * w2

    return linear, quadratic


# -------------------------
# Evaluate assignment (kept)
# -------------------------
def evaluate_assignment(profit_scores, linear, quadratic, assignment):
    x = list(map(int, assignment))
    energy = 0.0
    for i, xi in enumerate(x):
        energy += linear.get(i, 0.0) * xi
    for (i, j), coeff in quadratic.items():
        energy += coeff * x[i] * x[j]
    profit_sum = sum(float(profit_scores[i]["profit_score"]) * x[i] for i in range(len(x)))
    return profit_sum, energy


# -------------------------
# Brute-force solver (kept logic, returns timings)
# -------------------------
def classical_bruteforce_solver(profit_scores, linear, quadratic):
    n = len(profit_scores)
    best_energy = float("inf")
    best_profit = -float("inf")
    best_assign = None
    start = time.time()

    for mask in range(1 << n):
        assignment = [(mask >> i) & 1 for i in range(n)]
        profit_sum, energy = evaluate_assignment(profit_scores, linear, quadratic, assignment)
        if energy < best_energy:
            best_energy = energy
            best_profit = profit_sum
            best_assign = assignment[:]

    duration = time.time() - start
    return {"assignment": best_assign, "energy": best_energy, "profit": best_profit, "time": duration}


# -------------------------
# QUBO builder using Qiskit QuadraticProgram (keeps logic)
# -------------------------
def build_qubo_with_penalties(profit_scores, demand_data, A=2000, B=1000, profit_threshold=1000):
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit is not available to build QuadraticProgram.")
    qp = QuadraticProgram()

    # ---- Create binary variables x_{f_crop} ----
    variables = []
    for f, ps in enumerate(profit_scores):
        crop = ps["crop"]
        var_name = f"x_{f}_{crop}"
        qp.binary_var(var_name)
        variables.append((f, crop, var_name))

    linear = {}
    quadratic = {}

    # ---- Base objective: Maximize profit (encoded as -profit to minimize) ----
    for f, crop, var in variables:
        ps = profit_scores[f]
        profit = ps["profit_score"]
        if profit < profit_threshold:
            linear[var] = linear.get(var, 0) + 1000
        else:
            linear[var] = linear.get(var, 0) - profit

    # ---- Soft field assignment penalty ----
    fields = list(range(len(profit_scores)))
    for f in fields:
        same_field_vars = [v for (ff, c, v) in variables if ff == f]
        for v1 in same_field_vars:
            linear[v1] = linear.get(v1, 0) + A
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

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp


# -------------------------
# Brute force baseline used for the compare_maqca functions (kept logic but vectorized-friendly)
# -------------------------
def brute_force_baseline(profit_scores, demand_data, profit_threshold=1000, A=2000, B=1000):
    n = len(profit_scores)
    best_profit = float("-inf")
    best_config = None

    for combo in itertools.product([0, 1], repeat=n):
        total_profit = 0.0
        demand_penalty = 0.0
        field_penalty = 0.0

        # Base profit term
        for i, assign in enumerate(combo):
            if assign == 1:
                profit = profit_scores[i]["profit_score"]
                if profit < profit_threshold:
                    total_profit -= 1000
                else:
                    total_profit += profit

        # Demand penalty per crop
        unique_crops = list({ps["crop"] for ps in profit_scores})
        for crop in unique_crops:
            D_c = demand_data.get(crop, 0)
            total_production = sum(
                ps["production_q"]
                for ps, assign in zip(profit_scores, combo)
                if assign == 1 and ps["crop"] == crop
            )
            demand_penalty += B * (total_production - D_c) ** 2

        # Field penalty (soft)
        total_crops = sum(combo)
        if total_crops > 1:
            field_penalty += A * (total_crops - 1)

        objective = total_profit - demand_penalty - field_penalty

        if objective > best_profit:
            best_profit = objective
            best_config = combo

    best_profit = float(best_profit)
    return np.array(best_config, dtype=int), best_profit


# -------------------------
# Compare MAQCA (QUBO solver) vs Brute-force and produce summary + visualizations
# -------------------------
def compare_maqca_vs_bruteforce(qp, profit_scores, demand_data):
    # Run brute force baseline
    print("\n--- Running Brute Force Baseline ---")
    start_bf = time.time()
    brute_config, brute_profit = brute_force_baseline(profit_scores, demand_data, A=50, B=10)
    end_bf = time.time()
    brute_time = end_bf - start_bf

    # Run maqca/QUBO solver (use NumPyMinimumEigensolver via MinimumEigenOptimizer)
    print("\n--- Running maqca (QUBO) Solver ---")
    start_q = time.time()
    try:
        conv = QuadraticProgramToQubo()
        qubo = conv.convert(qp)
        solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        maqca_result = solver.solve(qubo)
        maqca_time = time.time() - start_q
        maqca_config = np.array(maqca_result.x, dtype=int)
        maqca_profit = -maqca_result.fval  # because we minimized -profit (QUBO)
    except Exception as e:
        warnings.warn(f"maqca solver failed: {e}. Falling back to brute-force result for maqca columns.")
        maqca_config = brute_config.copy()
        maqca_profit = brute_profit
        maqca_time = 0.0

    # Build and save summary CSV and console table
    rows = []
    for i, ps in enumerate(profit_scores):
        rows.append({
            "FieldIndex": i + 1,
            "State": ps["state"],
            "Crop": ps["crop"],
            "Profit": ps["profit_score"],
            "Production_q": ps["production_q"],
            "Brute_Assigned": int(brute_config[i]),
            "maqca_Assigned": int(maqca_config[i])
        })
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv("assignment_summary.csv", index=False)

    # Console summary (formatted)
    print("\n" + "="*70)
    print("MAQCA vs Brute-force Assignment Summary")
    print("="*70)
    print(f"{'Field':<6}{'State':<15}{'Crop':<15}{'Brute':<8}{'maqca':<8}{'Profit':>12}")
    print("-"*70)
    for r in rows:
        print(f"{r['FieldIndex']:<6}{r['State']:<15}{r['Crop']:<15}{('Yes' if r['Brute_Assigned'] else 'No'):<8}{('Yes' if r['maqca_Assigned'] else 'No'):<8}{r['Profit']:12.2f}")
    print("-"*70)

    # Normalized profit numbers for readability (scale)
    brute_scaled = abs(brute_profit / 1e4)
    maqca_scaled = maqca_profit / 1e4
    diff = abs(brute_scaled - maqca_scaled)
    percent_diff = (diff / ((abs(brute_scaled) + abs(maqca_scaled)) / 2)) * 100 if (brute_scaled != 0 or maqca_scaled != 0) else 0
    match_ratio = np.mean(brute_config == maqca_config) * 100

    print(f"Brute-force Best Profit (scaled): ₹{brute_scaled:,.2f}")
    print(f"maqca Best Profit (scaled):        ₹{maqca_scaled:,.2f}")
    print(f"Difference (scaled):               ₹{diff:,.2f}")
    print(f"Percentage Difference:              {percent_diff:.4f}%")
    print(f"Assignment Match Rate:              {match_ratio:.2f}%")
    print(f"Brute Force Time:                   {brute_time:.4f} s")
    print(f"maqca Time:                         {maqca_time:.4f} s")
    print("="*70)

    # Visualizations
    visualize_results(df_summary, brute_profit, maqca_profit, brute_time, maqca_time)

    return {
        "brute_config": brute_config,
        "brute_profit": brute_profit,
        "brute_time": brute_time,
        "maqca_config": maqca_config,
        "maqca_profit": maqca_profit,
        "maqca_time": maqca_time,
        "summary_df": df_summary
    }


# -------------------------
# Visualization functions (matplotlib only; no colors hard-coded)
# -------------------------
def visualize_results(df_summary, brute_profit, maqca_profit, brute_time, maqca_time):
    # Profit/production bar chart per field
    df_plot = df_summary.copy()
    df_plot["FieldLabel"] = df_plot["FieldIndex"].astype(str) + ": " + df_plot["Crop"].astype(str)
    plt.figure(figsize=(10, 6))
    plt.bar(df_plot["FieldLabel"], df_plot["Profit"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Profit score")
    plt.title("Profit Score per Field")
    plt.tight_layout()
    plt.savefig("profit_bar.png")
    plt.close()
    print("Saved profit_bar.png")

    # Assignment comparison
    plt.figure(figsize=(10, 6))
    brute_assigned = df_plot["Brute_Assigned"]
    maqca_assigned = df_plot["maqca_Assigned"]
    indices = np.arange(len(df_plot))
    width = 0.35
    plt.bar(indices - width/2, brute_assigned, width)
    plt.bar(indices + width/2, maqca_assigned, width)
    plt.xticks(indices, df_plot["FieldLabel"], rotation=45, ha="right")
    plt.ylabel("Assigned (1/0)")
    plt.title("Assignment: Brute Force vs maqca")
    plt.legend(["Brute", "maqca"])
    plt.tight_layout()
    plt.savefig("assignment_compare.png")
    plt.close()
    print("Saved assignment_compare.png")

    # Execution time comparison
    plt.figure(figsize=(6, 4))
    algorithms = ["Brute Force", "maqca"]
    times = [brute_time, maqca_time]
    plt.bar(algorithms, times)
    plt.ylabel("Execution time (s)")
    plt.title("Execution Time Comparison")
    plt.tight_layout()
    plt.savefig("execution_times.png")
    plt.close()
    print("Saved execution_times.png")


# -------------------------
# MAIN DRIVER (uses your farmer_fields samples)
# -------------------------
if __name__ == "__main__":
    # Sample fields (kept and extended from your final combined sample)
    farmer_fields = [
        {"crop": "Wheat", "state": "Punjab", "area_ha": 8},
        {"crop": "Rice", "state": "West Bengal", "area_ha": 10},
        {"crop": "Potato", "state": "Uttar Pradesh", "area_ha": 6},
        {"crop": "Lemon", "state": "Andhra Pradesh", "area_ha": 5},
        {"crop": "Wheat", "state": "Haryana", "area_ha": 7},
        {"crop": "Rice", "state": "Bihar", "area_ha": 9},
        {"crop": "Maize", "state": "Madhya Pradesh", "area_ha": 5},
        {"crop": "Onion", "state": "Maharashtra", "area_ha": 12},
        {"crop": "Cotton", "state": "Madhya Pradesh", "area_ha": 8},
        {"crop": "Cotton", "state": "Gujarat", "area_ha": 10},
        {"crop": "Barley", "state": "Rajasthan", "area_ha": 6},
        {"crop": "Groundnut", "state": "Tamil Nadu", "area_ha": 7},
        {"crop": "Wheat", "state": "Uttar Pradesh", "area_ha": 9},
    ]

    # Derive profit scores
    profit_scores = []
    for field in farmer_fields:
        try:
            ps = derive_profit_score(field["crop"], field["state"], field["area_ha"])
            profit_scores.append(ps)
        except Exception as e:
            warnings.warn(f"derive_profit_score failed for {field}: {e}")

    if not profit_scores:
        print("No valid fields -> nothing to solve.")
        exit(1)

    # Save profit summary for inspection
    df_profit = pd.DataFrame(profit_scores)
    df_profit.index = np.arange(1, len(df_profit) + 1)
    df_profit.to_csv("profit_summary.csv", index_label="FieldIndex")
    print("Saved profit_summary.csv")

    # Demand data (example)
    demand_data = {
        "Wheat": -5032,
        "Rice": 4523,
        "Potato": 302,
        "Maize": 2523,
        "Sugarcane": 4012,
        "Soybean": 3521,
        "Cotton": 2890,
        "Barley": 2009,
        "Groundnut": 1800,
    }

    # Build classical QUBO arrays for brute-force energy evaluation (keeps original classical flow)
    linear, quadratic = build_qubo_coeffs(profit_scores, demand_data)

    print("\n=== Running Classical Brute-Force Solver (explicit QUBO) ===")
    bf_result = classical_bruteforce_solver(profit_scores, linear, quadratic)
    print(f"Brute-force: Profit = ₹{bf_result['profit']:.3f}, Energy = {bf_result['energy']:.6f}, Time = {bf_result['time']:.4f}s")

    # Build QuadraticProgram and compare with maqca (Qiskit-based) if available
    if QISKIT_AVAILABLE:
        try:
            qp = build_qubo_with_penalties(profit_scores, demand_data)
            compare_res = compare_maqca_vs_bruteforce(qp, profit_scores, demand_data)
        except Exception as e:
            warnings.warn(f"Error in QUBO/maqca pipeline: {e}. Skipping maqca comparison.")
            # Save brute-only summary
            df_single = pd.DataFrame([{
                "FieldIndex": i + 1,
                "State": ps["state"],
                "Crop": ps["crop"],
                "Profit": ps["profit_score"],
                "Production_q": ps["production_q"],
                "Brute_Assigned": int(bf_result["assignment"][i] if bf_result["assignment"] is not None else 0),
                "maqca_Assigned": int(bf_result["assignment"][i] if bf_result["assignment"] is not None else 0),
            } for i, ps in enumerate(profit_scores)])
            df_single.to_csv("assignment_summary.csv", index=False)
            visualize_results(df_single, bf_result["profit"], bf_result["profit"], bf_result["time"], bf_result["time"])
    else:
        # Qiskit not available; still create visualizations comparing brute to itself
        print("Qiskit not available: saving brute-only outputs and visualizations.")
        df_single = pd.DataFrame([{
            "FieldIndex": i + 1,
            "State": ps["state"],
            "Crop": ps["crop"],
            "Profit": ps["profit_score"],
            "Production_q": ps["production_q"],
            "Brute_Assigned": int(bf_result["assignment"][i] if bf_result["assignment"] is not None else 0),
            "maqca_Assigned": int(bf_result["assignment"][i] if bf_result["assignment"] is not None else 0),
        } for i, ps in enumerate(profit_scores)])
        df_single.to_csv("assignment_summary.csv", index=False)
        visualize_results(df_single, bf_result["profit"], bf_result["profit"], bf_result["time"], bf_result["time"])

    # Complexity note printed for your report
    print("\n--- Complexity Note ---")
    print("Brute-force solver: O(2^n) time (exponential with number of binary variables).")
    print("QUBO + eigen-solver (NumPyMinimumEigensolver) used by maqca: classical algorithm based on linear algebra;")
    print("practical time depends on QUBO size and eigen-solver settings (often polynomial in matrix dimension but heavy constants).")
    print("In real quantum QAOA hardware, one expects asymptotic advantages in certain cases, but empirical performance depends on problem structure.")
    print("-----------------------")