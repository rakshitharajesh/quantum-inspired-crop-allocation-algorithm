"""
compare_and_visualize.py
-------------------------
Runs MAQCA (QAOA / classical QUBO solver) and brute-force baselines 
for varying numbers of farmer inputs, collects runtime metrics,
and visualizes execution time comparison.

Requires:
    pip install qiskit qiskit-optimization qiskit-aer matplotlib seaborn pandas numpy
"""

import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer


# -------------------------------
# STEP 1: Dummy Profit Generator
# -------------------------------
def generate_profit_scores(farmer_fields):
    """Generates mock profit and production data."""
    np.random.seed(len(farmer_fields))
    profit_scores = []
    for field in farmer_fields:
        profit_score = np.random.randint(3000, 7000)
        production_q = np.random.uniform(5, 15)
        profit_scores.append({
            "crop": field["crop"],
            "state": field["state"],
            "production_q": production_q,
            "profit_score": profit_score
        })
    return profit_scores


# -------------------------------
# STEP 2: Build QUBO
# -------------------------------
def build_qubo_with_penalties(profit_scores, demand_data, A=2000, B=1000, profit_threshold=1000):
    qp = QuadraticProgram()
    for i, ps in enumerate(profit_scores):
        qp.binary_var(f"x_{i}_{ps['crop']}")

    linear = {}
    quadratic = {}

    for i, ps in enumerate(profit_scores):
        var = f"x_{i}_{ps['crop']}"
        profit = ps["profit_score"]
        if profit < profit_threshold:
            linear[var] = 1000
        else:
            linear[var] = -profit

    crops = list({ps["crop"] for ps in profit_scores})
    for crop in crops:
        D_c = demand_data.get(crop, 0)
        crop_vars = [(i, f"x_{i}_{ps['crop']}") for i, ps in enumerate(profit_scores) if ps["crop"] == crop]
        for (i1, v1) in crop_vars:
            w1 = profit_scores[i1]["production_q"]
            linear[v1] = linear.get(v1, 0) + B * (w1**2 - 2 * D_c * w1)
            for (i2, v2) in crop_vars:
                if v1 < v2:
                    w2 = profit_scores[i2]["production_q"]
                    quadratic[(v1, v2)] = quadratic.get((v1, v2), 0) + 2 * B * w1 * w2

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp


# -------------------------------
# STEP 3: Solvers
# -------------------------------
def brute_force_solver(profit_scores, demand_data, A=2000, B=1000, profit_threshold=1000):
    """Simple brute-force approach (checks all combinations)."""
    n = len(profit_scores)
    best_profit = -float("inf")
    best_config = None

    for combo in itertools.product([0, 1], repeat=n):
        total_profit, demand_penalty, field_penalty = 0, 0, 0

        for i, assign in enumerate(combo):
            if assign == 1:
                profit = profit_scores[i]["profit_score"]
                total_profit += profit if profit >= profit_threshold else -1000

        unique_crops = list({ps["crop"] for ps in profit_scores})
        for crop in unique_crops:
            D_c = demand_data.get(crop, 0)
            total_prod = sum(ps["production_q"] for ps, a in zip(profit_scores, combo) if a == 1 and ps["crop"] == crop)
            demand_penalty += B * (total_prod - D_c)**2

        total_crops = sum(combo)
        if total_crops > 1:
            field_penalty += A * (total_crops - 1)

        score = total_profit - demand_penalty - field_penalty
        if score > best_profit:
            best_profit, best_config = score, combo

    return np.array(best_config), abs(best_profit)


def maqca_solver(qp):
    """MAQCA (Qiskit classical QUBO solver)."""
    conv = QuadraticProgramToQubo()
    qubo = conv.convert(qp)
    solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    result = solver.solve(qubo)
    return np.array(result.x, dtype=int), abs(result.fval)


# -------------------------------
# STEP 4: Experiment Runner
# -------------------------------
def run_experiments():
    demand_data = {
        "Wheat": 5000, "Rice": 4500, "Potato": 300, "Maize": 2500, "Cotton": 2900,
        "Barley": 2000, "Groundnut": 1800, "Onion": 4000, "Lemon": 2200
    }

    base_fields = [
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

    results = []

    for size in [4, 6, 8, 10, 12]:
        farmer_fields = base_fields[:size]
        profit_scores = generate_profit_scores(farmer_fields)
        qp = build_qubo_with_penalties(profit_scores, demand_data)

        start_b = time.time()
        brute_force_solver(profit_scores, demand_data)
        brute_time = time.time() - start_b

        start_q = time.time()
        maqca_solver(qp)
        maqca_time = time.time() - start_q

        results.append({
            "num_fields": size,
            "brute_time": brute_time,
            "maqca_time": maqca_time
        })
        print(f"✅ Completed for {size} fields")

    return pd.DataFrame(results)


# -------------------------------
# STEP 5: Visualization (Runtime Only)
# -------------------------------
def visualize_results(df):
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    
    plt.figure(figsize=(10, 6))
    plt.bar(df["num_fields"] - 0.2, df["brute_time"], width=0.4, label="Brute-force Time (s)")
    plt.bar(df["num_fields"] + 0.2, df["maqca_time"], width=0.4, label="MAQCA Time (s)")
    plt.title("⏱ Runtime Comparison vs Number of Fields")
    plt.xlabel("Number of Fields")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n📈 Summary of Runtimes:")
    print(df)


def visualize_profit_comparison(brute_profit, maqca_profit):
    """
    Visualize profit comparison between Brute Force and MAQCA.
    """
    
    # Normalize to lakhs (₹)
    brute_lakh = brute_profit / 1e5
    maqca_lakh = maqca_profit / 1e5
    diff_lakh = abs(brute_lakh - maqca_lakh)

    data = {
        "Method": ["Brute Force", "MAQCA", "Difference"],
        "Profit (₹ Lakhs)": [brute_lakh, maqca_lakh, diff_lakh]
    }

    sns.set(style="whitegrid", font_scale=1.2)

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x=data["Method"],
        y=data["Profit (₹ Lakhs)"],
    )

    plt.title("Profit Comparison: Brute Force vs MAQCA")
    plt.ylabel("Profit (₹ Lakhs)")
    plt.tight_layout()

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.2f}",
                    (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=11)

    plt.show()

    print("\n📊 Profit Comparison (Lakhs)")
    print("---------------------------------")
    print(f"Brute Force Profit: ₹{brute_lakh:.2f} Lakhs")
    print(f"MAQCA Profit:       ₹{maqca_lakh:.2f} Lakhs")
    print(f"Difference:         ₹{diff_lakh:.2f} Lakhs")

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    df_results = run_experiments()
    visualize_results(df_results)
    
