from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Sampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo

# ✅ Create a small example QUBO
qp = QuadraticProgram()
qp.binary_var('x')
qp.binary_var('y')
qp.minimize(linear={'x': -1, 'y': -2}, quadratic={('x', 'y'): 1})

# ✅ Convert to QUBO
qubo = QuadraticProgramToQubo().convert(qp)

# ✅ Use the older Sampler (not SamplerV2)
optimizer = COBYLA(maxiter=100)
sampler = Sampler()   # ⚠️ use this, not SamplerV2

qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1)
solver = MinimumEigenOptimizer(qaoa)

# ✅ Solve the QUBO
result = solver.solve(qubo)

print("\n✅ QAOA run successful!")
print("Result:", result)
print("Best assignment:", result.x)
print("Best objective value:", result.fval)
