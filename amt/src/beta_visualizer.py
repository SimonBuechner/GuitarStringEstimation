import math
import numpy as np


beta = 1.4e-4

fundamental = 100 #Hz
n_partials = 20

for i in range(1, n_partials):
    predicted_partial = (i+1) * fundamental  # Erwartete Partialfrequenz
    print(f"Predicted partial{i}: {predicted_partial}")

    maxBeta_partial = (i+1)*fundamental*math.sqrt(1+ beta * (i+1)**2)
    print(f"Max beta partial{i}: {maxBeta_partial}")