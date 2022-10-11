import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/vkraman/Research/agnostic_boosting/AgnosticMulticlassBoosting')

df = pd.read_csv('data/abalone.csv', sep = ',')
print(df)
