import sys

import pandas as pd


df = pd.read_csv(sys.argv[1])
df.to_csv(sys.argv[1], sep='\t', index=True)
