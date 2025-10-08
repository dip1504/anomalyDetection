import pandas as pd
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

df = pd.read_csv(input_path)
df.to_csv(output_path, index=False)
print(f"Preprocessed file saved to {output_path}")
