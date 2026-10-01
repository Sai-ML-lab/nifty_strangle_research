from pathlib import Path
import pandas as pd
from src.smoke_test import synthetic_chain

out = Path('data/sample_chain.csv')
out.parent.mkdir(exist_ok=True)
df = synthetic_chain()
df.to_csv(out, index=False)
print(f'wrote {len(df):,} rows to {out}')
