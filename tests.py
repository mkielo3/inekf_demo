import pandas as pd

def run_tests():
	df = pd.read_csv("data/output/output.csv")
	reference_df = pd.read_csv("data/reference/state_log.csv")
	for c in ['pos', 'rot', 'vel']:
		for obj in df['object'].unique():
			if obj == 'robot' or c == 'pos':
				cols = [x for x in df.columns if c in x]
				print (c, obj, ((df[df['object'] == obj][cols] - reference_df[reference_df['object'] == obj][cols]).abs().mean().mean()))
				assert (c and ((df[df['object'] == obj][cols] - reference_df[reference_df['object'] == obj][cols]).abs().mean().mean()) < 0.0001)

	print ("All tests pass!")
