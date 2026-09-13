"""Rebuild and verify the pilot; --collect also fetches missing supplemental cache."""
import subprocess
import sys
from pathlib import Path
root=Path(__file__).resolve().parent
steps=[]
if '--collect' in sys.argv:steps.append(['collect_dataset_v1.py'])
steps.extend([['clean_dataset_v1.py'],['build_diffusion_pilot.py','--clean'],['validate_dataset_v1.py']])
for step in steps:
    subprocess.run([sys.executable,str(root/step[0]),*step[1:]],cwd=root,check=True)
print('Pilot collection/cleaning checks complete; see DATASET_STATUS.md for research limitations.')
