"""Run the complete entrance-to-cup-exit episode with live PhysX water.

Exit code 0 requires every task and every strict physics gate to pass.
Failures retain the recording and measured report; this is not a demo shortcut.
"""
import os
import runpy
import sys
from pathlib import Path

tag = sys.argv[1] if len(sys.argv) > 1 else 'live_episode'
os.environ.update(MODE='gpu', EP_ITERS='32', PAD='1', GRIP_ONLY='0', KEEP_GOING='1', FILL_ML='700', SQ='0.5', LOW='1.0', TSQ='0.5',
                  STEPS='teabag,kettle,powder,stir', LEVEL='0')
runner = Path(__file__).with_name('full_episode.py')
sys.argv = [str(runner), tag, 'walk', 'mimic']
runpy.run_path(str(runner), run_name='__main__')
