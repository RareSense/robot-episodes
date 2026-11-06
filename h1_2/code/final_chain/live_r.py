"""Live teleop from a resume point. usage: live_r.py <tag> <source run> <step or sim time>. AI-operated: every command is kept in live\<tag>\cmd_*.py and out_*.txt."""
import runpy
import sys
sys.path.insert(0, r"D:\tea_robot\isaac_delivery")
import live_tap  # noqa: F401
p = r"D:\tea_robot\isaac_delivery\resume_episode.py"
sys.argv = [p] + sys.argv[1:]
runpy.run_path(p, run_name="__main__")
