# G1 in Isaac Sim

The G1 tasks rebuilt on Isaac Sim GPU PhysX, AI-operated, following the recorded MuJoCo motion. Each task stands alone.

| Task | Status | Videos | Data |
|---|---|---|---|
| Take the mug from the rack | Stages A and B done; the carry walk between them, t 14.3 to 18, is not | A: [Film](https://drive.google.com/file/d/1VaGEZVlBIZisGLwsW3JpKMb25Gjyzf7m/view), [Eye](https://drive.google.com/file/d/1V_VJuEcLRqH9jiWurLM2B4wA2eSTuRAQ/view). B: [Film](https://drive.google.com/file/d/1nGdkzt3EZ0dPJ-CaQdBNL-3564acMQqa/view), [Eye](https://drive.google.com/file/d/1PAt7C1ScKndEWHUuYYEU956u5m3-pgPG/view) | [mug/](mug/RESULT.md) |
| Tea bag into the mug | NOT DONE. From t 13 with the bag already in the pinch, the hand carries it over the counter and holds it to t 21, then the bag drops and ends on the counter beside the mug. The pick from the tray, t 0.5 to 13, and the release into the mug were not reached | [Film](https://drive.google.com/file/d/102YMpwusbTiLkDZCLviM5Wcmm7uhEK-O/view), [Eye camera](https://drive.google.com/file/d/1s3x6dK6KGf0roOHPsTO7B6Sk0amsLpdo/view) | [teabag/](teabag/RESULT.md) |
| Kettle pour | Pour done with real particle water from t 31.5, kettle already in the hand: 134.0 ml in the mug. Spill check FAILS: 13.8 ml outside, 10.1 ml of it beside the mug during the pour. The walk to the kettle, the lift and the carry back, t 0.5 to 31.5, are not done: the robot falls at t 4.7 | [Film](https://drive.google.com/file/d/1S9qm4lZOANyV1jyjnX1wyGFMH-DoOVkL/view), [Eye camera](https://drive.google.com/file/d/1i_fevoUGT-zE9wtOGtYTccWNh93r3JnG/view) | [pour/](pour/RESULT.md) |
| Take the cup and walk away | Motion done | [Eye camera](https://drive.google.com/file/d/15c2Xa5_sedsPQMGrjRjJWtHBCWHTa9Xr/view), [Film camera](https://drive.google.com/file/d/138QCs79SRuTiNNlxK7S8a8lEXyq1TgeD/view) | [cup/](cup/RESULT.md) |
| Stir with the spoon | Carry, stir and release done, no liquid in the mug; taking the spoon from the tub, t 0.5 to 15.5, is not | [Film](https://drive.google.com/file/d/1p26IsALfB4gBGNRDCIAOQ6XG_P87rwDA/view), [Eye camera](https://drive.google.com/file/d/1GPeanJIRdgpDnVcuGU1dvvfN3lD2mQtd/view) | [stir/](stir/) |

`tools/` holds the import, live control, take and render scripts, and the per-task live fixes (`fix_<task>.py`).
