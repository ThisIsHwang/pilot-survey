# Stage 2: Search-R1 specialist baselines

Do not start this stage until the zero-shot pilot shows a meaningful backend spread and oracle gap.

1. Enter the pilot venv and convert data:
   `python searchr1_stage2/make_hotpot_searchr1_data.py --work-dir work`
2. Follow the official Search-R1 README to create its native environment at the pinned commit.
3. Keep one retrieval server running and launch:
   `BACKEND=bm25 PORT=8001 bash searchr1_stage2/run_single_stack_grpo.sh`
4. Repeat sequentially for E5 and ColBERT.

This produces the specialist upper bounds needed before implementing mixed-stack or robust-group RL.
