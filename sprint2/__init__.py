"""Sprint 2: two stock strategies + Bayesian strategy/cash allocation.

Shared:   contracts (I/O spec), data (market input), portfolio (selection/weights/calendar/costs),
          ledger (quantity ledger, execution, daily loops)
Part 1·2: part1_momentum, part2_lowvol  -- signal functions only
Part 3:   models in separate files -- part3_b0_basic (B0), part3_b1_spread + part3_b1_level (B1),
          part3_b2_regime (B2), part3_b2_1_shared_mean (B2.1), part3_b2_2_joint (B2.2); part3_allocation (QP, HOLD/TRANSFER rule, 50:50 benchmark)
Part 4:   part4_evaluation (metrics, tables, figures)
Runner:   run (CLI: python -m sprint2 ...)
"""
