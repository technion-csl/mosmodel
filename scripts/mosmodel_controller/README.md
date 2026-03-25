Stage 4: synchronized shared windows on top of the working SID-based controller.

What this adds:
- keeps the stage-3 per-side instruction boundary tracking
- adds `--sync-interval-windows`
- whichever side reaches `I_start` first is STOPped
- once both sides reached `I_start`, both sides are CONTinued together
- once either side reaches `I_end` after synchronization starts, both sides are terminated

This keeps launch and SID-based cleanup unchanged.
