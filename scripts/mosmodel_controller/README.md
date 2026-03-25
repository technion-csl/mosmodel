Stage 4: synchronized shared windows on top of the working SID-based controller.

What this adds:
- keeps the stage-3 per-side instruction boundary tracking
- adds `--sync-interval-windows`
- whichever side reaches `I_start` first is STOPped
- once both sides reached `I_start`, both sides are CONTinued together
- once either side reaches `I_end` after synchronization starts, both sides are terminated

This keeps launch and SID-based cleanup unchanged.


External resume gate (new):
- optional CLI flag: --external-resume-gate-dir <dir>
- normal behavior is unchanged when the flag is absent
- when present in sync interval mode, once both sides reach the configured start thresholds the controller STOPs both benchmark PGIDs, writes READY and STATE.json under <dir>, and waits until <dir>/RESUME appears before continuing the synchronized startup
- this is intended for an outer scheduler that wants to quiesce other pairs before releasing the sampled pair
- if you want this pause to happen at an earlier arm threshold, pass that earlier threshold as the controller's I_START value from the outer scheduler


Socket-based external resume gate:
- --external-resume-socket-path <path> and --external-resume-token <token> enable a Unix domain socket handshake with the outer scheduler
- when present in sync interval mode, once both sides reach the configured start thresholds the controller STOPs both benchmark PGIDs, sends a READY JSON payload on the socket, and waits for a RESUME line on the same connection before continuing
- environment fallbacks are supported via MOSMODEL_CONTROLLER_EXTERNAL_RESUME_SOCKET_PATH and MOSMODEL_CONTROLLER_EXTERNAL_RESUME_TOKEN
