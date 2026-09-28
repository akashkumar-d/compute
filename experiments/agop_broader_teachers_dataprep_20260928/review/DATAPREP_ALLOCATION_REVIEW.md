# Confirmed DATA_PREP allocation review

The independent deployment reviewer checked the other workstream's release handoff, pause checkpoint and archive audit. It confirmed the user-authorized pause and preservation of all work, and approved removing the queue dependency after a fresh process/capacity check. Root's live check on the new DATA_PREP machine found 32 CPU cores, no GPU, and none of the old wrapper/coordinator/training PIDs.

The target is the existing scratch-studio-devbox Studio on its newly assigned DATA_PREP machine, not the previously discovered stopped candidate Studio. No other workstream is resumed or killed. The reviewed scientific files, 28 configurations/order, runtime limits and prior AGOP terminal records are unchanged. Only deployment metadata/docs and the isolated pinned environment differ from v2.

Static validation is required again after metadata freeze; remote numerical preflight is required on this machine.
