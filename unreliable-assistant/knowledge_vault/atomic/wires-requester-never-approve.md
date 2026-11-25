---
id: wires-requester-never-approve
title: The requester can never approve their own transfer, and no system or software agent may execute a transfer without a recorded human approval
domain: finance
risk_tier: 3
version: 1
part_of: wires
enforced_by: [schema:SubmitTransactionOut, module:plugins.builtin, module:agents.react, module:app.db]
---
The requester can never approve their own transfer, and no system or software agent may execute a transfer without a recorded human approval.

Part of [[wires]]. See also [[wires-every-wire-transfer]], [[changemgmt-every-production-change]].
