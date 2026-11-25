"""Deterministic, LLM-free reasoners so every route works end to end without a model (and as the FMEA #4 fallback).

GreetingReasoner          DIRECT_ANSWER: a fixed, honest greeting (it never invents policy facts).
RuleBasedExecutionReasoner TRANSACTION_EXEC: extracts account/amount/currency from the (already PII-redacted) request with
                          regexes and prepares a submit_transaction call, or asks for what is missing. It never executes
                          anything: the harness/loop freeze it at PENDING_APPROVAL.
"""
from __future__ import annotations

import re

from agents.react import Act, Final, Message

GREETING = ("Hello! I can answer questions about our policies and products, and prepare requests that a person must approve. "
            "What would you like to know?")
_ACC = re.compile(r"\bACC-\d{8}\b", re.I)
_MONEY = re.compile(r"(\d[\d,]*(?:\.\d{1,2})?)\s*(USD|EUR|GBP)\b", re.I)


class GreetingReasoner:
    async def next_step(self, prompt: list[Message]) -> Act | Final:
        return Final(GREETING)


class RuleBasedExecutionReasoner:
    def __init__(self, query: str) -> None:
        self.query = query

    async def next_step(self, prompt: list[Message]) -> Act | Final:
        q = self.query
        acc, money = _ACC.search(q), _MONEY.search(q)
        low = q.lower()
        if re.search(r"\b(status|suspend|close|cancel|reactivate)\b", low):
            action = "STATUS_CHANGE"
        elif re.search(r"\b(order|purchase|buy)\b", low):
            action = "ORDER"
        else:
            action = "TRANSFER"
        if acc is None:
            return Final("I can prepare that for approval, but I need the account id (it looks like ACC-12345678).")
        if action == "TRANSFER" and money is None:
            return Final("I can prepare that transfer for approval, but I need the amount and currency (for example 1250.50 USD).")
        args: dict = {"action": action, "account_id": acc.group().upper(),
                      "justification": f"Requested via assistant: {q.strip()}"[:500]}
        if money:
            args["amount"], args["currency"] = money.group(1).replace(",", ""), money.group(2).upper()
        return Act("submit_transaction", args, "prepare the request for human approval")
