"""Accuracy of the LLM claim judge on adversarial (premise, question, claim, truth) cases, incl. ones that fooled the NLI model.
Usage: uv run python scripts/eval_llm_verifier.py [model ...]   (needs OPENROUTER_API_KEY; ~ $0.01)"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.llm_reasoner import OpenRouterClient  # noqa: E402
from agents.llm_verifier import LLMClaimJudge  # noqa: E402
from app.config import load_env  # noqa: E402

W = ("Every wire transfer above 10,000 USD requires approval from two authorised approvers. Transfers above 250,000 USD additionally "
     "require sign-off from the compliance officer. The requester can never approve their own transfer.")
I = "A Severity 1 incident must be acknowledged within 15 minutes and an incident commander assigned. A written post-incident review is due within 5 business days."
A = "Access rights are reviewed every quarter by the owning manager."
R = "Customers may request a refund within 30 days of purchase. The enterprise licence SKU-099 is non-refundable once activated. Refunds above 5,000 USD need support manager approval."
P = "Passwords must be at least 14 characters long. Scheduled rotation is not required, but a password must be changed immediately if compromise is suspected. Employees must never reuse a work password elsewhere."
X = "Meals are reimbursed up to 75 USD per day while travelling. Expense reports must be submitted within 30 days of the trip. Alcohol is not reimbursable."
CASES = [  # (premise, question, claim, truth)
    (W, "How many approvers does a 12,000 USD wire need?", "A wire transfer of 12,000 USD requires two authorised approvers.", True),
    (W, "How many approvers does a 12,000 USD wire need?", "A wire transfer of 12,000 USD requires one approver.", False),
    (W, "Does an 8,000 USD wire need two approvers?", "Wire transfers of 8,000 USD do not fall under the two-approver requirement.", True),
    (W, "Does an 8,000 USD wire need two approvers?", "An 8,000 USD wire requires two authorised approvers.", False),
    (W, "We are wiring 300,000 USD. Who must approve?", "A wire transfer of 300,000 USD requires two authorised approvers and sign-off from the compliance officer.", True),
    (W, "We are wiring 300,000 USD. Who must approve?", "A wire transfer of 300,000 USD requires no compliance officer sign-off.", False),
    (W, "Can the requester approve their own 20,000 USD transfer?", "The requester cannot approve their own transfer.", True),
    (W, "Can the requester approve their own 20,000 USD transfer?", "The requester may approve their own transfer if a second approver also signs.", False),
    (I, "A severity 1 incident was raised at 09:00 and acknowledged at 09:40. Was the target met?", "The target was not met because acknowledgement took 40 minutes against a 15-minute limit.", True),
    (I, "A severity 1 incident was raised at 09:00 and acknowledged at 09:10. Was the target met?", "The target was not met.", False),
    (I, "Who is assigned in a severity 1 incident?", "An incident commander must be assigned.", True),
    (I, "Who is assigned in a severity 1 incident?", "The CEO must be assigned as incident commander.", False),
    (A, "If a manager reviews access rights only twice a year, is that enough?", "Twice-yearly review is not enough because the policy requires quarterly review.", True),
    (A, "If a manager reviews access rights only twice a year, is that enough?", "Twice-yearly review is enough under the access control policy.", False),
    (A, "Who reviews access rights?", "The owning manager reviews access rights.", True),
    (A, "Who reviews access rights?", "The security team reviews access rights.", False),
    (R, "A customer asks for a refund 20 days after buying hardware. Allowed?", "A refund request made 20 days after purchase is within the 30-day window.", True),
    (R, "A customer asks for a refund 45 days after buying hardware. Allowed?", "A refund request made 45 days after purchase is within the 30-day window.", False),
    (R, "Can the activated SKU-099 licence be refunded?", "SKU-099 cannot be refunded once activated.", True),
    (R, "Can the activated SKU-099 licence be refunded?", "SKU-099 can be refunded within 30 days even after activation.", False),
    (R, "Who approves a 7,000 USD refund?", "A 7,000 USD refund needs support manager approval.", True),
    (R, "Who approves a 7,000 USD refund?", "A 7,000 USD refund needs director approval.", False),
    (P, "May I reuse my work password elsewhere?", "No, work passwords must never be reused elsewhere.", True),
    (P, "May I reuse my work password elsewhere?", "Reuse is acceptable if the other site is a company service.", False),
    (P, "Do I have to change my 90-day-old password?", "Scheduled rotation is not required.", True),
    (P, "Do I have to change my 90-day-old password?", "Passwords must be rotated every 90 days.", False),
    (X, "Can I expense a beer at dinner on a trip?", "Alcohol is not reimbursable.", True),
    (X, "Can I expense a beer at dinner on a trip?", "Alcohol is reimbursable up to 20 USD per day.", False),
    (X, "My trip ended 45 days ago. Is it too late to file expenses?", "Expense reports must be submitted within 30 days, so 45 days is past the deadline.", True),
    (X, "My trip ended 20 days ago. Is it too late to file expenses?", "It is too late to file expenses.", False),
]


async def main(models: list[str]) -> None:
    load_env()
    for model in models:
        reasoning_required = "sonnet" in model
        c = OpenRouterClient(model=model, timeout_s=60, disable_reasoning=not reasoning_required)
        j = LLMClaimJudge(c, timeout_s=60, max_tokens=800 if reasoning_required else 8)
        res = await asyncio.gather(*[j.supported(cl, p, q) for p, q, cl, _ in CASES])
        tp = sum(r and t for r, (_, _, _, t) in zip(res, CASES)); fn = sum((not r) and t for r, (_, _, _, t) in zip(res, CASES))
        fp = sum(r and (not t) for r, (_, _, _, t) in zip(res, CASES)); tn = sum((not r) and (not t) for r, (_, _, _, t) in zip(res, CASES))
        print(f"{model}: true claims kept {tp}/{tp + fn} | FALSE claims wrongly approved {fp}/{fp + tn} | cost ${c.usage.cost_usd:.4f} errors {c.usage.errors}")
        for r, (_, _, cl, t) in zip(res, CASES):
            if r != t:
                print(f"   MISS ({'false-approve' if r else 'false-reject'}): {cl[:90]}")
        await c.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or ["openai/gpt-6-luna", "anthropic/claude-sonnet-5.5"]))
