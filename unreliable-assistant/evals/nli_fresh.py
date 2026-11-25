"""Fresh claim set written AFTER the verification guards were designed (the first set in nli_set.py was seen while
designing them, so it is no longer held-out). Evaluate once and report. (note_id, claim, supported)"""

CLAIMS = [
    ("retention", "Customer records are retained for 7 years once an account is closed.", True),
    ("incident", "The post-incident written review is due within 5 business days.", True),
    ("wires", "Transfers above 250,000 USD also require the compliance officer's sign-off.", True),
    ("refunds", "Refunds are returned to the original payment method within 10 business days.", True),
    ("pto", "Sick leave is not capped.", True),
    ("remote", "Working abroad for more than 30 days in a year needs approval from HR and Tax.", True),
    ("vendorrisk", "Tier 1 vendors must hold a current SOC 2 Type II report.", True),
    ("accessctl", "Multi-factor authentication is mandatory for every employee and contractor account.", True),
    ("pto", "Employees may carry over all of their unused leave.", False),
    ("refunds", "Refunds are always approved automatically.", False),
    ("wires", "Every wire needs the CEO's signature.", False),
    ("passwords", "Passwords never expire and never need to be changed.", False),
    ("incident", "The regulator must be notified within 7 days.", False),
    ("vendorrisk", "Tier 1 vendors must be assessed every two years.", False),
    ("accessctl", "Privileged sessions last 24 hours.", False),
    ("expenses", "Business class flights are allowed for trips over 3 hours.", False),
    ("warranty", "Liquid damage is covered by the warranty.", False),
    ("changemgmt", "Emergency changes never need a retrospective review.", False),
]
