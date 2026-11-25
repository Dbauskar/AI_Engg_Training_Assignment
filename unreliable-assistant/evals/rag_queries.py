"""Labeled queries over knowledge_vault/ (expected document id, or None = unanswerable)."""

ANSWERABLE = [
    ("How long do we keep customer records after an account closes?", "retention"),
    ("What is the deadline to notify the regulator after a data breach?", "incident"),
    ("Is SKU-099 refundable once it has been activated?", "refunds"),
    ("How long is the battery warranty for SKU-205?", "warranty"),
    ("Who has to approve a 300,000 USD wire transfer?", "wires"),
    ("When is the holiday change freeze for production deployments?", "changemgmt"),
    ("What is the minimum password length?", "passwords"),
    ("How many days per week must I work from the office?", "remote"),
    ("Which certification must tier 1 vendors hold?", "vendorrisk"),
    ("How many vacation days do I get each year?", "pto"),
    ("How often are user access rights reviewed?", "accessctl"),
    ("What is the daily meal allowance when travelling?", "expenses"),
]

UNANSWERABLE = [
    ("What is the capital of France?", None),
    ("How do I bake sourdough bread?", None),
    ("Who won the 2022 football World Cup?", None),
    ("What is the airspeed velocity of an unladen swallow?", None),
]

ALL = ANSWERABLE + UNANSWERABLE
