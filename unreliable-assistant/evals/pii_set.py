"""Labelled PII examples. Each item is (text, secrets): every string in `secrets` must be gone after redaction.
`CLEAN` texts contain no PII and must come back unchanged (false-redaction check).
DEV was used while building the detector; TEST was written afterwards and is evaluated ONCE for the report.
Note: the SSNs below are deliberately NOT famous invalid ones (e.g. 078-05-1120), which Presidio rejects by design."""

DEV = [
    ("Contact me at john.smith@example.com about the refund", ["john.smith@example.com"]),
    ("my email is jane_doe[at]example.org", ["jane_doe[at]example.org"]),
    ("reach me: bob (at) mail (dot) example (dot) com thanks", ["bob (at) mail (dot) example (dot) com"]),
    ("write to alice at example dot com please", ["alice at example dot com"]),
    ("it is a l i c e @ e x a m p l e . c o m", ["a l i c e @ e x a m p l e . c o m"]),
    ("ｊｏｈｎ＠ｅｘａｍｐｌｅ．ｃｏｍ is my mail", ["ｊｏｈｎ＠ｅｘａｍｐｌｅ．ｃｏｍ"]),
    ("pаul@example.com is mine", ["pаul@example.com"]),  # Cyrillic 'а'
    ("carl​@exam​ple.com works", ["carl​@exam​ple.com"]),
    ("dave [at] example [dot] net", ["dave [at] example [dot] net"]),
    ("erin @ example . com", ["erin @ example . com"]),
    ("call +1 415-555-0132 tomorrow", ["415-555-0132"]),
    ("my number is (415) 555 0132", ["(415) 555 0132"]),
    ("phone 415.555.0132 after five", ["415.555.0132"]),
    ("dial +44 20 7946 0958", ["20 7946 0958"]),
    ("number 4 1 5 5 5 5 0 1 3 2 thanks", ["4 1 5 5 5 5 0 1 3 2"]),
    ("my ssn is 219-09-9999", ["219-09-9999"]),
    ("SSN 219 09 9999 on file", ["219 09 9999"]),
    ("social security number 219099999 please", ["219099999"]),
    ("card 4111 1111 1111 1111 expires soon", ["4111 1111 1111 1111"]),
    ("pay with 4111-1111-1111-1111", ["4111-1111-1111-1111"]),
    ("amex 5500 0000 0000 0004 please", ["5500 0000 0000 0004"]),
    ("My name is Priya Raghavan and I need a refund", ["Priya Raghavan"]),
    ("Please contact Carlos Mendoza in accounting", ["Carlos Mendoza"]),
    ("request came from 192.168.4.17 yesterday", ["192.168.4.17"]),
    ("IBAN DE89 3704 0044 0532 0130 00 for the refund", ["DE89 3704 0044 0532 0130 00"]),
]

TEST = [
    ("Send the invoice to finance.team@bigcorp.co.uk today", ["finance.team@bigcorp.co.uk"]),
    ("my address is tom.h[at]gmail[dot]com", ["tom.h[at]gmail[dot]com"]),
    ("ping me → lisa{at}example{dot}io when ready", ["lisa{at}example{dot}io"]),
    ("Email: m a r k @ s t a r t u p . i o", ["m a r k @ s t a r t u p . i o"]),
    ("sam＠example．com is the contact", ["sam＠example．com"]),
    ("nina at proton dot me is her inbox", ["nina at proton dot me"]),
    ("ivan@exаmple.com owns the account", ["ivan@exаmple.com"]),  # Cyrillic 'а' in the domain
    ("mail <pat@example.org> for details", ["pat@example.org"]),
    ("Reach her on +49 30 901820 or by post", ["30 901820"]),
    ("tel: 800-555-0199 any time", ["800-555-0199"]),
    ("office (212)555-0147 ext 4", ["(212)555-0147"]),
    ("SSN: 536-90-4399 was entered", ["536-90-4399"]),
    ("536 90 4399 is the number", ["536 90 4399"]),
    ("Visa 4012888888881881 is declined", ["4012888888881881"]),
    ("Amex 3782 822463 10005 expired", ["3782 822463 10005"]),
    ("Dr. Elena Petrova approved it yesterday", ["Elena Petrova"]),
    ("Ask Marcus Webb to sign off on this", ["Marcus Webb"]),
    ("server 10.24.8.113 is down again", ["10.24.8.113"]),
    ("passport holder Sofia Lindqvist, email sofia.l[at]example.se", ["Sofia Lindqvist", "sofia.l[at]example.se"]),
    ("call me on 0044 7911 123456 or mail t.okafor at example dot ng", ["7911 123456", "t.okafor at example dot ng"]),
]

CLEAN = [
    "How long are customer records kept after an account is closed?",
    "Is SKU-099 refundable once it has been activated?",
    "Transfer 1250.50 USD from ACC-12345678 to ACC-87654321",
    "Wires above 10,000 USD need two approvers",
    "The freeze runs from 15 December to 3 January",
    "Call the helpdesk on extension 4471",
    "Order 4471 was shipped on 2026-11-18",
    "What is 250,000 divided by 12?",
    "Who has to approve a 300,000 USD wire transfer?",
    "How many vacation days do I get each year?",
    "What is the minimum password length?",
    "Does a refund above 5,000 USD need manager approval?",
    "Tier 1 vendors must hold a SOC 2 Type II report",
    "Please look at the example dot points in the policy",
    "The meeting is at 5 and the review is at 7",
    "Wire Transfer Approval Policy",
]
