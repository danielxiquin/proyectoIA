
DATASET_FILE = "bitext_customer_support.csv"

COL_TEXT  = "instruction"

COL_LABEL = "category"



CATEGORIES = [
    {"key": "ORDER",        "display": "Order",            "color": "#00F5FF"},
    {"key": "BILLING",      "display": "Billing",          "color": "#39FF14"},
    {"key": "SHIPPING",     "display": "Shipping",         "color": "#FFE600"},
    {"key": "REFUND",       "display": "Refund",           "color": "#FF2CF0"},
    {"key": "ACCOUNT",      "display": "Account",          "color": "#00F5FF"},
    {"key": "CANCEL",       "display": "Cancellation",     "color": "#FF2CF0"},
    {"key": "CANCELLATION_REQUEST", "display": "Cancellation", "color": "#FF2CF0"},
    {"key": "CONTACT",      "display": "Contact",          "color": "#39FF14"},
    {"key": "DELIVERY",     "display": "Delivery",         "color": "#FFE600"},
    {"key": "FEEDBACK",     "display": "Feedback",         "color": "#FFE600"},
    {"key": "NEWSLETTER_SUBSCRIPTION", "display": "Newsletter", "color": "#00F5FF"},
    {"key": "SUBSCRIPTION", "display": "Subscription",     "color": "#00F5FF"},
    {"key": "PAYMENT",      "display": "Payment",          "color": "#39FF14"},
    {"key": "INVOICE",      "display": "Invoice",          "color": "#39FF14"},
    {"key": "TECHNICAL",    "display": "Technical Support","color": "#FFE600"},
    {"key": "COMPLAINT",    "display": "Complaint",        "color": "#FF2CF0"},
]


CATEGORY_DISPLAY = {c["key"]: c["display"] for c in CATEGORIES}
CATEGORY_COLOR   = {c["key"]: c["color"]   for c in CATEGORIES}
