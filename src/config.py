DEFAULT_DATA_URL = "https://data.seattle.gov/resource/tmmm-ytt6.csv"

DEFAULT_DATE_COL_CANDIDATES = [
    "CheckoutDate",
    "Checkout Date",
    "checkoutdate",
    "Date",
    "date",
]

DEFAULT_YEAR_COL_CANDIDATES = [
    "CheckoutYear",
    "Checkout Year",
    "checkoutyear",
    "Year",
    "year",
]

DEFAULT_MONTH_COL_CANDIDATES = [
    "CheckoutMonth",
    "Checkout Month",
    "checkoutmonth",
    "Month",
    "month",
]

# Prefer category columns that stay meaningful after filtering to books.
DEFAULT_CATEGORY_COL_CANDIDATES = [
    "Collection",
    "collection",
    "UsageClass",
    "usageclass",
    "CheckoutType",
    "checkouttype",
    "Audience",
    "audience",
    "ItemType",
    "itemtype",
    "Format",
    "format",
    "MaterialType",
    "materialtype",
]

DEFAULT_COUNT_COL_CANDIDATES = [
    "Checkouts",
    "checkouts",
    "CheckoutCount",
    "checkoutcount",
    "Count",
    "count",
]

DEFAULT_BOOK_COL_CANDIDATES = [
    "MaterialType",
    "materialtype",
    "Format",
    "format",
    "ItemType",
    "itemtype",
]

SEASONAL_PERIODS = 12
