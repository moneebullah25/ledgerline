"""Canonical metric vocabulary shared by ingestion, extraction and the typed query layer.

R8 (XBRL tag inconsistency): companies report the same concept under different us-gaap
tags. CONCEPTS lists the accepted tags per canonical metric, in preference order. A claim
about a metric whose tags a company never filed resolves to `unverifiable`, never to a
nearby number.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Concept:
    name: str
    unit: str  # unit of the claim-facing value: USD | USD/share | percent
    tags: tuple[str, ...] = ()  # XBRL tags, preference order; empty for derived metrics
    synonyms: tuple[str, ...] = ()  # phrases used in releases and on calls, lowercase
    # Derived metrics: (numerator, denominator) for ratios, or (minuend, subtrahend) for diffs.
    ratio_of: tuple[str, str] | None = None
    diff_of: tuple[str, str] | None = None


CONCEPTS: dict[str, Concept] = {c.name: c for c in [
    Concept("revenue", "USD", (
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:Revenues",
        "us-gaap:SalesRevenueNet",
        "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
    ), ("revenue", "revenues", "net revenue", "net revenues", "net sales", "total net sales",
        "total revenue", "total revenues", "sales")),
    Concept("cost_of_revenue", "USD", (
        "us-gaap:CostOfGoodsAndServicesSold", "us-gaap:CostOfRevenue",
    ), ("cost of revenue", "cost of sales", "cost of goods sold")),
    Concept("gross_profit", "USD", ("us-gaap:GrossProfit",), ("gross profit", "gross margin dollars")),
    Concept("operating_income", "USD", ("us-gaap:OperatingIncomeLoss",),
            ("operating income", "income from operations", "operating profit")),
    Concept("net_income", "USD", ("us-gaap:NetIncomeLoss",), ("net income", "net earnings", "net profit")),
    Concept("eps_diluted", "USD/share", ("us-gaap:EarningsPerShareDiluted",),
            ("diluted earnings per share", "diluted eps", "earnings per diluted share",
             "diluted net income per share", "eps")),
    Concept("rd_expense", "USD", ("us-gaap:ResearchAndDevelopmentExpense",),
            ("research and development", "r&d")),
    Concept("sga_expense", "USD", ("us-gaap:SellingGeneralAndAdministrativeExpense",),
            ("selling, general and administrative", "sg&a")),
    Concept("operating_cash_flow", "USD", ("us-gaap:NetCashProvidedByUsedInOperatingActivities",),
            ("operating cash flow", "cash flow from operations", "cash from operations",
             "net cash provided by operating activities")),
    Concept("capex", "USD", ("us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",),
            ("capital expenditures", "capex", "purchases of property and equipment")),
    Concept("cash", "USD", ("us-gaap:CashAndCashEquivalentsAtCarryingValue",),
            ("cash and cash equivalents",)),
    Concept("gross_margin", "percent", ratio_of=("gross_profit", "revenue"),
            synonyms=("gross margin", "gross margin percentage")),
    Concept("operating_margin", "percent", ratio_of=("operating_income", "revenue"),
            synonyms=("operating margin",)),
    Concept("free_cash_flow", "USD", diff_of=("operating_cash_flow", "capex"),
            synonyms=("free cash flow",)),
]}

# Tag -> canonical name, for the ingest-side audit that every stored tag maps somewhere.
TAG_TO_CONCEPT: dict[str, str] = {t: c.name for c in CONCEPTS.values() for t in c.tags}

# The build universe: 15 large US filers with regular 8-K Item 2.02 earnings releases.
# Banks are excluded on purpose: their revenue concepts do not map to the tags above.
UNIVERSE: dict[str, str] = {
    "AAPL": "0000320193", "MSFT": "0000789019", "GOOGL": "0001652044", "AMZN": "0001018724",
    "META": "0001326801", "NVDA": "0001045810", "INTC": "0000050863", "CSCO": "0000858877",
    "ADBE": "0000796343", "KO": "0000021344", "PEP": "0000077476", "MCD": "0000063908",
    "NKE": "0000320187", "HD": "0000354950", "PG": "0000080424",
}
