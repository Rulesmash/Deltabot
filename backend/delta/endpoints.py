"""
delta/endpoints.py — URL constants for Delta Exchange India.

India Testnet  : https://cdn-ind.testnet.deltaex.org
India Production: https://api.india.delta.exchange

All endpoint paths follow the official Delta Exchange v2 REST API.
Reference: https://docs.delta.exchange/
"""

# ── REST API Paths ─────────────────────────────────────────────────────────────

# Market data
PRODUCTS      = "/v2/products"
TICKERS       = "/v2/tickers"
TICKER        = "/v2/tickers/{symbol}"         # format with symbol
ORDERBOOK     = "/v2/l2orderbook/{symbol}"
TRADES        = "/v2/trades/{symbol}"
CANDLES       = "/v2/history/candles"          # ?symbol=&resolution=&start=&end=

# Account
WALLET        = "/v2/wallet/balances"
POSITIONS     = "/v2/positions"
ORDERS        = "/v2/orders"
ORDER_BY_ID   = "/v2/orders/{order_id}"
CANCEL_ORDER  = "/v2/orders/{order_id}"
CANCEL_ALL    = "/v2/orders"

# Trade history
FILLS         = "/v2/fills"

# ── Resolutions supported by Delta Exchange candle endpoint ────────────────────
# Maps human-friendly strings → API resolution values (minutes for REST)
RESOLUTIONS = {
    "1m":  "1",
    "3m":  "3",
    "5m":  "5",
    "15m": "15",
    "30m": "30",
    "1h":  "60",
    "2h":  "120",
    "4h":  "240",
    "6h":  "360",
    "1d":  "1D",
}

# ── WebSocket channel names ────────────────────────────────────────────────────
WS_CANDLESTICK = "candlestick_{resolution}"   # e.g. candlestick_5
WS_TICKER      = "ticker"
WS_MARK_PRICE  = "mark_price"
WS_ORDERBOOK   = "l2_orderbook"

# Private channels (require auth)
WS_ORDERS      = "orders"
WS_POSITIONS   = "positions"
WS_USER_FILLS  = "user_fills"
WS_BALANCE     = "user_balance"

# ── Order types ────────────────────────────────────────────────────────────────
ORDER_TYPE_LIMIT      = "limit_order"
ORDER_TYPE_MARKET     = "market_order"
ORDER_TYPE_STOP_LIMIT = "stop_order"       # stop-limit (SL)

# ── Order sides ───────────────────────────────────────────────────────────────
SIDE_BUY  = "buy"
SIDE_SELL = "sell"

# ── Time-in-force ─────────────────────────────────────────────────────────────
TIF_GTC = "gtc"   # Good Till Cancel (default)
TIF_FOK = "fok"   # Fill or Kill
TIF_IOC = "ioc"   # Immediate or Cancel
