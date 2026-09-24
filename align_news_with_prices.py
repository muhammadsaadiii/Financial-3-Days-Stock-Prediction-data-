"""
===========================================================
Timestamp Alignment + Future Returns (Labeling Prep)
===========================================================

For every news row (title, date, stock):

  1. Aligns it to the correct REFERENCE trading day
     (same day if published before market close, next
     trading day if after -- avoids lookahead bias)

  2. Finds the trading day FUTURE_WINDOW_DAYS trading days
     later and computes the % return -- this is the raw
     number label thresholds (Crash/Neutral/Spike) will be
     built from

  3. Adds two quality/context columns so labeling isn't a
     blind flat-% cutoff:
       - pre_event_volatility (14-day rolling std of returns,
         so a 6% move on a normally-calm stock isn't treated
         the same as 6% on a stock that swings that much on
         an average day)
       - avg_volume_20d (flags illiquid/penny stocks where
         price data is less reliable)

  4. Flags rows where there isn't enough future price data
     yet (e.g. very recent news, or a ticker delisted soon
     after) so you can exclude them from labeling instead of
     silently mislabeling them

OUTPUT:
    news_with_prices.csv
        title, date, stock, aligned_date, matched_trading_date,
        ref_open, ref_high, ref_low, ref_close, ref_adj_close,
        ref_volume, pre_event_volatility, avg_volume_20d,
        future_date, future_adj_close, return_pct,
        trading_days_available, insufficient_future_data

    unmatched_news.csv
        news rows that couldn't be aligned to any price data
        at all (see the "reason" column)
===========================================================
"""

from pathlib import Path
import logging

import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "gap_fill_project" / "output"

NEWS_FILE = OUTPUT_DIR / "merged_final.csv"

PRICE_FILE = OUTPUT_DIR / "stock_price_data.csv"

ALIGNED_OUTPUT = OUTPUT_DIR / "news_with_prices.csv"

UNMATCHED_OUTPUT = OUTPUT_DIR / "unmatched_news.csv"

# Market close time, US Eastern
MARKET_CLOSE_HOUR = 16  # 4:00 PM

# Max gap allowed when aligning news to a reference trading day.
# Prevents old news being wrongly matched to a trading day years
# later just because a ticker had no data near that date.
MAX_GAP_DAYS = 7

# How many TRADING days ahead to measure the return for labeling.
# Adjust this once you've looked at the return distribution.
FUTURE_WINDOW_DAYS = 1

# Rolling windows for context columns
VOLATILITY_WINDOW = 14
VOLUME_WINDOW = 20


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# =========================================================
# MAIN
# =========================================================

def main():

    logger.info("Loading news dataset...")
    news = pd.read_csv(NEWS_FILE)
    logger.info("News rows: %s", len(news))

    logger.info("Loading price dataset...")
    prices = pd.read_csv(PRICE_FILE)
    logger.info("Price rows: %s", len(prices))

    # -----------------------------------------------------
    # PARSE DATES
    # -----------------------------------------------------

    logger.info("Parsing dates...")

    news["date"] = pd.to_datetime(news["date"], utc=True)
    news["date"] = news["date"].dt.tz_convert("America/New_York")

    prices["date"] = pd.to_datetime(prices["date"], utc=True, errors="coerce")
    prices["date"] = prices["date"].dt.tz_convert("America/New_York")

    prices = prices.dropna(subset=["date"])

    # -----------------------------------------------------
    # DECIDE ALIGNED DATE (same day vs next day)
    # -----------------------------------------------------

    logger.info("Computing aligned date (market-close cutoff rule)...")

    news_hour = news["date"].dt.hour
    news_date_only = news["date"].dt.normalize()

    news["aligned_date"] = news_date_only.where(
        news_hour < MARKET_CLOSE_HOUR,
        news_date_only + pd.Timedelta(days=1),
    )

    # Drop tz for the joins below (already used for the cutoff above)
    news["aligned_date"] = news["aligned_date"].dt.tz_localize(None)
    prices["date"] = prices["date"].dt.tz_localize(None)

    # -----------------------------------------------------
    # PRECOMPUTE PER-TICKER PRICE SERIES COLUMNS
    # -----------------------------------------------------
    #
    # day_index          : sequential trading-day number per ticker
    #                      (0, 1, 2, ...), used to look up "N trading
    #                      days later" without a manual date-math loop
    # daily_return       : day-over-day % change, feeds volatility
    # pre_event_volatility: rolling std of daily_return
    # avg_volume_20d     : rolling mean volume
    # ticker_max_day_index: last available day_index for that ticker,
    #                      used to flag insufficient future data
    # -----------------------------------------------------

    logger.info("Computing per-ticker rolling features (volatility, volume)...")

    prices = prices.sort_values(["stock", "date"]).reset_index(drop=True)

    prices["day_index"] = prices.groupby("stock").cumcount()

    prices["daily_return"] = prices.groupby("stock")["adj_close"].pct_change()

    prices["pre_event_volatility"] = (
        prices.groupby("stock")["daily_return"]
        .transform(lambda s: s.rolling(VOLATILITY_WINDOW, min_periods=5).std())
    )

    prices["avg_volume_20d"] = (
        prices.groupby("stock")["volume"]
        .transform(lambda s: s.rolling(VOLUME_WINDOW, min_periods=5).mean())
    )

    ticker_max_index = (
        prices.groupby("stock")["day_index"].max()
        .rename("ticker_max_day_index")
        .reset_index()
    )

    # A lookup table to fetch price info by (stock, day_index) --
    # used for the future-return lookup further down.
    price_by_index = prices[
        ["stock", "day_index", "date", "adj_close"]
    ].rename(columns={"date": "future_date", "adj_close": "future_adj_close"})

    # -----------------------------------------------------
    # SORT (required for merge_asof)
    # -----------------------------------------------------

    news = news.sort_values("aligned_date")
    prices_sorted = prices.sort_values("date")

    # -----------------------------------------------------
    # ALIGN NEWS -> REFERENCE TRADING DAY (within MAX_GAP_DAYS)
    # -----------------------------------------------------

    logger.info(
        "Aligning news to nearest available trading day per ticker "
        "(max gap: %s days)...",
        MAX_GAP_DAYS,
    )

    merged = pd.merge_asof(
        news,
        prices_sorted,
        left_on="aligned_date",
        right_on="date",
        by="stock",
        direction="forward",
        tolerance=pd.Timedelta(days=MAX_GAP_DAYS),
        suffixes=("", "_price"),
    )

    merged = merged.rename(columns={
        "date_price": "matched_trading_date",
        "open": "ref_open",
        "high": "ref_high",
        "low": "ref_low",
        "close": "ref_close",
        "adj_close": "ref_adj_close",
        "volume": "ref_volume",
    })

    # -----------------------------------------------------
    # SPLIT MATCHED VS UNMATCHED
    # -----------------------------------------------------

    unmatched_mask = merged["ref_close"].isna()

    unmatched = merged[unmatched_mask].copy()
    matched = merged[~unmatched_mask].copy()

    logger.info("Matched: %s | Unmatched: %s", len(matched), len(unmatched))

    tickers_with_any_price = set(prices["stock"].unique())

    unmatched["reason"] = unmatched["stock"].apply(
        lambda s: "no_price_data_for_ticker"
        if s not in tickers_with_any_price
        else "no_trading_day_within_gap_window"
    )

    logger.info("Unmatched breakdown:\n%s", unmatched["reason"].value_counts())

    # -----------------------------------------------------
    # ATTACH day_index + TICKER MAX INDEX TO MATCHED ROWS
    # -----------------------------------------------------

    matched = matched.merge(
        ticker_max_index, on="stock", how="left"
    )

    # -----------------------------------------------------
    # FUTURE RETURN LOOKUP
    # -----------------------------------------------------

    logger.info(
        "Computing %s-trading-day future returns...",
        FUTURE_WINDOW_DAYS,
    )

    matched["future_day_index"] = matched["day_index"] + FUTURE_WINDOW_DAYS

    matched = matched.merge(
        price_by_index,
        left_on=["stock", "future_day_index"],
        right_on=["stock", "day_index"],
        how="left",
        suffixes=("", "_future"),
    )

    matched["return_pct"] = (
        (matched["future_adj_close"] - matched["ref_adj_close"])
        / matched["ref_adj_close"]
        * 100
    )

    matched["trading_days_available"] = (
        (matched["ticker_max_day_index"] - matched["day_index"])
        .clip(upper=FUTURE_WINDOW_DAYS)
    )

    matched["insufficient_future_data"] = (
        matched["trading_days_available"] < FUTURE_WINDOW_DAYS
    )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    final_columns = [
        "title", "date", "stock", "aligned_date", "matched_trading_date",
        "ref_open", "ref_high", "ref_low", "ref_close", "ref_adj_close",
        "ref_volume", "pre_event_volatility", "avg_volume_20d",
        "future_date", "future_adj_close", "return_pct",
        "trading_days_available", "insufficient_future_data",
    ]

    matched[final_columns].to_csv(
        ALIGNED_OUTPUT, index=False, encoding="utf-8-sig"
    )

    unmatched[["title", "date", "stock", "reason"]].to_csv(
        UNMATCHED_OUTPUT, index=False, encoding="utf-8-sig"
    )

    logger.info("=================================================")
    logger.info("ALIGNMENT + FUTURE RETURNS COMPLETE")
    logger.info("=================================================")
    logger.info("Aligned output:    %s", ALIGNED_OUTPUT)
    logger.info("Unmatched output:  %s", UNMATCHED_OUTPUT)
    logger.info("Match rate: %.1f%%", 100 * len(matched) / len(merged))
    logger.info(
        "Rows with full %s-day future data: %s / %s",
        FUTURE_WINDOW_DAYS,
        (~matched["insufficient_future_data"]).sum(),
        len(matched),
    )

    print("\nSAMPLE:")
    print(matched[final_columns].head(5).to_string(index=False))

    print("\nRETURN_PCT DISTRIBUTION (rows with full future data):")
    print(
        matched.loc[~matched["insufficient_future_data"], "return_pct"]
        .describe()
    )


if __name__ == "__main__":
    main()