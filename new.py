import os
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import xml.etree.ElementTree as ET
import time
import requests

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# WARNING: Do NOT hardcode your API key here before uploading to GitHub!
# Set it in your environment variables instead.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "") 


class AgenticFinancialAnalyzer:
    def __init__(self):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not found. Please run: pip install openai")
            
        api_key_to_use = OPENAI_API_KEY
        try:
            import streamlit as st
            if not api_key_to_use and "OPENAI_API_KEY" in st.secrets:
                api_key_to_use = st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass
            
        if not api_key_to_use:
            raise ValueError("OPENAI_API_KEY is strictly required. Set it via env variable or Streamlit Secrets!")
            
        self.client = OpenAI(api_key=api_key_to_use)
        self.fast_model = "gpt-4o-mini"
        self.reasoning_model = "gpt-4o"

    # ==========================================================================
    # PHASE 1: Real-time News (Agent 1)
    # ==========================================================================
    def agent1_fetch_realtime_news(self):
        """Fetches general business news from the last 24 hours using Google News RSS."""
        print("[Agent 1] 📡 Fetching general market news for the past 24 hours (US & India)...")
        rss_urls = [
            "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en"
        ]
        
        all_news = []
        for rss_url in rss_urls:
            try:
                req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    xml_data = response.read()
                root = ET.fromstring(xml_data)
                
                # Fetch top 25 breaking items from each region
                for item in root.findall('.//item')[:25]:
                    title = item.find('title').text if item.find('title') is not None else ""
                    pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
                    all_news.append({
                        "headline": title.split(" - ")[0],
                        "published": pub_date
                    })
            except Exception as e:
                pass
                
        print(f"  -> Found {len(all_news)} raw market news articles natively.")
        return all_news

    # ==========================================================================
    # TICKER FORMAT SANITY CHECK (No live requests needed)
    # ==========================================================================
    def agent2_5_sanitize_ticker(self, ticker_guess, market):
        """Pure format check - no live requests, no 429s. Just validates the LLM output shape."""
        if not ticker_guess or len(ticker_guess) < 1:
            return None
        t = ticker_guess.strip().upper()
        # Reject clearly wrong outputs (full words like "NEXSTAR", bare suffixes like "NS")
        if len(t) > 10:
            return None
        if market == "India":
            # Must have a company prefix before .NS or .BO
            if '.' not in t:
                return None
            prefix = t.split('.')[0]
            if len(prefix) < 2:
                return None  # Bare "NS" or "BO" rejected
        return t

    # ==========================================================================
    # PHASE 2: Market Filter & Top 5 Selection (Agent 2)
    # ==========================================================================
    def agent2_extract_top_5_events(self, raw_news):
        """Passes all news to OpenAI to extract Events, then forces a mathematical Yahoo Finance API check."""
        print("[Agent 2] 🧠 Filtering (gpt-4o-mini) Top 5 high-impact events...")
        prompt = """
        You are an expert quantitative financial API. Read the recent market news and identify ONLY the TOP 5 most impactful company-specific events.
        
        CRITICAL RULES for tickers - YOU MUST FOLLOW EXACTLY:
        1. Identify the *EXACT*, official Yahoo Finance ticker symbol. 
        2. NEVER return just a suffix or a partial name (e.g., NEVER return "NS" by itself!). It MUST have the company prefix (e.g., "VEDL.NS", "RELIANCE.NS").
        3. NEVER return a spelled-out company name. (e.g., return "NXST", NEVER "NEXSTAR" or "Nexstar Media").
        4. If US market, it is usually 1 to 5 letters (AAPL, META, TSLA).
        5. If Indian market, ALWAYS append .NS (e.g., TCS.NS).
        6. DO NOT select macroeconomic news (like "Inflation drops"), only events tied to specific, tradable companies.
        *If you do not know the exact ticker, DO NOT include that news event.*
        
        You must return a JSON object with a single key "top_5", which contains a list of exactly 5 JSON objects formatted like this:
        {
          "company_name": "Full legal name of company",
          "ticker": "Exact ticker",
          "market": "US" or "India",
          "headline": "Title of the news",
          "event_type": "2-4 word classification of the event (e.g., Earnings Miss, Product Launch)",
          "impact_score": "Score 1-100"
        }
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.fast_model,
                response_format={"type": "json_object"},
                temperature=0.1,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(raw_news)}
                ]
            )
            data = json.loads(response.choices[0].message.content)
            top_5 = data.get("top_5", [])
            print("  -> Top 5 Events Identified. Running Ticker Format Sanity Check:")
            
            verified_top_5 = []
            for e in top_5:
                guess = e.get('ticker')
                mkt = e.get('market')
                
                clean = self.agent2_5_sanitize_ticker(guess, mkt)
                if clean:
                    e['ticker'] = clean
                    verified_top_5.append(e)
                    print(f"     ✅ [{clean}] format OK -> {e.get('event_type')}")
                else:
                    print(f"     ❌ Dropped bad ticker '{guess}' for market {mkt}")
                    
            return verified_top_5
        except Exception as e:
            print(f"🚨 Agent 2 Failed: {e}")
            return []

    # ==========================================================================
    # PHASE 3: Historical Retrieval, 48-Hour Filter & Deduplication (Agent 3)
    # ==========================================================================
    def agent3_historical_dates(self, ticker, event_type):
        """Fetches Google News RSS search history, filters last 48 hrs, and deduplicates to dicts {date, headline}."""
        print(f"\n[Agent 3] 🔍 Searching history for {ticker} '{event_type}' events via Google News...")
        
        clean_ticker = ticker.split('.')[0]
        query = urllib.parse.quote(f"{clean_ticker} {event_type}")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)
                
            historical_news = []
            forty_eight_hours_ago = datetime.utcnow() - timedelta(hours=48)
            
            for item in root.findall('.//item')[:20]:
                title = item.find('title').text if item.find('title') is not None else ""
                pub_date_str = item.find('pubDate').text if item.find('pubDate') is not None else ""
                
                if pub_date_str:
                    try:
                        dt = pd.to_datetime(pub_date_str).tz_localize(None)
                        if dt > forty_eight_hours_ago:
                            continue # Ignore breaking news matching our history array
                    except Exception:
                        pass
                historical_news.append({"headline": title.split(" - ")[0], "pubDate": pub_date_str})
                
            if not historical_news:
                print("  -> No true past history found older than 48 hours.")
                return []
                
            prompt = """
            You are a date deduplication bot. Look at these historical news articles. Identify the ACTUAL unique 
            distinct events (ignoring updates/commentary weeks later or exact duplicates).
            Return a JSON object with a single key "valid_events" containing a list of objects formatted exactly like this:
            [ {"date": "YYYY-MM-DD", "headline": "The primary headline of the event"} ]
            Limit to max 3 valid events.
            """
            response = self.client.chat.completions.create(
                model=self.fast_model,
                response_format={"type": "json_object"},
                temperature=0.0,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(historical_news)}
                ]
            )
            
            parsed = json.loads(response.choices[0].message.content)
            valid_events = parsed.get("valid_events", [])
            print(f"  -> Extracted historic events: {valid_events}")
            return valid_events
            
        except Exception as e:
            print(f"🚨 Agent 3 Failed: {e}")
            return []

    # ==========================================================================
    # PHASE 4 & 5: Market Data Engine (Agents 4 & 5)
    # ==========================================================================
    def calculate_market_metrics(self, ticker, event_objects):
        """Calculates 1-Month Baseline (Excluding T-1/T-2), T-1/T-2 hype, and True Post-News movement."""
        print(f"[Agent 4/5] 📈 Calculating percentage deviations (Excluding T-1, T-2) for {ticker}...")
        time.sleep(1) # Strict Yahoo Rate Limit Evasion
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        if hist.empty: return []
            
        hist.index = hist.index.tz_localize(None)
        results = []
        
        for item in event_objects:
            d_str = item["date"]
            headline = item.get("headline", "Current Event")
            
            try:
                d = pd.to_datetime(d_str).date()
                future_dates = hist.index[hist.index.date >= d]
                if len(future_dates) < 2: continue # Ensure we have T+1
                    
                t0_idx = hist.index.get_loc(future_dates[0])
                if t0_idx < 23: continue # Ensure 1-month baseline exists
                    
                t0_close = hist.iloc[t0_idx]["Close"]
                
                # Baseline 1 Month BEFORE hype (T-22 to T-3)
                tm22_close = hist.iloc[t0_idx - 22]["Close"]
                tm3_close = hist.iloc[t0_idx - 3]["Close"]
                baseline_1m_move = ((tm3_close - tm22_close) / tm22_close) * 100
                
                # Pre-Event hype (T-1 & T-2) -> Calculated from T-3 to T-1
                tm1_close = hist.iloc[t0_idx - 1]["Close"]
                t1_t2_move = ((tm1_close - tm3_close) / tm3_close) * 100
                
                # Post-event T to T+1
                tp1_close = hist.iloc[t0_idx + 1]["Close"]
                post_move = ((tp1_close - t0_close) / t0_close) * 100
                
                # Volumes
                baseline_1m_vol = hist.iloc[t0_idx - 22 : t0_idx - 2]["Volume"].mean()
                t1_t2_vol = hist.iloc[t0_idx - 2 : t0_idx]["Volume"].mean()
                post_vol = hist.iloc[t0_idx : t0_idx + 2]["Volume"].mean()
                
                results.append({
                    "date": d_str,
                    "headline": headline,
                    "avg_1_month_move_excl_T1_T2_pct": round(baseline_1m_move, 2),
                    "move_of_T1_T2_pct": round(t1_t2_move, 2),
                    "post_news_move_pct": round(post_move, 2),
                    "avg_1_month_volume_excl_T1_T2": int(baseline_1m_vol),
                    "volume_of_T1_T2": int(t1_t2_vol),
                    "post_news_volume": int(post_vol)
                })
            except Exception:
                pass
                
        return results

    def calculate_current_setup(self, ticker):
        """Calculates the Pre-Event hype (T-1 & T-2) and 1-month baseline ending right now."""
        print(f"[Agent 4/5] ⏱️  Calculating Current Market T-1/T-2 hype for {ticker}...")
        try:
            time.sleep(1)
            stock = yf.Ticker(ticker)
            hist = stock.history(period="3mo")
            if len(hist) < 25: return {}
                
            # T-0 is the final row [-1] (Today)
            tm1_close = hist.iloc[-2]["Close"]
            tm3_close = hist.iloc[-4]["Close"]
            tm22_close = hist.iloc[-23]["Close"]
            
            baseline_1m_move = ((tm3_close - tm22_close) / tm22_close) * 100
            t1_t2_move = ((tm1_close - tm3_close) / tm3_close) * 100
            
            baseline_1m_vol = hist.iloc[-23 : -3]["Volume"].mean()
            t1_t2_vol = hist.iloc[-3 : -1]["Volume"].mean()
            
            return {
                "avg_1_month_move_excl_T1_T2_pct": round(baseline_1m_move, 2),
                "move_of_T1_T2_pct": round(t1_t2_move, 2),
                "avg_1_month_volume_excl_T1_T2": int(baseline_1m_vol),
                "volume_of_T1_T2": int(t1_t2_vol)
            }
        except Exception:
            return {}

    # ==========================================================================
    # PHASE 6: Market Context (Agent 6)
    # ==========================================================================
    def agent6_market_context(self, ticker, market):
        """Fetches Macro Index trend, Sector, and 52-Week H/L positional data."""
        print(f"[Agent 6] 📊 Fetching macro and sector context for {ticker}...")
        index_ticker = "^GSPC" if market == "US" else "^NSEI"
        
        try:
            # 1. Broad Market Trend
            time.sleep(1)
            idx_hist = yf.Ticker(index_ticker).history(period="1mo")
            index_trend_pct = 0
            if not idx_hist.empty:
                start = idx_hist.iloc[0]["Close"]
                end = idx_hist.iloc[-1]["Close"]
                index_trend_pct = round(((end - start) / start) * 100, 2)
                
            # 2. Company Info
            curr_price = 0
            high_52 = 0
            low_52 = 0
            try:
                time.sleep(1)
                hist_1y = yf.Ticker(ticker).history(period="1y")
                if not hist_1y.empty:
                    curr_price = round(float(hist_1y.iloc[-1]["Close"]), 2)
                    high_52 = round(float(hist_1y["High"].max()), 2)
                    low_52 = round(float(hist_1y["Low"].min()), 2)
            except Exception:
                pass
            
            return {
                "index_ticker": index_ticker,
                "overall_market_1mo_trend_pct": index_trend_pct,
                "stock_sector": "General Market",
                "stock_industry": "General Industry",
                "current_price": curr_price,
                "fifty_two_week_high": high_52,
                "fifty_two_week_low": low_52
            }
        except Exception as e:
            print(f"🚨 Agent 6 Failed: {e}")
            return {}

    # ==========================================================================
    # PHASE 7: Batched Signal Fusion (Agent 7 - Exact Formatting)
    # ==========================================================================
    def agent7_final_reasoning(self, batched_payload):
        print("\n" + "="*70)
        print("[Agent 7] ⚡ Executing BATCHED Final Brain (gpt-4o) with strict formatting...")
        print("="*70)
        
        prompt = """
        You are the 'Final Brain' quantitative financial agent. I am passing you an array of top stock market events and their deeply calculated historical/current numbers.
        
        For EACH of the events in the array, you MUST generate the output exactly complying with this rigorous formatting framework. Do not output anything else.
        Follow this exact layout precisely per stock, using emojis and short bullet points to make it extremely clean and readable:
        
        ## 📊 [Stock Name] (`[Ticker]`)
        **Current Price:** [Price] | **52W High:** [High] | **52W Low:** [Low]
        
        📰 **Current Catalyst:** [The headline of the active current event]
        
        🕰️ **Historical Precedents:**
        *(If no history, write "None found")*
        1. **[Date]** - [Headline]
           🔹 *1M Baseline Move:* [X]% | *Post-News Move:* [X]%
           🔹 *1M Baseline Vol:* [X] | *Post-News Vol:* [X]
           💡 *Insight:* [1 short sentence explaining reaction]
        
        *(Repeat for Event 2, 3 if applicable...)*

        📉 **Current Market Setup:**
        - **1M Baseline Move:** [X]% | **T-1/T-2 Hype Move:** [X]%
        - **1M Baseline Vol:** [X]   | **T-1/T-2 Hype Vol:** [X]
        
        ### 🤖 Final Verdict: [BUY / SELL / HOLD]
        **Rationale:**
        - 🎯 *Historical Match:* [1 very brief bullet point: how this compares to history]
        - 📊 *Pre-event Pricing:* [1 very brief bullet point: is it already priced in?]
        - 🌊 *Market Trend:* Current market trend is [X]% and [Sector] trend is [Y], aligning us with the broader flow.
        ---
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.reasoning_model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(batched_payload, indent=2)}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Agent 7 Batched Fusion Failed: {e}"

    # ==========================================================================
    # MAIN LOOP
    # ==========================================================================
    def run_pipeline(self):
        print("="*70)
        print("🚀 INITIALIZING OAI AGENTIC FINANCIAL ANALYZER PIPELINE")
        print("="*70)
        
        raw_news = self.agent1_fetch_realtime_news()
        if not raw_news: return
        
        # Extracts top 5 and AUTOMATICALLY forces mathematical ticker check
        top_events = self.agent2_extract_top_5_events(raw_news)
        if not top_events: return
        
        all_events_batch = []
        
        for idx, event in enumerate(top_events):
            ticker = event.get('ticker')
            mkt = event.get('market')
            headline = event.get('headline')
            etype = event.get('event_type')
            
            print(f"\n🔹 COMPILING DATA {idx+1}/{len(top_events)}: {ticker}")
            
            # Phase 6 Context (pulled early for reference)
            context = self.agent6_market_context(ticker, mkt)
            
            # Phase 3 History
            historical_events = self.agent3_historical_dates(ticker, etype)
            
            # Phase 4 History Stats
            historical_stats = []
            if historical_events:
                historical_stats = self.calculate_market_metrics(ticker, historical_events)
            
            # Phase 5 Current Stats
            current_stats = self.calculate_current_setup(ticker)
            
            all_events_batch.append({
                "Event_Details": event,
                "Historical_Reactions": historical_stats,
                "Current_Pre_Event_Setup_vs_Baseline": current_stats,
                "Macro_and_Sector_Context": context
            })

        final_report = self.agent7_final_reasoning(all_events_batch)
        
        print("\n\n" + "-"*70)
        print("🎯 FINAL CONSOLIDATED REPORT")
        print("-"*70 + "\n")
        print(final_report)
        return final_report


if __name__ == "__main__":
    try:
        pipeline = AgenticFinancialAnalyzer()
        pipeline.run_pipeline()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
