import streamlit as st
import time
from new import AgenticFinancialAnalyzer, OPENAI_AVAILABLE, OPENAI_API_KEY

# ==============================================================================
# PAGE CONFIGURATION (Attractive Premium UI)
# ==============================================================================
st.set_page_config(
    page_title="Antigravity Financial Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for Glassmorphism & Modern Styling
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Gradient */
    .hero-title {
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
    }
    
    .hero-subtitle {
        text-align: center;
        color: #8b949e;
        font-size: 1.2rem;
        margin-bottom: 40px;
    }
    
    /* Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #4ECDC4, #556270);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 1.2rem;
        font-weight: bold;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(78, 205, 196, 0.4);
    }
    
    /* Output Markdown Tweaks */
    .report-container {
        background-color: #161b22;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border: 1px solid #30363d;
        margin-top: 20px;
    }
    
    hr { margin-top: 40px; margin-bottom: 40px; border-color: #30363d; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# UI STRUCTURE
# ==============================================================================
st.markdown("<h1 class='hero-title'>Agentic Financial Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Autonomous Market Scanning & Quantitative Signal Fusion</p>", unsafe_allow_html=True)

# Check Prerequisites
if not OPENAI_AVAILABLE:
    st.error("🚨 OpenAI library is not installed. Please run `pip install openai` in your terminal.")
    st.stop()
    
try:
    if not OPENAI_API_KEY and "OPENAI_API_KEY" not in st.secrets:
        st.warning("⚠️ OPENAI_API_KEY is not set. You must add it to your Streamlit secrets.")
        st.stop()
except Exception:
    if not OPENAI_API_KEY:
        st.warning("⚠️ OPENAI_API_KEY is not set.")
        st.stop()

# Layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    start_scan = st.button("🚀 INITIATE MARKET SCAN", use_container_width=True)

# Variables to hold execution state
top_events = []
final_report = ""
pipeline_success = False

# Pipeline Execution
if start_scan:
    st.markdown("---")
    
    # Create the visual status pipeline
    with st.status("🤖 Autonomous Agents Active...", expanded=True) as status:
        st.write("📡 Agent 1: Scanning global news feeds (US & India)...")
        
        try:
            # Instantiate our pipeline natively
            pipeline = AgenticFinancialAnalyzer()
            
            # Step 1: Real-time News
            raw_news = pipeline.agent1_fetch_realtime_news()
            st.write(f"🧠 Agent 2: Filtering massive dataset down to Top 5 specific events...")
            
            # Step 2: Extract Events
            top_events = pipeline.agent2_extract_top_5_events(raw_news)
            
            if top_events:
                st.write(f"✅ Fast Models successfully identified catalyst events.")
                st.write(f"📈 Agents 3-6: Digging deep into historical correlation, math deviation grids, and indices...")
                
                # Master array
                all_events_batch = []
                
                # Setup progress bar
                progress_bar = st.progress(0)
                
                for idx, event in enumerate(top_events):
                    ticker = event.get('ticker')
                    mkt = event.get('market')
                    headline = event.get('headline')
                    etype = event.get('event_type')
                    
                    st.write(f"&nbsp;&nbsp;&nbsp; └ Processing quantitative stack for: {ticker}...")
                    
                    # Context + Math Pipeline
                    context = pipeline.agent6_market_context(ticker, mkt)
                    historical_events = pipeline.agent3_historical_dates(ticker, etype)
                    
                    historical_stats = []
                    if historical_events:
                        historical_stats = pipeline.calculate_market_metrics(ticker, historical_events)
                    
                    current_stats = pipeline.calculate_current_setup(ticker)

                    all_events_batch.append({
                        "Event_Details": event,
                        "Historical_Reactions": historical_stats,
                        "Current_Pre_Event_Setup_vs_Baseline": current_stats,
                        "Macro_and_Sector_Context": context
                    })
                    
                    progress_bar.progress((idx + 1) / len(top_events))
                
                st.write("⚡ Agent 7: Final Brain (gpt-4o) parsing all massive data arrays into the Final Verdict...")
                
                # Final Execute
                final_report = pipeline.agent7_final_reasoning(all_events_batch)
                
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                pipeline_success = True
                
            else:
                status.update(label="❌ Failed to extract events.", state="error", expanded=True)
                st.error("The LLM could not extract valid events.")
                
        except Exception as e:
            status.update(label="🚨 Pipeline crashed", state="error", expanded=True)
            st.error(f"FATAL ERROR: {str(e)}")

# Render beautifully in the UI (OUTSIDE THE STATUS BLOCK)
if pipeline_success and final_report:
    st.markdown("<div class='report-container'>", unsafe_allow_html=True)
    st.markdown(final_report)
    st.markdown("</div>", unsafe_allow_html=True)
