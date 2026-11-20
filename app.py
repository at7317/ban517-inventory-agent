"""
app.py
BAN 517 Inventory Analytics Agent Application.

Features:
1. Natural Language AI Inventory Agent (Customer question -> model identification -> parameter extraction -> Python tool -> verification -> explanation).
2. Direct Interactive Calculators for all 5 Lecture Models (EOQ, Quantity Discount, Safety Stock & ROP, Fixed-Period Review, Newsvendor).
3. Multicriteria Inventory Stratification & CSV Analysis (Sales + POS Hits, Pareto classification, Item counts, Total sales, Top 5 per class, Item 1170 lookup, CSV download).
4. Secure Gemini API Key handling via st.secrets["GEMINI_API_KEY"]. Works seamlessly without an API key for calculations and classification.
"""

import os
import io
import math
import pandas as pd
import numpy as np
import streamlit as st

import inventory_tools
import inventory_agent

# Flag indicating Streamlit availability (import succeeded above)
STREAMLIT_AVAILABLE = True

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BAN 517 Inventory Analytics Agent",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2563EB;
        margin-bottom: 12px;
    }
    .step-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        background-color: #E0E7FF;
        color: #3730A3;
    }
    .card-title {
        font-weight: 600;
        font-size: 1.1rem;
        color: #1F2937;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SIDEBAR & SECRETS CONFIGURATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/warehouse.png", width=64)
    st.title("Inventory Agent")
    st.caption("BAN 517: Operations & Supply Chain Analytics")
    
    st.markdown("---")
    st.subheader("🔑 Gemini API Configuration")
    
    raw_secrets_key = None
    if STREAMLIT_AVAILABLE:
        try:
            if "GEMINI_API_KEY" in st.secrets:
                raw_secrets_key = str(st.secrets["GEMINI_API_KEY"]).strip()
        except Exception:
            pass

    key_is_placeholder = inventory_agent.is_placeholder_key(raw_secrets_key)
    
    if raw_secrets_key and not key_is_placeholder:
        st.info("📝 **Key Configured:** Saved in `.streamlit/secrets.toml`")
        
        # Test Gemini Connection
        if "gemini_conn_status" not in st.session_state:
            st.session_state.gemini_conn_status = None
            st.session_state.gemini_conn_msg = ""

        if st.button("🧪 Verify Gemini Connection", use_container_width=True):
            with st.spinner("Verifying connection to Google Gemini API..."):
                verified, msg = inventory_agent.verify_gemini_connection(raw_secrets_key)
                st.session_state.gemini_conn_status = verified
                st.session_state.gemini_conn_msg = msg

        if st.session_state.gemini_conn_status is True:
            st.success("✅ **Gemini Connection Verified** (Live API Active)")
        elif st.session_state.gemini_conn_status is False:
            st.error(f"❌ **Connection Failed:** {st.session_state.gemini_conn_msg}")
        else:
            st.caption("Click above to verify your key against Gemini.")
    else:
        st.warning("⚠️ **Key Not Configured**")
        if key_is_placeholder:
            st.caption("Placeholder value detected in `.streamlit/secrets.toml`.")
        else:
            st.caption("No `.streamlit/secrets.toml` detected.")
        st.info(
            "To connect Gemini, enter your API key in:\n"
            "`BAN Agent Project/.streamlit/secrets.toml`\n\n"
            "```toml\nGEMINI_API_KEY = \"your-api-key-here\"\n```\n\n"
            "*Note: The CSV Stratification engine and deterministic direct calculators work fully without an API key!*"
        )

    st.markdown("---")
    st.subheader("Lecture Inventory Models")
    st.markdown("""
    1. **EOQ** (Economic Order Quantity)
    2. **Quantity Discount** (Cases A & B)
    3. **Safety Stock & ROP** (Continuous Review)
    4. **Fixed-Period Review** (Periodic Review)
    5. **Newsvendor** (Perishable Single-Period)
    6. **Multicriteria Stratification** (Sales & Hits)
    """)
    st.markdown("---")
    st.caption("UNCW Cameron School of Business • BAN 517")


# -----------------------------------------------------------------------------
# HELPER: RENDER COMPLETE AUDIT CARD
# -----------------------------------------------------------------------------
def render_audit_card(result: dict):
    """
    Renders the required transparent calculation breakdown:
    - Extracted inputs & business synonyms
    - Variable definitions & units
    - Mathematical formula & substituted values
    - Intermediate calculations
    - Final result
    - Model assumptions
    - Managerial explanation
    """
    st.success(f"**Model Selected:** {result.get('model_name', 'Inventory Model')}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 1. Extracted Inputs & Variable Definitions")
        inputs = result.get("extracted_inputs", {})
        units = result.get("units", {})
        var_defs = result.get("variable_definitions", {})

        input_data = []
        for k, v in inputs.items():
            if v is not None and not isinstance(v, (list, dict)):
                unit_str = units.get(k, "")
                desc = var_defs.get(k, k.replace("_", " ").title())
                input_data.append({"Parameter": k, "Extracted Value": f"{v:g}" if isinstance(v, (int, float)) else str(v), "Unit": unit_str, "Definition": desc})
        if input_data:
            st.dataframe(pd.DataFrame(input_data), use_container_width=True, hide_index=True)

        st.markdown("#### 2. Mathematical Formula")
        formula_latex = result.get("formula", "")
        if formula_latex:
            st.latex(formula_latex)

        st.markdown("#### 3. Substituted Values & Calculation Trace")
        sub_vals = result.get("substituted_values", "")
        st.code(sub_vals, language="text")

    with col2:
        st.markdown("#### 4. Final Numerical Results")
        final_res = result.get("final_result", {})
        metrics = []
        for k, v in final_res.items():
            if isinstance(v, (int, float)):
                formatted_v = f"{v:,.2f}" if isinstance(v, float) else f"{v:,}"
                metrics.append({"Metric": k.replace("_", " ").title(), "Value": formatted_v})
        if metrics:
            st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)

        st.markdown("#### 5. Step-by-Step Intermediate Calculations")
        intermediates = result.get("intermediate_calculations", {})
        if intermediates:
            if "tier_comparisons" in intermediates:
                st.markdown("**Discount Tier Comparison Table:**")
                tiers_df = pd.DataFrame(intermediates["tier_comparisons"])
                st.dataframe(tiers_df, use_container_width=True, hide_index=True)
            else:
                inter_items = [
                    {"Intermediate Step": k.replace("_", " ").title(), "Calculated Value": str(v)}
                    for k, v in intermediates.items()
                ]
                st.dataframe(pd.DataFrame(inter_items), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 6. Managerial Explanation & Operational Takeaway")
    st.info(f"💡 **Managerial Advice:** {result.get('managerial_explanation', '')}")

    with st.expander("🔍 View Explicit Model Assumptions", expanded=False):
        for assumption in result.get("assumptions", []):
            st.markdown(f"- {assumption}")


# -----------------------------------------------------------------------------
# MAIN APP TABS
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">📦 BAN 517 Inventory Analytics Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Assisted Decision Making, Deterministic Calculations & Multicriteria Inventory Stratification</div>', unsafe_allow_html=True)

tab_agent, tab_classification, tab_calculators = st.tabs([
    "💬 AI Decision Agent",
    "📊 CSV Multicriteria Stratification",
    "🧮 Direct Interactive Calculators"
])


# =============================================================================
# TAB 1: AI DECISION AGENT
# =============================================================================
with tab_agent:
    st.subheader("Inventory Question & Decision Assistant")
    st.markdown("""
    Ask an operational question in plain business English. The agent adheres strictly to the lecture architecture:
    `Customer question -> identify model -> extract inputs -> select Python tool -> calculate -> verify -> explain.`
    """)

    # Quick test prompts buttons
    st.caption("Try an example business scenario:")
    col_ex1, col_ex2, col_ex3, col_ex4, col_ex5 = st.columns(5)
    example_query = None

    if col_ex1.button("EOQ Example", help="Annual demand, setup cost, holding cost"):
        example_query = (
            "We sell 12,000 units per year. Setup cost is $50 per order and inventory carrying cost "
            "is $2.00 per unit per year. The supplier lead time is 6 days. What is our optimal order quantity and reorder point?"
        )
    if col_ex2.button("Discount Example", help="Quantity discount evaluation"):
        example_query = (
            "Our annual demand is 10,000 units and ordering cost is $20. Holding cost is fixed at $4 per unit/year. "
            "Supplier offers discounts: 1-499 units at $5.00, 500-999 units at $4.50, and 1000+ units at $3.90. What should we order?"
        )
    if col_ex3.button("Safety Stock Example", help="Uncertain demand and lead time"):
        example_query = (
            "Average daily demand is 80 units with a standard deviation of 10. Replenishment lead time is 9 days. "
            "We want to limit our stockout risk to no more than 5%. What safety stock and reorder point do we need?"
        )
    if col_ex4.button("Fixed Period Example", help="Periodic review and protection window"):
        example_query = (
            "We review inventory every 7 days (review period). Lead time is 3 days. Mean daily demand is 25 units "
            "with std dev of 5 units. We have 50 units on hand and 20 on order. We target a 95% service level. How many units should we order?"
        )
    if col_ex5.button("Newsvendor Example", help="Single-period perishable good"):
        example_query = (
            "We sell fresh seasonal bouquets for $30 retail. Each bouquet costs $12 wholesale. Unsold bouquets are salvaged "
            "for $4. Demand is normally distributed with mean 200 and standard deviation 40. How many bouquets should we stock?"
        )

    api_key = inventory_agent.get_gemini_api_key()
    agent = inventory_agent.InventoryAnalyticsAgent(api_key=api_key)

    user_query = st.text_area(
        "Enter your inventory scenario or business question:",
        value=example_query if example_query else "",
        placeholder="e.g. Annual demand is 5,000 units, order fee is $40, holding cost is $1.50 per unit. When and how much should I order?",
        height=100,
    )

    if st.button("🚀 Analyze with Inventory Agent", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a question or click one of the example scenario buttons above.")
        else:
            with st.spinner("Analyzing question, mapping synonyms, and executing deterministic tool..."):
                response = agent.process_query(user_query)

            lifecycle = response["lifecycle"]
            is_gemini = lifecycle.get("gemini_connection_verified", False)
            engine_name = lifecycle.get("interpretation_engine", "Deterministic Fallback Parser (Offline)")

            # Visual progress pill row
            st.markdown(
                f"""
                <div style="margin-bottom: 16px;">
                    <span class="step-pill">1. Question Received</span>
                    <span class="step-pill">2. Model: {lifecycle.get('step_2_identified_model', 'N/A')}</span>
                    <span class="step-pill">3. Inputs Extracted</span>
                    <span class="step-pill">4. Tool: {lifecycle.get('step_4_selected_tool', 'N/A')}</span>
                    <span class="step-pill">5. Deterministic Math</span>
                    <span class="step-pill">6. Verified: {'✅' if lifecycle.get('step_6_verification', {}).get('passed') else '⚠️'}</span>
                    <span class="step-pill">7. Managerial Insight</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Explicit engine badge distinguishing Gemini from Fallback
            if is_gemini:
                st.success(f"🤖 **Interpretation Engine:** {engine_name} — Live Gemini API connection verified.")
            else:
                st.warning(f"⚙️ **Interpretation Engine:** {engine_name} — Offline rule-based parsing was used (not a Gemini connection).")

            if lifecycle.get("api_notice"):
                st.info(f"ℹ️ **Agent Notice:** {lifecycle['api_notice']}")

            if response["status"] == "clarification_needed":
                st.warning(f"⚠️ {response['message']}")
            elif response["status"] == "error":
                st.error(f"❌ {response['message']}")
            elif response["status"] == "success" and response["tool_output"]:
                render_audit_card(response["tool_output"])


# =============================================================================
# TAB 2: CSV MULTICRITERIA STRATIFICATION
# =============================================================================
with tab_classification:
    st.subheader("Multicriteria Inventory Stratification (Sales & POS Hits)")
    st.markdown(r"""
    Conducts multicriteria stratification following `Inventory_Classification_Methodology.pdf`:
    1. **Independent Pareto Ranking**: Classify Sales and POS Hits separately (A: $\le 60\%$, B: $60-80\%$, C: $80-90\%$, D: $> 90\%$).
    2. **Numerical Scoring**: A = 4, B = 3, C = 2, D = 1.
    3. **Weighted Combined Score**: $\text{Score} = 0.65 \times \text{Sales Score} + 0.35 \times \text{Hits Score}$.
    4. **Final Inventory Class**: A ($\ge 3.25$), B ($2.50 - 3.24$), C ($1.75 - 2.49$), D ($< 1.75$).
    """)

    col_upload, col_default = st.columns([2, 1])
    uploaded_file = col_upload.file_uploader("Upload CSV Dataset", type=["csv"], help="Upload an inventory CSV file")
    load_default = col_default.button("📁 Load Default `ProjectDataV2_UTF8.csv`", use_container_width=True)

    df_to_analyze = None

    default_csv_path = os.path.join(os.path.dirname(__file__), "ProjectDataV2_UTF8.csv")
if uploaded_file is not None:
    try:
        df_to_analyze = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded uploaded CSV: **{uploaded_file.name}**")
    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")
elif load_default or os.path.exists(default_csv_path):
    if os.path.exists(default_csv_path):
        df_to_analyze = pd.read_csv(default_csv_path)
        if load_default:
            st.success("Loaded project dataset: `ProjectDataV2_UTF8.csv`")
        try:
            df_to_analyze = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded uploaded CSV: **{uploaded_file.name}**")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
    elif load_default or os.path.exists("ProjectDataV2_UTF8.csv"):
        if os.path.exists("ProjectDataV2_UTF8.csv"):
            df_to_analyze = pd.read_csv("ProjectDataV2_UTF8.csv")
            if load_default:
                st.success("Loaded project dataset: `ProjectDataV2_UTF8.csv`")

    if df_to_analyze is not None:
        # Display dataset metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Rows / Items", f"{len(df_to_analyze):,}")
        m2.metric("Total Columns", f"{len(df_to_analyze.columns)}")
        m3.metric("Column Names", ", ".join(list(df_to_analyze.columns)))

        with st.expander("👀 View Raw Uploaded Data", expanded=False):
            st.dataframe(df_to_analyze.head(20), use_container_width=True)

        # Run Multicriteria Classification
        with st.spinner("Executing Pareto stratification and multicriteria scoring..."):
            classification_results = inventory_tools.classify_inventory_multicriteria(df_to_analyze)

        st.markdown("---")
        st.subheader("📋 Required Classification Summary")

        summary_df = classification_results["summary_table"]
        st.dataframe(
            summary_df.style.format({
                "Total Sales ($)": "${:,.2f}",
                "% of Total Items": "{:.2f}%",
                "% of Total Sales": "{:.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Visualizations
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("**Sales Breakdown by Final Class**")
            chart_sales = summary_df.set_index("Final Inventory Class")[["Total Sales ($)"]]
            st.bar_chart(chart_sales)

        with chart_col2:
            st.markdown("**Item Count by Final Class**")
            chart_items = summary_df.set_index("Final Inventory Class")[["Number of Items"]]
            st.bar_chart(chart_items)

        st.markdown("---")
        # Lookup for Item Number 1170
        st.subheader("🔍 Specific Item Lookup: Item Number 1170")
        item_1170 = classification_results["item_1170_lookup"]
        if "Item Name" in item_1170:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Item Number", f"{item_1170.get('Item Number')}")
            c2.metric("Item Name", f"{item_1170.get('Item Name')}")
            c3.metric("Sales ($)", f"${item_1170.get('Sales ($)', 0):,.2f}")
            c4.metric("POS Hits", f"{item_1170.get('Hits', 0):,.0f}")
            c5.metric("Final Class", f"Class {item_1170.get('Final_Class')} (Score: {item_1170.get('Combined_Score', 0):.2f})")

            st.markdown(
                f"""
                > **Item 1170 Detail:** Sales: **${item_1170.get('Sales ($)', 0):,.2f}** (Cum: {item_1170.get('Sales_Cum_Pct', 0):.2f}% -> Class **{item_1170.get('Sales_Class')}**, Score: **{item_1170.get('Sales_Score')}**) | 
                > Hits: **{item_1170.get('Hits', 0):,.0f}** (Cum: {item_1170.get('Hits_Cum_Pct', 0):.2f}% -> Class **{item_1170.get('Hits_Class')}**, Score: **{item_1170.get('Hits_Score')}**) | 
                > Combined Score: $0.65(4) + 0.35(4) = \\mathbf{{{item_1170.get('Combined_Score', 0):.2f}}}$ $\\rightarrow$ **Class {item_1170.get('Final_Class')}**.
                """
            )
        else:
            st.info("Item 1170 is not present in this dataset.")

        st.markdown("---")
        st.subheader("⭐ Five Highest-Sales Items in Each Final Class")
        top5_dict = classification_results["top5_per_class"]
        for cls in ["A", "B", "C", "D"]:
            with st.expander(f"Top 5 Items in Class {cls}", expanded=(cls == "A")):
                top_df = pd.DataFrame(top5_dict[cls])
                st.dataframe(
                    top_df.style.format({
                        "Sales ($)": "${:,.2f}",
                        "Hits": "{:,.0f}",
                        "Combined_Score": "{:.2f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("---")
        st.subheader("📄 Complete Stratified Dataset")
        full_table = classification_results["full_table"]
        st.dataframe(
            full_table.style.format({
                "Sales ($)": "${:,.2f}",
                "Hits": "{:,.0f}",
                "Sales_Cum_Pct": "{:.2f}%",
                "Hits_Cum_Pct": "{:.2f}%",
                "Combined_Score": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Download CSV
        csv_buffer = io.StringIO()
        full_table.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Full Stratified Classification (CSV)",
            data=csv_buffer.getvalue(),
            file_name="multicriteria_inventory_classification.csv",
            mime="text/csv",
            type="primary",
        )


# =============================================================================
# TAB 3: DIRECT INTERACTIVE CALCULATORS
# =============================================================================
with tab_calculators:
    st.subheader("Deterministic Calculators for All 5 Lecture Models")
    st.markdown("Test formulas directly without natural language interpretation.")

    calc_model = st.selectbox(
        "Choose Inventory Model to Calculate:",
        [
            "1. Economic Order Quantity (EOQ)",
            "2. Quantity Discount Problem",
            "3. Safety Stock & Reorder Point",
            "4. Fixed-Period Review Model",
            "5. Newsvendor (Perishable Single-Period)",
        ],
    )

    if calc_model.startswith("1."):
        st.markdown("### Model 1: Economic Order Quantity (EOQ)")
        c1, c2, c3 = st.columns(3)
        d_val = c1.number_input("Annual Demand (D, units/yr)", min_value=1.0, value=12000.0, step=100.0)
        s_val = c2.number_input("Setup / Ordering Cost (S, $/order)", min_value=0.01, value=50.0, step=5.0)
        h_val = c3.number_input("Holding / Carrying Cost (H, $/unit/yr)", min_value=0.01, value=2.00, step=0.10)

        c4, c5, c6 = st.columns(3)
        unit_c = c4.number_input("Unit Purchase Price (C, $/unit, optional)", min_value=0.0, value=10.0, step=1.0)
        lt_days = c5.number_input("Replenishment Lead Time (days, optional)", min_value=0.0, value=6.0, step=1.0)
        work_days = c6.number_input("Operating Days per Year", min_value=1.0, value=365.0, step=1.0)

        if st.button("Calculate EOQ", type="primary"):
            res = inventory_tools.calculate_eoq(
                demand=d_val,
                setup_cost=s_val,
                holding_cost=h_val,
                unit_cost=unit_c if unit_c > 0 else None,
                lead_time_days=lt_days if lt_days > 0 else None,
                work_days_per_year=work_days,
            )
            render_audit_card(res)

    elif calc_model.startswith("2."):
        st.markdown("### Model 2: Quantity Discount Problem")
        c1, c2 = st.columns(2)
        d_val = c1.number_input("Annual Demand (D, units/yr)", min_value=1.0, value=10000.0, step=100.0, key="qd_d")
        s_val = c2.number_input("Setup Cost (S, $/order)", min_value=0.01, value=20.0, step=5.0, key="qd_s")

        case_opt = st.radio("Holding Cost Structure:", ["Case A: Fixed Dollar Holding Cost ($/unit/yr)", "Case B: Holding Cost % of Unit Price (I)"])
        if "Case A" in case_opt:
            h_fixed = st.number_input("Holding Cost H ($/unit/yr)", min_value=0.01, value=4.00, step=0.10)
            h_rate = None
        else:
            h_rate = st.number_input("Holding Cost Rate I (decimal, e.g. 0.20 for 20%)", min_value=0.001, max_value=1.0, value=0.20, step=0.01)
            h_fixed = None

        st.markdown("**Discount Tiers:**")
        demo_schedule = [
            {"min_qty": 1, "max_qty": 499, "unit_price": 5.00},
            {"min_qty": 500, "max_qty": 999, "unit_price": 4.50},
            {"min_qty": 1000, "max_qty": None, "unit_price": 3.90},
        ]
        tiers_df = st.data_editor(pd.DataFrame(demo_schedule), num_rows="dynamic", use_container_width=True)

        if st.button("Evaluate Quantity Discounts", type="primary"):
            tiers_list = tiers_df.to_dict(orient="records")
            res = inventory_tools.calculate_quantity_discount(
                demand=d_val,
                setup_cost=s_val,
                discount_schedule=tiers_list,
                holding_cost_fixed=h_fixed,
                holding_cost_rate=h_rate,
            )
            render_audit_card(res)

    elif calc_model.startswith("3."):
        st.markdown("### Model 3: Safety Stock and Reorder Point (Continuous Review)")
        sl_mode = st.radio("Specify Service Level by:", ["Service Level (e.g. 95%)", "Stockout Risk (e.g. 5%)"])
        if "Service Level" in sl_mode:
            sl_val = st.slider("Cycle Service Level", min_value=0.50, max_value=0.999, value=0.95, step=0.005)
            risk_val = None
        else:
            risk_val = st.slider("Allowable Stockout Risk", min_value=0.001, max_value=0.50, value=0.05, step=0.005)
            sl_val = None

        case_sel = st.selectbox("Uncertainty Structure:", [
            "Case 1: Direct Lead Time Demand (mean & std dev)",
            "Case 2: Variable Daily Demand & Constant Lead Time",
            "Case 3: Constant Daily Demand & Variable Lead Time",
            "Case 4: Variable Daily Demand & Variable Lead Time",
        ])

        if "Case 1" in case_sel:
            c1, c2 = st.columns(2)
            ltd_m = c1.number_input("Lead Time Demand Mean (units)", min_value=0.1, value=50.0)
            ltd_s = c2.number_input("Lead Time Demand Std Dev (units)", min_value=0.0, value=5.0)
            args = {"lead_time_demand_mean": ltd_m, "lead_time_demand_std": ltd_s}
        elif "Case 2" in case_sel:
            c1, c2, c3 = st.columns(3)
            d_m = c1.number_input("Mean Daily Demand (units/day)", min_value=0.1, value=100.0)
            d_s = c2.number_input("Daily Demand Std Dev (units/day)", min_value=0.0, value=15.0)
            lt_d = c3.number_input("Constant Lead Time (days)", min_value=0.1, value=4.0)
            args = {"daily_demand_mean": d_m, "daily_demand_std": d_s, "lead_time_days": lt_d}
        elif "Case 3" in case_sel:
            c1, c2, c3 = st.columns(3)
            d_m = c1.number_input("Constant Daily Demand (units/day)", min_value=0.1, value=100.0)
            lt_m = c2.number_input("Mean Lead Time (days)", min_value=0.1, value=4.0)
            lt_s = c3.number_input("Lead Time Std Dev (days)", min_value=0.01, value=1.0)
            args = {"daily_demand_mean": d_m, "lead_time_days": lt_m, "lead_time_std_days": lt_s}
        else:
            c1, c2 = st.columns(2)
            d_m = c1.number_input("Mean Daily Demand", min_value=0.1, value=100.0, key="c4_dm")
            d_s = c2.number_input("Daily Demand Std Dev", min_value=0.01, value=15.0, key="c4_ds")
            c3, c4 = st.columns(2)
            lt_m = c3.number_input("Mean Lead Time (days)", min_value=0.1, value=4.0, key="c4_ltm")
            lt_s = c4.number_input("Lead Time Std Dev (days)", min_value=0.01, value=1.0, key="c4_lts")
            args = {"daily_demand_mean": d_m, "daily_demand_std": d_s, "lead_time_days": lt_m, "lead_time_std_days": lt_s}

        if st.button("Calculate Safety Stock & ROP", type="primary"):
            res = inventory_tools.calculate_safety_stock_rop(service_level=sl_val, stockout_risk=risk_val, **args)
            render_audit_card(res)

    elif calc_model.startswith("4."):
        st.markdown("### Model 4: Fixed-Period Review Model (Periodic Review)")
        c1, c2, c3 = st.columns(3)
        t_rev = c1.number_input("Review Period (T, days)", min_value=1.0, value=7.0, step=1.0)
        l_lead = c2.number_input("Lead Time (L, days)", min_value=0.0, value=3.0, step=1.0)
        sl = c3.slider("Service Level", min_value=0.50, max_value=0.999, value=0.95, step=0.005, key="fpr_sl")

        c4, c5 = st.columns(2)
        d_mean = c4.number_input("Mean Daily Demand (units/day)", min_value=0.1, value=25.0)
        d_std = c5.number_input("Daily Demand Std Dev (units/day)", min_value=0.0, value=5.0)

        c6, c7, c8 = st.columns(3)
        on_hand = c6.number_input("Current On-Hand Inventory", min_value=0.0, value=50.0)
        on_order = c7.number_input("Current On-Order Inventory", min_value=0.0, value=20.0)
        backorders = c8.number_input("Backorders", min_value=0.0, value=0.0)

        if st.button("Calculate Target Inventory & Order Quantity", type="primary"):
            res = inventory_tools.calculate_fixed_period_review(
                review_period=t_rev,
                lead_time=l_lead,
                daily_demand_mean=d_mean,
                daily_demand_std=d_std,
                service_level=sl,
                current_on_hand=on_hand,
                current_on_order=on_order,
                backorders=backorders,
            )
            render_audit_card(res)

    elif calc_model.startswith("5."):
        st.markdown("### Model 5: Newsvendor Model (Perishable / Single-Period)")
        c1, c2, c3 = st.columns(3)
        p_price = c1.number_input("Selling Price ($/unit)", min_value=0.01, value=30.0, step=1.0)
        c_cost = c2.number_input("Procurement Cost ($/unit)", min_value=0.01, value=12.0, step=1.0)
        v_salvage = c3.number_input("Salvage Value ($/unit)", min_value=0.0, value=4.0, step=0.5)

        c4, c5, c6 = st.columns(3)
        g_penalty = c4.number_input("Shortage Goodwill Penalty ($/unit, optional)", min_value=0.0, value=0.0, step=0.5)
        d_mean = c5.number_input("Demand Mean (units)", min_value=1.0, value=200.0, step=10.0)
        d_std = c6.number_input("Demand Std Dev (units)", min_value=0.0, value=40.0, step=5.0)

        if st.button("Calculate Newsvendor Stocking Policy", type="primary"):
            res = inventory_tools.calculate_newsvendor(
                selling_price=p_price,
                unit_cost=c_cost,
                salvage_value=v_salvage,
                shortage_penalty=g_penalty,
                demand_mean=d_mean,
                demand_std=d_std,
            )
            render_audit_card(res)
