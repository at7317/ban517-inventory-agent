"""
test_inventory_tools.py
Comprehensive Unit Test Suite for BAN 517 Inventory Analytics Agent.

Tests all five lecture models and multicriteria inventory stratification:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Problem
3. Safety Stock & Reorder Point (Continuous Review)
4. Fixed-Period Review Model (Periodic Review)
5. Newsvendor Model (Single-Period / Perishable)
6. Multicriteria Inventory Classification (Sales + Hits)
"""

import math
import os
import pytest
import pandas as pd
import numpy as np

import inventory_tools


# =====================================================================
# 1. EOQ TESTS
# =====================================================================

def test_eoq_basic():
    # D = 1000, S = $10, H = $2/unit/year
    # Q* = sqrt(2 * 1000 * 10 / 2) = sqrt(10000) = 100
    res = inventory_tools.calculate_eoq(
        demand=1000,
        setup_cost=10,
        holding_cost=2,
        unit_cost=5,
        lead_time_days=5,
        work_days_per_year=250,
    )
    assert res["final_result"]["optimal_order_quantity"] == 100.0
    assert res["final_result"]["recommended_order_quantity_int"] == 100
    assert res["final_result"]["annual_setup_cost"] == 100.0
    assert res["final_result"]["annual_holding_cost"] == 100.0
    assert res["final_result"]["total_annual_inventory_cost"] == 200.0
    assert res["final_result"]["total_cost_including_purchases"] == 5200.0
    assert res["final_result"]["orders_per_year"] == 10.0
    # Daily demand = 1000 / 250 = 4, ROP = 4 * 5 = 20
    assert res["final_result"]["reorder_point"] == 20.0
    assert "formula" in res
    assert "substituted_values" in res
    assert "assumptions" in res
    assert "managerial_explanation" in res


def test_eoq_validation():
    # Negative demand
    with pytest.raises(ValueError, match="Demand"):
        inventory_tools.calculate_eoq(demand=-100, setup_cost=10, holding_cost=2)
    # Zero setup cost
    with pytest.raises(ValueError, match="Setup"):
        inventory_tools.calculate_eoq(demand=100, setup_cost=0, holding_cost=2)
    # Negative holding cost
    with pytest.raises(ValueError, match="Holding"):
        inventory_tools.calculate_eoq(demand=100, setup_cost=10, holding_cost=-2)
    # Inconsistent time units
    with pytest.raises(ValueError, match="consistent"):
        inventory_tools.calculate_eoq(
            demand=100, setup_cost=10, holding_cost=2,
            demand_time_unit="year", holding_cost_time_unit="month"
        )


# =====================================================================
# 2. QUANTITY DISCOUNT TESTS
# =====================================================================

def test_quantity_discount_case_a():
    # D = 10,000, S = $20, H = $4 (fixed)
    # Schedule:
    # Tier 1: 1 to 499: $5.00 -> EOQ = sqrt(2*10000*20/4) = 316.23 (feasible)
    # Tier 2: 500 to 999: $4.50 -> EOQ = 316.23 < 500 -> candidate Q = 500
    # Tier 3: 1000+: $3.90 -> EOQ = 316.23 < 1000 -> candidate Q = 1000
    schedule = [
        {"min_qty": 1, "max_qty": 499, "unit_price": 5.00},
        {"min_qty": 500, "max_qty": 999, "unit_price": 4.50},
        {"min_qty": 1000, "max_qty": None, "unit_price": 3.90},
    ]
    res = inventory_tools.calculate_quantity_discount(
        demand=10000,
        setup_cost=20,
        discount_schedule=schedule,
        holding_cost_fixed=4.0,
    )
    # Tier 3 candidate Q=1000 gives: Purchase = 39,000, Ordering = (10000/1000)*20 = 200, Holding = (1000/2)*4 = 2000 => TC = 41,200
    # Tier 1 Q=316.23 gives: Purchase = 50,000 => TC > 50,000
    # Tier 2 Q=500 gives: Purchase = 45,000 => TC > 45,000
    assert res["final_result"]["optimal_tier"] == 3
    assert res["final_result"]["optimal_order_quantity"] == 1000.0
    assert res["final_result"]["total_annual_cost"] == 41200.0
    assert len(res["intermediate_calculations"]["tier_comparisons"]) == 3


def test_quantity_discount_case_b():
    # Case B: Holding cost is 20% of unit price (I = 0.20)
    schedule = [
        {"min_qty": 1, "max_qty": 999, "unit_price": 10.00},
        {"min_qty": 1000, "max_qty": None, "unit_price": 8.00},
    ]
    res = inventory_tools.calculate_quantity_discount(
        demand=2000,
        setup_cost=50,
        discount_schedule=schedule,
        holding_cost_rate=0.20,
    )
    # Tier 1: H = 0.2*10 = 2.0 -> EOQ = sqrt(2*2000*50/2) = 316.23 (feasible)
    # TC1 = 2000*10 + (2000/316.23)*50 + (316.23/2)*2 = 20000 + 316.23 + 316.23 = 20632.46
    # Tier 2: H = 0.2*8 = 1.6 -> EOQ = sqrt(2*2000*50/1.6) = 353.55 < 1000 -> candidate Q = 1000
    # TC2 = 2000*8 + (2000/1000)*50 + (1000/2)*1.6 = 16000 + 100 + 800 = 16900.0
    assert res["final_result"]["optimal_tier"] == 2
    assert res["final_result"]["optimal_order_quantity"] == 1000.0
    assert res["final_result"]["total_annual_cost"] == 16900.0


# =====================================================================
# 3. SAFETY STOCK & ROP TESTS
# =====================================================================

def test_safety_stock_rop_case_1():
    # Direct lead time demand: mu_L = 50, sigma_L = 5, risk = 0.05 => SL = 0.95
    res = inventory_tools.calculate_safety_stock_rop(
        stockout_risk=0.05,
        lead_time_demand_mean=50,
        lead_time_demand_std=5,
    )
    # Z for 0.95 ~ 1.64485
    # SS = 1.64485 * 5 = 8.22
    # ROP = 50 + 8.22 = 58.22
    assert pytest.approx(res["final_result"]["z_score"], 0.01) == 1.645
    assert pytest.approx(res["final_result"]["safety_stock"], 0.05) == 8.22
    assert pytest.approx(res["final_result"]["reorder_point"], 0.05) == 58.22


def test_safety_stock_rop_weeks_to_days():
    # Variable daily demand d_bar = 20, sigma_d = 4
    # Lead time in weeks: 2 weeks, std = 0.5 weeks. Operating 5 days/week.
    # Converted lead time: mean = 2 * 5 = 10 days, std = 0.5 * 5 = 2.5 days.
    res = inventory_tools.calculate_safety_stock_rop(
        service_level=0.95,
        daily_demand_mean=20,
        daily_demand_std=4,
        lead_time_weeks=2,
        lead_time_std_weeks=0.5,
        days_per_week=5.0,
    )
    # Case 4 applies
    # Variance LTD = 10 * (4^2) + (20^2) * (2.5^2) = 160 + 400 * 6.25 = 160 + 2500 = 2660
    # Sigma LTD = sqrt(2660) ~ 51.575
    # SS = 1.64485 * 51.575 ~ 84.83
    assert pytest.approx(res["extracted_inputs"]["lead_time_days"], 0.01) == 10.0
    assert pytest.approx(res["extracted_inputs"]["lead_time_std_days"], 0.01) == 2.5
    assert any("5 days/week" in a for a in res["assumptions"])


# =====================================================================
# 4. FIXED-PERIOD REVIEW TESTS
# =====================================================================

def test_fixed_period_review_basic():
    # Review period T = 7 days, Lead time L = 3 days => Protection Period = 10 days
    # d_bar = 25 units/day, sigma_d = 5 units/day, SL = 0.95
    # Protection demand: mean = 25 * 10 = 250, sigma = 5 * sqrt(10) ~ 15.811
    # SS = 1.64485 * 15.811 ~ 26.01
    # Target M = 250 + 26.01 = 276.01
    # On hand = 50, On order = 20, Backorders = 0 => IP = 70
    # Order quantity q = 276.01 - 70 = 206.01
    res = inventory_tools.calculate_fixed_period_review(
        review_period=7,
        lead_time=3,
        daily_demand_mean=25,
        daily_demand_std=5,
        service_level=0.95,
        current_on_hand=50,
        current_on_order=20,
        backorders=0,
    )
    assert res["final_result"]["protection_period"] == 10.0
    assert pytest.approx(res["final_result"]["target_inventory_level"], 0.1) == 276.01
    assert pytest.approx(res["final_result"]["order_quantity"], 0.1) == 206.01


def test_fixed_period_review_order_quantity_non_negative():
    # When IP > Target M, order quantity must be bounded below by 0
    res = inventory_tools.calculate_fixed_period_review(
        review_period=7,
        lead_time=3,
        daily_demand_mean=25,
        daily_demand_std=5,
        service_level=0.95,
        current_on_hand=400, # Much greater than M ~ 276
    )
    assert res["final_result"]["order_quantity"] == 0.0


# =====================================================================
# 5. NEWSVENDOR TESTS
# =====================================================================

def test_newsvendor_normal():
    # Selling price P = $100, Cost C = $60, Salvage V = $20
    # Cu = 100 - 60 = 40, Co = 60 - 20 = 40
    # CR = 40 / (40 + 40) = 0.50 => Z = 0.0 => Q* = demand_mean
    res = inventory_tools.calculate_newsvendor(
        selling_price=100,
        unit_cost=60,
        salvage_value=20,
        demand_mean=500,
        demand_std=50,
    )
    assert res["final_result"]["cost_underage"] == 40.0
    assert res["final_result"]["cost_overage"] == 40.0
    assert res["final_result"]["critical_ratio"] == 0.50
    assert res["final_result"]["optimal_order_quantity"] == 500.0


def test_newsvendor_validation():
    # Unit cost >= Selling price
    with pytest.raises(ValueError, match="profitable"):
        inventory_tools.calculate_newsvendor(selling_price=50, unit_cost=60)
    # Salvage value > Unit cost
    with pytest.raises(ValueError, match="exceed unit cost"):
        inventory_tools.calculate_newsvendor(selling_price=100, unit_cost=50, salvage_value=60)


# =====================================================================
# 6. MULTICRITERIA INVENTORY STRATIFICATION TESTS
# =====================================================================

def test_multicriteria_classification_synthetic():
    # Synthetic dataset with known tie-breakers and values
    data = {
        "Item Number": [101, 102, 103, 104],
        "Item Name": ["Item A", "Item B", "Item C", "Item D"],
        "Sales ($)": [600, 200, 100, 100], # Total = 1000. Pct: 60% (A), 80% (B), 90% (C), 100% (D)
        "Hits": [10, 30, 40, 20],          # Sorted: 103: 40 (40%), 102: 30 (70%), 104: 20 (90%), 101: 10 (100%)
    }
    df = pd.DataFrame(data)
    res = inventory_tools.classify_inventory_multicriteria(df)
    full = res["full_table"]

    # Check Item 101: Sales = 600 (60% -> Class A = 4), Hits = 10 (100% -> Class D = 1)
    # Combined score = 0.65 * 4 + 0.35 * 1 = 2.60 + 0.35 = 2.95 -> Final Class B (2.50 <= score < 3.25)
    row_101 = full[full["Item Number"] == 101].iloc[0]
    assert row_101["Sales_Class"] == "A"
    assert row_101["Hits_Class"] == "D"
    assert pytest.approx(row_101["Combined_Score"], 0.001) == 2.95
    assert row_101["Final_Class"] == "B"


def test_multicriteria_classification_actual_csv():
    # Test on the actual ProjectDataV2_UTF8.csv
    csv_path = "ProjectDataV2_UTF8.csv"
    assert os.path.exists(csv_path), "ProjectDataV2_UTF8.csv must exist in project directory."
    df = pd.read_csv(csv_path)
    res = inventory_tools.classify_inventory_multicriteria(df)

    assert res["total_items"] == 205
    summary = res["summary_table"]
    assert len(summary) == 4
    assert set(summary["Final Inventory Class"]) == {"A", "B", "C", "D"}
    assert summary["Number of Items"].sum() == 205

    # Test Item Number 1170 lookup
    item_1170 = res["item_1170_lookup"]
    assert item_1170["Item Number"] == 1170
    assert "Silver Edition Buckyballs" in item_1170["Item Name"]
    assert item_1170["Final_Class"] in ["A", "B", "C", "D"]

    # Verify top 5 items per class
    for cls in ["A", "B", "C", "D"]:
        assert len(res["top5_per_class"][cls]) <= 5


# =====================================================================
# 7. AGENT LIFECYCLE & SYNONYM MAPPING TESTS
# =====================================================================

def test_agent_eoq_synonyms():
    import inventory_agent
    agent = inventory_agent.InventoryAnalyticsAgent()
    
    # Query using business synonyms: "cost to place an order" instead of setup cost, "carrying cost" instead of holding cost
    question = "We sell 12000 units a year. Cost to place an order is $50 and our carrying cost is $2 per unit per year. What is the optimal order size?"
    result = agent.process_query(question)
    
    assert result["status"] == "success"
    lc = result["lifecycle"]
    assert lc["step_2_identified_model"] == "eoq"
    assert lc["step_4_selected_tool"] == "inventory_tools.calculate_eoq"
    assert lc["step_6_verification"]["passed"] is True
    
    # Q* = sqrt(2 * 12000 * 50 / 2) = sqrt(600000) ~ 774.596 -> 775
    tool_out = result["tool_output"]
    assert pytest.approx(tool_out["final_result"]["optimal_order_quantity"], 0.1) == 774.60
    assert tool_out["final_result"]["recommended_order_quantity_int"] == 775


def test_agent_safety_stock_synonyms():
    import inventory_agent
    agent = inventory_agent.InventoryAnalyticsAgent()
    
    # Query using "stockout risk" (10%) and "mean daily demand"
    question = "Our mean daily demand is 80 units with std dev of 10. Lead time is 9 days. We want an acceptable risk of stockout of 5%."
    result = agent.process_query(question)
    
    assert result["status"] == "success"
    lc = result["lifecycle"]
    assert lc["step_2_identified_model"] == "safety_stock_rop"
    tool_out = result["tool_output"]
    assert tool_out["final_result"]["safety_stock"] > 0
    assert tool_out["final_result"]["reorder_point"] > tool_out["final_result"]["safety_stock"]


def test_agent_newsvendor():
    import inventory_agent
    agent = inventory_agent.InventoryAnalyticsAgent()
    
    question = "We sell perishable flowers for retail price of $30. Each bunch costs $12 wholesale. Unsold bunches can be salvaged for $4. Mean demand is 200 with standard deviation of 40."
    result = agent.process_query(question)
    
    assert result["status"] == "success"
    lc = result["lifecycle"]
    assert lc["step_2_identified_model"] == "newsvendor"
    tool_out = result["tool_output"]
    assert tool_out["final_result"]["cost_underage"] == 18.0
    assert tool_out["final_result"]["cost_overage"] == 8.0
    # CR = 18 / 26 ~ 0.6923
    assert pytest.approx(tool_out["final_result"]["critical_ratio"], 0.01) == 0.6923


def test_agent_placeholder_and_connection_status():
    import inventory_agent
    # Test placeholder detection
    assert inventory_agent.is_placeholder_key("your-gemini-api-key-here") is True
    assert inventory_agent.is_placeholder_key("") is True
    assert inventory_agent.is_placeholder_key(None) is True
    assert inventory_agent.is_placeholder_key("AIzaSyFakeKey12345") is False

    # Test verification function with placeholder
    verified, msg = inventory_agent.verify_gemini_connection("your-gemini-api-key-here")
    assert verified is False
    assert "not configured" in msg.lower() or "placeholder" in msg.lower()

    # Test fallback parser engine designation
    agent = inventory_agent.InventoryAnalyticsAgent()
    res = agent.process_query("Annual demand is 5000, setup cost is $40, holding cost is $2.")
    assert res["status"] == "success"
    assert res["lifecycle"]["gemini_connection_verified"] is False
    assert "fallback" in res["lifecycle"]["interpretation_engine"].lower()


