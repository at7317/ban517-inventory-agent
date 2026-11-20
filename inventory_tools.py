"""
inventory_tools.py
Deterministic Python Calculation Engine for Inventory Analytics.

Follows the BAN 517 course architecture:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Problem
3. Safety Stock & Reorder Point (Continuous Review)
4. Fixed-Period Review Model (Periodic Review)
5. Newsvendor Model (Single-Period / Perishable)
6. Multicriteria Inventory Stratification (Sales + POS Hits)

Every calculation returns a comprehensive, transparent dictionary:
- model_name
- extracted_inputs
- variable_definitions
- units
- formula
- substituted_values
- intermediate_calculations
- final_result
- assumptions
- managerial_explanation
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


# =====================================================================
# 1. ECONOMIC ORDER QUANTITY (EOQ)
# =====================================================================

def calculate_eoq(
    demand: float,
    setup_cost: float,
    holding_cost: float,
    unit_cost: Optional[float] = None,
    lead_time_days: Optional[float] = None,
    work_days_per_year: float = 365.0,
    demand_time_unit: str = "year",
    holding_cost_time_unit: str = "year",
) -> Dict[str, Any]:
    """
    Calculates deterministic Economic Order Quantity (EOQ).
    
    Parameters:
        demand: Annual demand (D).
        setup_cost: Cost per order / setup cost (S).
        holding_cost: Carrying/holding cost per unit per year (H).
        unit_cost: Purchase price per unit (C), optional.
        lead_time_days: Replenishment lead time in days (L), optional.
        work_days_per_year: Operating work days per year, default 365.
        demand_time_unit: Time horizon for demand, default 'year'.
        holding_cost_time_unit: Time horizon for holding cost, default 'year'.
    """
    # Validation
    if demand <= 0:
        raise ValueError("Demand (D) must be strictly positive.")
    if setup_cost <= 0:
        raise ValueError("Setup/ordering cost (S) must be strictly positive.")
    if holding_cost <= 0:
        raise ValueError("Holding/carrying cost (H) must be strictly positive.")
    if unit_cost is not None and unit_cost < 0:
        raise ValueError("Unit cost (C) cannot be negative.")
    if lead_time_days is not None and lead_time_days < 0:
        raise ValueError("Lead time cannot be negative.")
    if work_days_per_year <= 0:
        raise ValueError("Work days per year must be positive.")

    # Time unit consistency check
    if demand_time_unit.lower() != holding_cost_time_unit.lower():
        raise ValueError(
            f"Demand time unit ('{demand_time_unit}') and holding cost time unit "
            f"('{holding_cost_time_unit}') must be consistent. Please annualize both."
        )

    # Core Calculations
    q_star = math.sqrt((2.0 * demand * setup_cost) / holding_cost)
    orders_per_year = demand / q_star
    annual_setup_cost = (demand / q_star) * setup_cost
    annual_holding_cost = (q_star / 2.0) * holding_cost
    total_inventory_cost = annual_setup_cost + annual_holding_cost
    cycle_time_years = q_star / demand
    cycle_time_days = cycle_time_years * work_days_per_year

    annual_purchase_cost = (demand * unit_cost) if unit_cost is not None else None
    total_cost_with_purchase = (
        total_inventory_cost + annual_purchase_cost
        if annual_purchase_cost is not None
        else None
    )

    daily_demand = demand / work_days_per_year
    rop = (daily_demand * lead_time_days) if lead_time_days is not None else None

    # Variable Definitions
    var_defs = {
        "D": "Annual Demand (units/year)",
        "S": "Setup or Ordering Cost ($/order)",
        "H": "Holding or Carrying Cost ($/unit/year)",
        "Q*": "Optimal Order Quantity (units)",
        "TC": "Total Relevant Annual Inventory Cost ($/year)",
        "N": "Order Frequency (orders/year)",
        "T": "Length of Order Cycle (days between orders)",
    }
    if unit_cost is not None:
        var_defs["C"] = "Unit Purchase Cost ($/unit)"
        var_defs["TC_total"] = "Total Annual Cost including Purchases ($/year)"
    if lead_time_days is not None:
        var_defs["L"] = "Replenishment Lead Time (days)"
        var_defs["ROP"] = "Reorder Point under certainty (units)"

    units = {
        "demand": "units/year",
        "setup_cost": "$/order",
        "holding_cost": "$/unit/year",
        "q_star": "units",
        "orders_per_year": "orders/year",
        "annual_setup_cost": "$/year",
        "annual_holding_cost": "$/year",
        "total_inventory_cost": "$/year",
        "cycle_time_days": "days",
    }
    if unit_cost is not None:
        units["unit_cost"] = "$/unit"
        units["annual_purchase_cost"] = "$/year"
        units["total_cost_with_purchase"] = "$/year"
    if lead_time_days is not None:
        units["lead_time_days"] = "days"
        units["daily_demand"] = "units/day"
        units["reorder_point"] = "units"

    formula_str = (
        r"Q^* = \sqrt{\frac{2 \cdot D \cdot S}{H}}, \quad "
        r"TC = \frac{D}{Q} \cdot S + \frac{Q}{2} \cdot H"
    )
    substituted_str = (
        f"Q* = sqrt((2 * {demand:g} * {setup_cost:g}) / {holding_cost:g}) = {q_star:.2f} units\n"
        f"TC = ({demand:g} / {q_star:.2f}) * {setup_cost:g} + ({q_star:.2f} / 2) * {holding_cost:g} "
        f"= ${annual_setup_cost:.2f} + ${annual_holding_cost:.2f} = ${total_inventory_cost:.2f}"
    )

    intermediate_calcs = {
        "numerator (2*D*S)": round(2.0 * demand * setup_cost, 4),
        "denominator (H)": round(holding_cost, 4),
        "orders_per_year (D/Q*)": round(orders_per_year, 2),
        "annual_setup_cost (D/Q* * S)": round(annual_setup_cost, 2),
        "annual_holding_cost (Q*/2 * H)": round(annual_holding_cost, 2),
        "order_cycle_time_days": round(cycle_time_days, 1),
    }
    if rop is not None:
        intermediate_calcs["daily_demand (D/days)"] = round(daily_demand, 2)
        intermediate_calcs["reorder_point (daily_demand * L)"] = round(rop, 2)
    if annual_purchase_cost is not None:
        intermediate_calcs["annual_purchase_cost (D*C)"] = round(annual_purchase_cost, 2)

    final_result = {
        "optimal_order_quantity": round(q_star, 2),
        "recommended_order_quantity_int": int(round(q_star)),
        "annual_setup_cost": round(annual_setup_cost, 2),
        "annual_holding_cost": round(annual_holding_cost, 2),
        "total_annual_inventory_cost": round(total_inventory_cost, 2),
        "orders_per_year": round(orders_per_year, 2),
        "days_between_orders": round(cycle_time_days, 1),
    }
    if total_cost_with_purchase is not None:
        final_result["total_cost_including_purchases"] = round(total_cost_with_purchase, 2)
    if rop is not None:
        final_result["reorder_point"] = round(rop, 2)
        final_result["reorder_point_int"] = int(math.ceil(rop))

    assumptions = [
        "Demand is known, constant, and continuous over time.",
        "Lead time is known and constant.",
        "Replenishment is instantaneous (entire order arrives in one batch).",
        "No shortages or stockouts are permitted.",
        "Holding cost is strictly linear and per unit per year.",
        "Setup/ordering cost is fixed per order regardless of order size.",
        "Unit purchase price is constant (no quantity discounts in basic EOQ).",
    ]

    managerial_explanation = (
        f"To minimize total inventory costs, the firm should order {int(round(q_star)):,} units "
        f"approximately every {cycle_time_days:.1f} operating days ({orders_per_year:.1f} times per year). "
        f"At this optimal quantity, annual ordering costs (${annual_setup_cost:,.2f}) and annual holding "
        f"costs (${annual_holding_cost:,.2f}) are balanced, yielding a minimum relevant inventory "
        f"cost of ${total_inventory_cost:,.2f}/year."
    )
    if rop is not None:
        managerial_explanation += (
            f" With a lead time of {lead_time_days:g} days, place each replenishment order "
            f"when inventory drops to {int(math.ceil(rop)):,} units."
        )

    return {
        "model_name": "Economic Order Quantity (EOQ)",
        "extracted_inputs": {
            "demand": demand,
            "setup_cost": setup_cost,
            "holding_cost": holding_cost,
            "unit_cost": unit_cost,
            "lead_time_days": lead_time_days,
            "work_days_per_year": work_days_per_year,
            "demand_time_unit": demand_time_unit,
            "holding_cost_time_unit": holding_cost_time_unit,
        },
        "variable_definitions": var_defs,
        "units": units,
        "formula": formula_str,
        "substituted_values": substituted_str,
        "intermediate_calculations": intermediate_calcs,
        "final_result": final_result,
        "assumptions": assumptions,
        "managerial_explanation": managerial_explanation,
    }


# =====================================================================
# 2. QUANTITY DISCOUNT MODEL
# =====================================================================

def calculate_quantity_discount(
    demand: float,
    setup_cost: float,
    discount_schedule: List[Dict[str, Any]],
    holding_cost_fixed: Optional[float] = None,
    holding_cost_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluates quantity discount pricing schedules under either:
    - Case A: Fixed holding cost H ($/unit/year), or
    - Case B: Holding cost as percentage I of purchase price (H_i = I * C_i).
    
    Compares all feasible candidate quantities using Total Annual Cost:
    TC(Q) = D * C_i + (D / Q) * S + (Q / 2) * H_i.
    """
    if demand <= 0:
        raise ValueError("Demand must be strictly positive.")
    if setup_cost <= 0:
        raise ValueError("Setup cost must be strictly positive.")
    if holding_cost_fixed is None and holding_cost_rate is None:
        raise ValueError("Must provide either fixed holding cost or holding cost percentage rate.")
    if holding_cost_fixed is not None and holding_cost_fixed <= 0:
        raise ValueError("Fixed holding cost must be strictly positive.")
    if holding_cost_rate is not None and holding_cost_rate <= 0:
        raise ValueError("Holding cost rate must be strictly positive.")
    if not discount_schedule:
        raise ValueError("Discount schedule cannot be empty.")

    is_percentage = holding_cost_rate is not None
    case_type = "Case B (Holding Cost % of Price)" if is_percentage else "Case A (Fixed Holding Cost)"

    tier_evaluations = []
    for idx, tier in enumerate(discount_schedule):
        min_qty = float(tier.get("min_qty", 0))
        max_qty_raw = tier.get("max_qty")
        max_qty = float(max_qty_raw) if max_qty_raw is not None and not math.isinf(float(max_qty_raw)) else None
        price = float(tier["unit_price"])
        
        if price <= 0:
            raise ValueError(f"Tier {idx + 1} unit price must be positive.")

        # Determine holding cost for tier
        h_tier = (holding_cost_rate * price) if is_percentage else holding_cost_fixed

        # Unconstrained EOQ for this tier
        raw_eoq = math.sqrt((2.0 * demand * setup_cost) / h_tier)

        # Check feasibility and adjust candidate order quantity
        if raw_eoq < min_qty:
            candidate_q = min_qty
            status = f"EOQ ({raw_eoq:.1f}) < min threshold ({min_qty:.0f}); adjusted up to {min_qty:.0f}"
            feasible = True
        elif max_qty is not None and raw_eoq > max_qty:
            # EOQ is beyond the upper boundary of this tier
            candidate_q = None
            status = f"EOQ ({raw_eoq:.1f}) > max threshold ({max_qty:.0f}); infeasible for this tier"
            feasible = False
        else:
            candidate_q = raw_eoq
            status = f"EOQ ({raw_eoq:.1f}) falls within range [{min_qty:.0f}, {max_qty if max_qty else 'inf'})"
            feasible = True

        if feasible and candidate_q is not None:
            purchase_cost = demand * price
            ordering_cost = (demand / candidate_q) * setup_cost
            holding_cost_val = (candidate_q / 2.0) * h_tier
            total_cost = purchase_cost + ordering_cost + holding_cost_val
        else:
            purchase_cost = None
            ordering_cost = None
            holding_cost_val = None
            total_cost = None

        tier_evaluations.append({
            "tier_index": idx + 1,
            "min_qty": min_qty,
            "max_qty": max_qty,
            "unit_price": price,
            "holding_cost_h": round(h_tier, 4),
            "raw_eoq": round(raw_eoq, 2),
            "feasible": feasible,
            "candidate_q": round(candidate_q, 2) if candidate_q else None,
            "annual_purchase_cost": round(purchase_cost, 2) if purchase_cost else None,
            "annual_ordering_cost": round(ordering_cost, 2) if ordering_cost else None,
            "annual_holding_cost": round(holding_cost_val, 2) if holding_cost_val else None,
            "total_annual_cost": round(total_cost, 2) if total_cost else None,
            "status": status,
        })

    # Find optimal tier among feasible candidates
    feasible_tiers = [t for t in tier_evaluations if t["feasible"] and t["total_annual_cost"] is not None]
    if not feasible_tiers:
        raise ValueError("No feasible order quantity found across discount tiers.")

    best_tier = min(feasible_tiers, key=lambda x: x["total_annual_cost"])

    var_defs = {
        "D": "Annual Demand (units/year)",
        "S": "Setup or Ordering Cost ($/order)",
        "C_i": "Unit Purchase Price in Tier i ($/unit)",
        "H_i": "Holding Cost in Tier i ($/unit/year)",
        "Q_candidate": "Feasible Order Quantity for Tier i (units)",
        "TC(Q)": "Total Annual Cost = Purchase Cost + Ordering Cost + Holding Cost",
    }
    units = {
        "demand": "units/year",
        "setup_cost": "$/order",
        "unit_price": "$/unit",
        "holding_cost": "$/unit/year",
        "candidate_q": "units",
        "total_annual_cost": "$/year",
    }
    formula_str = (
        r"TC(Q) = D \cdot C_i + \frac{D}{Q} \cdot S + \frac{Q}{2} \cdot H_i, \quad "
        r"\text{where } H_i = H \text{ (fixed) or } I \cdot C_i \text{ (percentage)}"
    )

    substituted_str = (
        f"Optimal Tier: Tier {best_tier['tier_index']} at Price ${best_tier['unit_price']:.2f}/unit\n"
        f"Order Quantity Q* = {best_tier['candidate_q']} units\n"
        f"TC = ({demand:g} * {best_tier['unit_price']:.2f}) + "
        f"({demand:g} / {best_tier['candidate_q']}) * {setup_cost:g} + "
        f"({best_tier['candidate_q']} / 2) * {best_tier['holding_cost_h']:.4f}\n"
        f"TC = ${best_tier['annual_purchase_cost']:,.2f} + ${best_tier['annual_ordering_cost']:,.2f} + "
        f"${best_tier['annual_holding_cost']:,.2f} = ${best_tier['total_annual_cost']:,.2f}"
    )

    assumptions = [
        "A lower purchase price does not automatically imply the lowest total cost.",
        "Total relevant cost must include purchase cost, ordering cost, and carrying cost.",
        "When an unconstrained EOQ falls below a discount breakpoint, the order quantity is rounded up to the minimum required for the discount.",
        "Holding cost can either be a fixed dollar amount or proportional to the unit price.",
    ]

    managerial_explanation = (
        f"Based on evaluating all discount tiers ({case_type}), the optimal decision is to place orders of "
        f"{best_tier['candidate_q']:,} units in Tier {best_tier['tier_index']} at ${best_tier['unit_price']:.2f}/unit. "
        f"This policy minimizes Total Annual Cost to ${best_tier['total_annual_cost']:,.2f} "
        f"(Purchase: ${best_tier['annual_purchase_cost']:,.2f}, Ordering: ${best_tier['annual_ordering_cost']:,.2f}, "
        f"Holding: ${best_tier['annual_holding_cost']:,.2f}). "
        f"Note: {best_tier['status']}."
    )

    return {
        "model_name": f"Quantity Discount Problem ({case_type})",
        "extracted_inputs": {
            "demand": demand,
            "setup_cost": setup_cost,
            "holding_cost_fixed": holding_cost_fixed,
            "holding_cost_rate": holding_cost_rate,
            "discount_schedule": discount_schedule,
            "case_type": case_type,
        },
        "variable_definitions": var_defs,
        "units": units,
        "formula": formula_str,
        "substituted_values": substituted_str,
        "intermediate_calculations": {"tier_comparisons": tier_evaluations},
        "final_result": {
            "optimal_order_quantity": best_tier["candidate_q"],
            "optimal_tier": best_tier["tier_index"],
            "unit_price": best_tier["unit_price"],
            "annual_purchase_cost": best_tier["annual_purchase_cost"],
            "annual_ordering_cost": best_tier["annual_ordering_cost"],
            "annual_holding_cost": best_tier["annual_holding_cost"],
            "total_annual_cost": best_tier["total_annual_cost"],
        },
        "assumptions": assumptions,
        "managerial_explanation": managerial_explanation,
    }


# =====================================================================
# 3. SAFETY STOCK & REORDER POINT (CONTINUOUS REVIEW)
# =====================================================================

def calculate_safety_stock_rop(
    service_level: Optional[float] = None,
    stockout_risk: Optional[float] = None,
    lead_time_demand_mean: Optional[float] = None,
    lead_time_demand_std: Optional[float] = None,
    daily_demand_mean: Optional[float] = None,
    daily_demand_std: Optional[float] = None,
    lead_time_days: Optional[float] = None,
    lead_time_std_days: Optional[float] = None,
    lead_time_weeks: Optional[float] = None,
    lead_time_std_weeks: Optional[float] = None,
    days_per_week: float = 7.0,
) -> Dict[str, Any]:
    """
    Calculates Safety Stock (SS) and Reorder Point (ROP) under demand and/or lead time uncertainty.
    
    Handles 4 distinct lecture uncertainty cases:
    - Case 1: Lead time demand distribution is given directly (mu_L, sigma_L).
    - Case 2: Demand is variable (d_bar, sigma_d), Lead time (L) is constant.
    - Case 3: Demand is constant (d), Lead time is variable (L_bar, sigma_L).
    - Case 4: Both Demand and Lead time are variable (d_bar, sigma_d, L_bar, sigma_L).
    
    Adheres strictly to the requirement:
    - When converting lead time from weeks to days, convert both mean and standard deviation.
    - State the days-per-week assumption explicitly.
    """
    # Service level resolution
    if service_level is None and stockout_risk is not None:
        service_level = 1.0 - stockout_risk
    if service_level is None:
        raise ValueError("Must provide either service_level (e.g. 0.95) or stockout_risk (e.g. 0.05).")
    if not (0.0 < service_level < 1.0):
        raise ValueError(f"Service level must be between 0 and 1 (exclusive), got {service_level}.")

    conversion_note = None
    # Convert weeks to days if provided
    if lead_time_weeks is not None:
        if days_per_week <= 0:
            raise ValueError("days_per_week must be positive.")
        lead_time_days = lead_time_weeks * days_per_week
        if lead_time_std_weeks is not None:
            lead_time_std_days = lead_time_std_weeks * days_per_week
        conversion_note = (
            f"Explicit assumption: Operating {days_per_week:g} days/week. "
            f"Converted lead time mean = {lead_time_weeks:g} wks -> {lead_time_days:.2f} days"
            + (f", std = {lead_time_std_weeks:g} wks -> {lead_time_std_days:.2f} days." if lead_time_std_weeks is not None else ".")
        )

    z_value = float(stats.norm.ppf(service_level))

    # Identify which case applies
    if lead_time_demand_mean is not None and lead_time_demand_std is not None:
        # Case 1: Direct lead time demand
        case_name = "Case 1: Direct Lead Time Demand Distribution"
        expected_ltd = float(lead_time_demand_mean)
        sigma_ltd = float(lead_time_demand_std)
        formula_ss = r"SS = Z \cdot \sigma_L"
        formula_rop = r"ROP = \mu_L + SS"
        sub_ss = f"SS = {z_value:.4f} * {sigma_ltd:g}"

    elif daily_demand_mean is not None and lead_time_days is not None:
        if daily_demand_std is not None and (lead_time_std_days is None or lead_time_std_days == 0):
            # Case 2: Variable demand, constant lead time
            case_name = "Case 2: Variable Demand, Constant Lead Time"
            expected_ltd = float(daily_demand_mean * lead_time_days)
            sigma_ltd = float(daily_demand_std * math.sqrt(lead_time_days))
            formula_ss = r"SS = Z \cdot \sigma_d \sqrt{L}"
            formula_rop = r"ROP = \bar{d} \cdot L + SS"
            sub_ss = f"SS = {z_value:.4f} * {daily_demand_std:g} * sqrt({lead_time_days:g})"

        elif (daily_demand_std is None or daily_demand_std == 0) and lead_time_std_days is not None and lead_time_std_days > 0:
            # Case 3: Constant demand, variable lead time
            case_name = "Case 3: Constant Demand, Variable Lead Time"
            expected_ltd = float(daily_demand_mean * lead_time_days)
            sigma_ltd = float(daily_demand_mean * lead_time_std_days)
            formula_ss = r"SS = Z \cdot d \cdot \sigma_L"
            formula_rop = r"ROP = d \cdot \bar{L} + SS"
            sub_ss = f"SS = {z_value:.4f} * {daily_demand_mean:g} * {lead_time_std_days:g}"

        elif daily_demand_std is not None and lead_time_std_days is not None and lead_time_std_days > 0:
            # Case 4: Both demand and lead time variable
            case_name = "Case 4: Variable Demand and Variable Lead Time"
            expected_ltd = float(daily_demand_mean * lead_time_days)
            variance_ltd = (lead_time_days * (daily_demand_std ** 2)) + ((daily_demand_mean ** 2) * (lead_time_std_days ** 2))
            sigma_ltd = float(math.sqrt(variance_ltd))
            formula_ss = r"SS = Z \cdot \sqrt{\bar{L} \sigma_d^2 + \bar{d}^2 \sigma_L^2}"
            formula_rop = r"ROP = \bar{d} \cdot \bar{L} + SS"
            sub_ss = (
                f"SS = {z_value:.4f} * sqrt({lead_time_days:g} * ({daily_demand_std:g}^2) + "
                f"({daily_demand_mean:g}^2) * ({lead_time_std_days:g}^2))"
            )
        else:
            # Deterministic lead time demand (no variance)
            case_name = "Deterministic Demand & Lead Time (No Uncertainty)"
            expected_ltd = float(daily_demand_mean * lead_time_days)
            sigma_ltd = 0.0
            formula_ss = r"SS = 0"
            formula_rop = r"ROP = d \cdot L"
            sub_ss = "SS = 0 (no variance specified)"
    else:
        raise ValueError(
            "Insufficient parameters to calculate Safety Stock and ROP. "
            "Provide either (lead_time_demand_mean, lead_time_demand_std) or "
            "(daily_demand_mean, lead_time_days, with optional std devs)."
        )

    safety_stock = z_value * sigma_ltd
    rop = expected_ltd + safety_stock

    var_defs = {
        "SL": f"Cycle Service Level ({service_level * 100:.1f}%)",
        "Z": f"Standard Normal Deviate (Z-score for {service_level * 100:.1f}% service level)",
        "mu_L": "Expected Demand during Lead Time (units)",
        "sigma_L": "Standard Deviation of Demand during Lead Time (units)",
        "SS": "Safety Stock required to buffer uncertainty (units)",
        "ROP": "Reorder Point inventory trigger level (units)",
    }
    units = {
        "service_level": "dimensionless ratio (0 to 1)",
        "z_value": "standard deviations",
        "expected_lead_time_demand": "units",
        "sigma_lead_time_demand": "units",
        "safety_stock": "units",
        "reorder_point": "units",
    }

    formula_combined = f"{formula_ss}, \\quad {formula_rop}"
    substituted_str = (
        f"{case_name}\n"
        f"Z = norm.ppf({service_level:.4f}) = {z_value:.4f}\n"
        f"Expected Lead Time Demand = {expected_ltd:.2f} units\n"
        f"Sigma Lead Time Demand = {sigma_ltd:.2f} units\n"
        f"{sub_ss} = {safety_stock:.2f} units\n"
        f"ROP = {expected_ltd:.2f} + {safety_stock:.2f} = {rop:.2f} units"
    )

    intermediate_calcs = {
        "service_level": round(service_level, 4),
        "stockout_risk": round(1.0 - service_level, 4),
        "z_score": round(z_value, 4),
        "expected_lead_time_demand": round(expected_ltd, 2),
        "sigma_lead_time_demand": round(sigma_ltd, 2),
        "safety_stock_exact": round(safety_stock, 2),
        "rop_exact": round(rop, 2),
    }
    if conversion_note:
        intermediate_calcs["time_unit_conversion_note"] = conversion_note

    assumptions = [
        "Demand during lead time follows a normal distribution.",
        "Daily demands across consecutive days are independent and identically distributed.",
        "Replenishment orders are placed exactly when inventory drops to ROP.",
        "Service level represents cycle service level (probability of not stocking out during a replenishment cycle).",
    ]
    if conversion_note:
        assumptions.append(conversion_note)

    managerial_explanation = (
        f"Under {case_name}, to maintain a {service_level * 100:.1f}% cycle service level "
        f"(accepting at most a {(1.0 - service_level) * 100:.1f}% stockout risk), the firm requires "
        f"a Safety Stock of {math.ceil(safety_stock):,} units. "
        f"A replenishment order should be initiated whenever inventory on hand plus on order drops to "
        f"the Reorder Point of {math.ceil(rop):,} units."
    )

    return {
        "model_name": f"Safety Stock & Reorder Point ({case_name})",
        "extracted_inputs": {
            "service_level": service_level,
            "stockout_risk": 1.0 - service_level,
            "lead_time_demand_mean": lead_time_demand_mean,
            "lead_time_demand_std": lead_time_demand_std,
            "daily_demand_mean": daily_demand_mean,
            "daily_demand_std": daily_demand_std,
            "lead_time_days": lead_time_days,
            "lead_time_std_days": lead_time_std_days,
            "lead_time_weeks": lead_time_weeks,
            "lead_time_std_weeks": lead_time_std_weeks,
            "days_per_week": days_per_week,
        },
        "variable_definitions": var_defs,
        "units": units,
        "formula": formula_combined,
        "substituted_values": substituted_str,
        "intermediate_calculations": intermediate_calcs,
        "final_result": {
            "z_score": round(z_value, 4),
            "safety_stock": round(safety_stock, 2),
            "safety_stock_int": int(math.ceil(safety_stock)),
            "reorder_point": round(rop, 2),
            "reorder_point_int": int(math.ceil(rop)),
        },
        "assumptions": assumptions,
        "managerial_explanation": managerial_explanation,
    }


# =====================================================================
# 4. FIXED-PERIOD REVIEW MODEL (PERIODIC REVIEW)
# =====================================================================

def calculate_fixed_period_review(
    review_period: float,
    lead_time: float,
    daily_demand_mean: float,
    daily_demand_std: float,
    service_level: Optional[float] = None,
    stockout_risk: Optional[float] = None,
    current_on_hand: float = 0.0,
    current_on_order: float = 0.0,
    backorders: float = 0.0,
    time_unit: str = "days",
) -> Dict[str, Any]:
    """
    Calculates Target Inventory Level (M) and Order Quantity (q) for a Fixed-Period Review system.
    
    Adheres strictly to the requirement:
    - Order quantity must be max(0, target inventory minus inventory position).
    - Protection period = Review Period (T) + Lead Time (L).
    """
    if review_period <= 0:
        raise ValueError("Review period (T) must be strictly positive.")
    if lead_time < 0:
        raise ValueError("Lead time (L) cannot be negative.")
    if daily_demand_mean <= 0:
        raise ValueError("Average daily demand must be strictly positive.")
    if daily_demand_std < 0:
        raise ValueError("Demand standard deviation cannot be negative.")

    if service_level is None and stockout_risk is not None:
        service_level = 1.0 - stockout_risk
    if service_level is None:
        raise ValueError("Must provide either service_level (e.g. 0.95) or stockout_risk (e.g. 0.05).")
    if not (0.0 < service_level < 1.0):
        raise ValueError(f"Service level must be between 0 and 1 (exclusive), got {service_level}.")

    # Key distinction: Protection Period includes BOTH review interval and lead time
    protection_period = review_period + lead_time
    z_value = float(stats.norm.ppf(service_level))

    expected_demand_protection = daily_demand_mean * protection_period
    sigma_protection = daily_demand_std * math.sqrt(protection_period)
    safety_stock = z_value * sigma_protection

    # Target Inventory Level (Order-up-to level M)
    target_inventory_m = expected_demand_protection + safety_stock

    # Inventory position = On hand + On order - Backorders
    inventory_position = current_on_hand + current_on_order - backorders

    # Strict rule: Order quantity must be max(0, target inventory - inventory position)
    order_quantity = max(0.0, target_inventory_m - inventory_position)

    var_defs = {
        "T": f"Review Interval / Period ({time_unit})",
        "L": f"Replenishment Lead Time ({time_unit})",
        "T + L": f"Protection Period ({time_unit})",
        "d_bar": f"Average Demand per {time_unit} (units/{time_unit})",
        "sigma_d": f"Standard Deviation of Demand per {time_unit} (units/{time_unit})",
        "Z": f"Normal Deviate for {service_level * 100:.1f}% Service Level",
        "M": "Target Inventory Level / Order-Up-To Level (units)",
        "IP": "Inventory Position = On-Hand + On-Order - Backorders (units)",
        "q": "Replenishment Order Quantity = max(0, M - IP) (units)",
    }
    units = {
        "review_period": time_unit,
        "lead_time": time_unit,
        "protection_period": time_unit,
        "daily_demand_mean": f"units/{time_unit}",
        "daily_demand_std": f"units/{time_unit}",
        "target_inventory_m": "units",
        "inventory_position": "units",
        "order_quantity": "units",
    }
    formula_str = (
        r"M = \bar{d} \cdot (T + L) + Z \cdot \sigma_d \sqrt{T + L}, \quad "
        r"q = \max(0, M - IP)"
    )
    substituted_str = (
        f"Protection Period = {review_period:g} + {lead_time:g} = {protection_period:g} {time_unit}\n"
        f"Z = norm.ppf({service_level:.4f}) = {z_value:.4f}\n"
        f"Expected Protection Demand = {daily_demand_mean:g} * {protection_period:g} = {expected_demand_protection:.2f} units\n"
        f"Sigma Protection = {daily_demand_std:g} * sqrt({protection_period:g}) = {sigma_protection:.2f} units\n"
        f"Safety Stock = {z_value:.4f} * {sigma_protection:.2f} = {safety_stock:.2f} units\n"
        f"Target Inventory (M) = {expected_demand_protection:.2f} + {safety_stock:.2f} = {target_inventory_m:.2f} units\n"
        f"Inventory Position (IP) = {current_on_hand:g} + {current_on_order:g} - {backorders:g} = {inventory_position:.2f} units\n"
        f"Order Quantity q = max(0, {target_inventory_m:.2f} - {inventory_position:.2f}) = {order_quantity:.2f} units"
    )

    intermediate_calcs = {
        "protection_period": protection_period,
        "z_score": round(z_value, 4),
        "expected_demand_over_protection": round(expected_demand_protection, 2),
        "sigma_over_protection": round(sigma_protection, 2),
        "safety_stock": round(safety_stock, 2),
        "target_inventory_m": round(target_inventory_m, 2),
        "inventory_position": round(inventory_position, 2),
        "unconstrained_q": round(target_inventory_m - inventory_position, 2),
        "constrained_q": round(order_quantity, 2),
    }

    assumptions = [
        "Inventory is reviewed only at regular, fixed intervals of T periods.",
        "Protection period must span both the review interval (T) and the supplier lead time (L).",
        "Demand in each period is normally, independently distributed.",
        "Negative replenishment order quantities are impossible; order is bounded below by 0.",
    ]

    managerial_explanation = (
        f"Under a periodic review policy with a review interval of {review_period:g} {time_unit} and lead time "
        f"of {lead_time:g} {time_unit}, the protection window is {protection_period:g} {time_unit}. "
        f"To achieve a {service_level * 100:.1f}% service level, maintain an order-up-to target of "
        f"{int(math.ceil(target_inventory_m)):,} units (including {int(math.ceil(safety_stock)):,} units of safety stock). "
        f"Given the current inventory position of {inventory_position:,.0f} units, place an order for "
        f"{int(math.ceil(order_quantity)):,} units today."
    )

    return {
        "model_name": "Fixed-Period Review Model (Periodic Review)",
        "extracted_inputs": {
            "review_period": review_period,
            "lead_time": lead_time,
            "protection_period": protection_period,
            "daily_demand_mean": daily_demand_mean,
            "daily_demand_std": daily_demand_std,
            "service_level": service_level,
            "current_on_hand": current_on_hand,
            "current_on_order": current_on_order,
            "backorders": backorders,
            "time_unit": time_unit,
        },
        "variable_definitions": var_defs,
        "units": units,
        "formula": formula_str,
        "substituted_values": substituted_str,
        "intermediate_calculations": intermediate_calcs,
        "final_result": {
            "protection_period": protection_period,
            "target_inventory_level": round(target_inventory_m, 2),
            "target_inventory_level_int": int(math.ceil(target_inventory_m)),
            "safety_stock": round(safety_stock, 2),
            "safety_stock_int": int(math.ceil(safety_stock)),
            "order_quantity": round(order_quantity, 2),
            "order_quantity_int": int(math.ceil(order_quantity)),
        },
        "assumptions": assumptions,
        "managerial_explanation": managerial_explanation,
    }


# =====================================================================
# 5. NEWSVENDOR MODEL (PERISHABLE / SINGLE-PERIOD)
# =====================================================================

def calculate_newsvendor(
    selling_price: float,
    unit_cost: float,
    salvage_value: float = 0.0,
    shortage_penalty: float = 0.0,
    demand_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    discrete_demand_data: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """
    Calculates optimal order quantity for single-period perishable inventory (Newsvendor).
    
    Parameters:
        selling_price: Regular unit selling price (P).
        unit_cost: Unit procurement/purchase cost (C).
        salvage_value: Clearance/scrap revenue per unsold unit (V), default 0.
        shortage_penalty: Goodwill loss or penalty per unsatisfied customer (g), default 0.
        demand_mean: Mean demand for normally distributed demand (mu).
        demand_std: Standard deviation for normally distributed demand (sigma).
        discrete_demand_data: Optional list of (demand_value, probability) pairs for discrete cases.
    """
    if selling_price <= 0:
        raise ValueError("Selling price must be strictly positive.")
    if unit_cost <= 0:
        raise ValueError("Unit cost must be strictly positive.")
    if unit_cost >= selling_price:
        raise ValueError("Unit cost must be less than selling price for a profitable product.")
    if salvage_value > unit_cost:
        raise ValueError("Salvage value cannot exceed unit cost.")

    # Underage & Overage costs
    cost_underage = (selling_price - unit_cost) + shortage_penalty
    cost_overage = unit_cost - salvage_value
    critical_ratio = cost_underage / (cost_underage + cost_overage)

    if demand_mean is not None and demand_std is not None:
        if demand_mean <= 0:
            raise ValueError("Demand mean must be positive.")
        if demand_std < 0:
            raise ValueError("Demand standard deviation cannot be negative.")

        z_value = float(stats.norm.ppf(critical_ratio))
        q_star = demand_mean + (z_value * demand_std)
        q_star = max(0.0, q_star)

        # Expected lost sales and leftover units using standard normal loss function
        # E[Lost Sales] = sigma * (norm.pdf(z) - z * (1 - norm.cdf(z)))
        pdf_z = float(stats.norm.pdf(z_value))
        cdf_z = float(stats.norm.cdf(z_value))
        expected_lost_sales = demand_std * (pdf_z - z_value * (1.0 - cdf_z))
        expected_sales = demand_mean - expected_lost_sales
        expected_leftover = q_star - expected_sales

        expected_profit = (
            (selling_price * expected_sales)
            + (salvage_value * expected_leftover)
            - (unit_cost * q_star)
            - (shortage_penalty * expected_lost_sales)
        )

        intermediate_calcs = {
            "cost_underage (Cu = P - C + g)": round(cost_underage, 2),
            "cost_overage (Co = C - V)": round(cost_overage, 2),
            "critical_ratio (Cu / (Cu + Co))": round(critical_ratio, 4),
            "z_score": round(z_value, 4),
            "expected_sales": round(expected_sales, 2),
            "expected_lost_sales": round(expected_lost_sales, 2),
            "expected_leftover": round(expected_leftover, 2),
            "expected_profit": round(expected_profit, 2),
        }
        sub_str = (
            f"Cu = {selling_price:g} - {unit_cost:g} + {shortage_penalty:g} = ${cost_underage:.2f}\n"
            f"Co = {unit_cost:g} - {salvage_value:g} = ${cost_overage:.2f}\n"
            f"CR = {cost_underage:.2f} / ({cost_underage:.2f} + {cost_overage:.2f}) = {critical_ratio:.4f}\n"
            f"Z = norm.ppf({critical_ratio:.4f}) = {z_value:.4f}\n"
            f"Q* = {demand_mean:g} + ({z_value:.4f} * {demand_std:g}) = {q_star:.2f} units\n"
            f"Expected Profit = ${expected_profit:,.2f}"
        )

    elif discrete_demand_data is not None:
        # Sort by demand value
        sorted_pairs = sorted(discrete_demand_data, key=lambda x: x[0])
        cum_prob = 0.0
        q_star = sorted_pairs[-1][0]
        for val, prob in sorted_pairs:
            cum_prob += prob
            if cum_prob >= critical_ratio:
                q_star = val
                break
        intermediate_calcs = {
            "cost_underage (Cu)": round(cost_underage, 2),
            "cost_overage (Co)": round(cost_overage, 2),
            "critical_ratio": round(critical_ratio, 4),
            "discrete_cutoff_cum_prob": round(cum_prob, 4),
        }
        sub_str = (
            f"Cu = ${cost_underage:.2f}, Co = ${cost_overage:.2f}, CR = {critical_ratio:.4f}\n"
            f"Smallest discrete quantity where P(D <= Q) >= {critical_ratio:.4f} is Q* = {q_star}"
        )
    else:
        raise ValueError("Must provide either (demand_mean, demand_std) or discrete_demand_data.")

    var_defs = {
        "P": "Selling Price ($/unit)",
        "C": "Unit Cost ($/unit)",
        "V": "Salvage Value ($/unit)",
        "g": "Shortage Penalty ($/unit)",
        "Cu": "Cost of Underage / Stockout Cost = P - C + g ($/unit)",
        "Co": "Cost of Overage / Excess Cost = C - V ($/unit)",
        "CR": "Critical Ratio / Service Level = Cu / (Cu + Co)",
        "Q*": "Optimal Stocking Quantity (units)",
    }
    units = {
        "selling_price": "$/unit",
        "unit_cost": "$/unit",
        "salvage_value": "$/unit",
        "shortage_penalty": "$/unit",
        "cost_underage": "$/unit",
        "cost_overage": "$/unit",
        "critical_ratio": "probability ratio (0 to 1)",
        "optimal_order_quantity": "units",
    }
    formula_str = (
        r"C_u = P - C + g, \quad C_o = C - V, \quad "
        r"CR = \frac{C_u}{C_u + C_o}, \quad Q^* = F^{-1}(CR)"
    )

    assumptions = [
        "Single selling season with perishable or seasonal demand.",
        "Unsold inventory cannot be carried over to future periods (liquidated at salvage value V).",
        "Demand occurs during a single discrete selling window.",
        "Stockouts result in lost sales and potential goodwill loss.",
    ]

    managerial_explanation = (
        f"With a profit margin of ${cost_underage:.2f}/unit and overstock risk of ${cost_overage:.2f}/unit, "
        f"the Critical Ratio is {critical_ratio * 100:.1f}%. "
        f"The firm should stock {int(round(q_star)):,} units. "
        f"This stocking level optimizes the tradeoff between lost sales margin and inventory markdowns."
    )

    return {
        "model_name": "Newsvendor Model (Single-Period / Perishable Products)",
        "extracted_inputs": {
            "selling_price": selling_price,
            "unit_cost": unit_cost,
            "salvage_value": salvage_value,
            "shortage_penalty": shortage_penalty,
            "demand_mean": demand_mean,
            "demand_std": demand_std,
        },
        "variable_definitions": var_defs,
        "units": units,
        "formula": formula_str,
        "substituted_values": sub_str,
        "intermediate_calculations": intermediate_calcs,
        "final_result": {
            "cost_underage": round(cost_underage, 2),
            "cost_overage": round(cost_overage, 2),
            "critical_ratio": round(critical_ratio, 4),
            "optimal_order_quantity": round(q_star, 2),
            "recommended_order_quantity_int": int(round(q_star)),
        },
        "assumptions": assumptions,
        "managerial_explanation": managerial_explanation,
    }


# =====================================================================
# 6. MULTICRITERIA INVENTORY STRATIFICATION (SALES + POS HITS)
# =====================================================================

def classify_inventory_multicriteria(
    df: pd.DataFrame,
    sales_col: str = "Sales ($)",
    hits_col: str = "Hits",
    item_col: str = "Item Number",
    name_col: str = "Item Name",
    uom_col: str = "UoM",
) -> Dict[str, Any]:
    """
    Executes the exact Multicriteria Inventory Stratification Methodology from
    Inventory_Classification_Methodology.pdf with user clarifications:

    1. Sort Sales and Hits independently in descending order.
       Calculate cumulative percentages including the current item, using full precision:
       A: <= 60%
       B: > 60% and <= 80%
       C: > 80% and <= 90%
       D: > 90%
       Do not round cumulative percentages before assigning classes.
       Use Item Number ascending to break ties.
       Match separately calculated Sales and Hits classes by Item Number.
    2. Convert A=4, B=3, C=2, D=1.
       Combined Score = 0.65 * Sales Score + 0.35 * Hits Score.
    3. Final classes:
       A: score >= 3.25
       B: 2.50 <= score < 3.25
       C: 1.75 <= score < 2.50
       D: score < 1.75
    4. Required outputs:
       - Full classification table
       - Item counts and total Sales by final class
       - Five highest-Sales items within each final class
       - Lookup for Item Number 1170
    """
    # Clean column names in input
    clean_df = df.copy()
    clean_df.columns = [c.strip() for c in clean_df.columns]

    # Verify required columns exist
    for req_col, col_name in [("Sales", sales_col), ("Hits", hits_col), ("Item Number", item_col)]:
        if col_name not in clean_df.columns:
            # Fallback search
            matches = [c for c in clean_df.columns if req_col.lower() in c.lower()]
            if matches:
                clean_df.rename(columns={matches[0]: col_name}, inplace=True)
            else:
                raise ValueError(f"Required column '{col_name}' not found in dataset. Columns: {list(clean_df.columns)}")

    # Clean numeric fields (strip $ and commas)
    if clean_df[sales_col].dtype == object:
        clean_df[sales_col] = clean_df[sales_col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip().astype(float)
    else:
        clean_df[sales_col] = clean_df[sales_col].astype(float)

    if clean_df[hits_col].dtype == object:
        clean_df[hits_col] = clean_df[hits_col].astype(str).str.replace(",", "", regex=False).str.strip().astype(float)
    else:
        clean_df[hits_col] = clean_df[hits_col].astype(float)

    # Convert Item Number to integer/string consistently
    clean_df[item_col] = pd.to_numeric(clean_df[item_col], errors="coerce").fillna(0).astype(int)

    # Helper function for Pareto Class Assignment
    def _assign_pareto(cum_pct: float) -> str:
        if cum_pct <= 60.0:
            return "A"
        elif cum_pct <= 80.0:
            return "B"
        elif cum_pct <= 90.0:
            return "C"
        else:
            return "D"

    # Step 1A: Sort Sales descending, tie-breaker Item Number ascending
    sales_sorted = clean_df[[item_col, sales_col]].sort_values(
        by=[sales_col, item_col], ascending=[False, True]
    ).copy()
    total_sales = sales_sorted[sales_col].sum()
    sales_sorted["Sales_Cum_Pct"] = (sales_sorted[sales_col].cumsum() / total_sales) * 100.0
    sales_sorted["Sales_Class"] = sales_sorted["Sales_Cum_Pct"].apply(_assign_pareto)

    # Step 1B: Sort Hits descending, tie-breaker Item Number ascending
    hits_sorted = clean_df[[item_col, hits_col]].sort_values(
        by=[hits_col, item_col], ascending=[False, True]
    ).copy()
    total_hits = hits_sorted[hits_col].sum()
    hits_sorted["Hits_Cum_Pct"] = (hits_sorted[hits_col].cumsum() / total_hits) * 100.0
    hits_sorted["Hits_Class"] = hits_sorted["Hits_Cum_Pct"].apply(_assign_pareto)

    # Merge by Item Number
    score_map = {"A": 4, "B": 3, "C": 2, "D": 1}
    sales_part = sales_sorted[[item_col, "Sales_Cum_Pct", "Sales_Class"]]
    hits_part = hits_sorted[[item_col, "Hits_Cum_Pct", "Hits_Class"]]

    merged = clean_df.merge(sales_part, on=item_col).merge(hits_part, on=item_col)

    # Step 2: Numerical scores and Combined Weighted Score
    merged["Sales_Score"] = merged["Sales_Class"].map(score_map)
    merged["Hits_Score"] = merged["Hits_Class"].map(score_map)
    merged["Combined_Score"] = 0.65 * merged["Sales_Score"] + 0.35 * merged["Hits_Score"]

    # Step 3: Final Inventory Class
    def _assign_final_class(score: float) -> str:
        if score >= 3.25:
            return "A"
        elif score >= 2.50:
            return "B"
        elif score >= 1.75:
            return "C"
        else:
            return "D"

    merged["Final_Class"] = merged["Combined_Score"].apply(_assign_final_class)

    # Sort final classification table by Final_Class, then Sales descending
    class_order = {"A": 1, "B": 2, "C": 3, "D": 4}
    merged["_sort_rank"] = merged["Final_Class"].map(class_order)
    merged = merged.sort_values(by=["_sort_rank", sales_col], ascending=[True, False]).drop(columns=["_sort_rank"])

    # Required Summary: Count of items and Total Sales by final class
    summary_list = []
    total_items = len(merged)
    for cls in ["A", "B", "C", "D"]:
        subset = merged[merged["Final_Class"] == cls]
        cnt = len(subset)
        cls_sales = subset[sales_col].sum()
        pct_items = (cnt / total_items) * 100.0 if total_items > 0 else 0.0
        pct_sales = (cls_sales / total_sales) * 100.0 if total_sales > 0 else 0.0
        summary_list.append({
            "Final Inventory Class": cls,
            "Number of Items": cnt,
            "% of Total Items": round(pct_items, 2),
            "Total Sales ($)": round(cls_sales, 2),
            "% of Total Sales": round(pct_sales, 2),
        })
    summary_df = pd.DataFrame(summary_list)

    # Five highest-Sales items within each final class
    top5_per_class = {}
    for cls in ["A", "B", "C", "D"]:
        sub = merged[merged["Final_Class"] == cls].sort_values(by=sales_col, ascending=False).head(5)
        cols_to_show = [item_col]
        if name_col in sub.columns:
            cols_to_show.append(name_col)
        cols_to_show.extend([sales_col, hits_col, "Sales_Class", "Hits_Class", "Combined_Score", "Final_Class"])
        top5_per_class[cls] = sub[cols_to_show].to_dict(orient="records")

    # Item 1170 lookup
    item_1170_row = merged[merged[item_col] == 1170]
    if not item_1170_row.empty:
        item_1170_info = item_1170_row.to_dict(orient="records")[0]
    else:
        item_1170_info = {"Item Number": 1170, "Status": "Item 1170 not found in dataset."}

    return {
        "full_table": merged,
        "summary_table": summary_df,
        "top5_per_class": top5_per_class,
        "item_1170_lookup": item_1170_info,
        "total_items": total_items,
        "total_sales": total_sales,
        "total_hits": total_hits,
    }
