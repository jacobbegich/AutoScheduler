import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpBinary, LpStatus

DAYS = ["M", "T", "W", "TH", "F", "SAT", "SUN"]
SHIFTS = ["4am-12pm", "11am-7pm", "6pm-2am"]
STORES = ["AK&CO", "Bookstore", "AK Mercantile", "Cabin", "Moosetique"]
STORE_STAFFING = {"AK&CO": 2, "Bookstore": 2, "AK Mercantile": 2, "Cabin": 1, "Moosetique": 1}
EMPLOYEE_COL = "Employee"
STORE_PREFERENCE_COL = "Store Preference"  # Column E - soft preference
HARD_PREFERENCE_COL = "Hard Preference"    # Column F - strict constraint

def parse_availability(df):
    availability = {}
    store_preferences = {}
    hard_preferences = {}
    
    for _, row in df.iterrows():
        emp = row[EMPLOYEE_COL]
        availability[emp] = {}
        
        # Parse soft store preference (Column E) - can be multiple stores
        preference = row[STORE_PREFERENCE_COL] if STORE_PREFERENCE_COL in df.columns else None
        if pd.notna(preference) and str(preference).strip():
            # Split by comma and clean up each preference
            pref_list = [p.strip() for p in str(preference).split(",") if p.strip()]
            store_preferences[emp] = pref_list if pref_list else None
        else:
            store_preferences[emp] = None
        
        # Parse hard store preference (Column F) - can be multiple stores
        hard_pref = row[HARD_PREFERENCE_COL] if HARD_PREFERENCE_COL in df.columns else None
        if pd.notna(hard_pref) and str(hard_pref).strip():
            # Split by comma and clean up each preference
            hard_pref_list = [p.strip() for p in str(hard_pref).split(",") if p.strip()]
            hard_preferences[emp] = hard_pref_list if hard_pref_list else None
        else:
            hard_preferences[emp] = None
        
        # Parse availability for each shift
        for shift in SHIFTS:
            days = str(row[shift]).replace(" ", "").split(",") if pd.notna(row[shift]) else []
            for day in days:
                if day:
                    for store in STORES:
                        availability[emp][(day, shift, store)] = 1
    
    return availability, store_preferences, hard_preferences

def schedule(availability, store_preferences, hard_preferences, employees, days, shifts, stores, store_staffing, max_shifts=5):
    prob = LpProblem("StoreShiftScheduling", LpMinimize)
    x = LpVariable.dicts("assign", [(e, d, s, st) for e in employees for d in days for s in shifts for st in stores], 0, 1, LpBinary)
    # Understaffing variables (how many staff are missing from the ideal)
    understaff = LpVariable.dicts("understaff", [(d, s, st) for d in days for s in shifts for st in stores], 0, None, cat='Integer')

    # Objective: Minimize total understaffing + soft preference violations (weighted)
    # Make understaffing much more important than preference violations
    understaffing_penalty = 1000  # High penalty for understaffing
    preference_penalty = 1        # Low penalty for preference violations
    
    understaffing_cost = lpSum([understaff[d, s, st] * understaffing_penalty for d in days for s in shifts for st in stores])
    preference_violations = lpSum([
        x[e, d, s, st] * preference_penalty 
        for e in employees 
        for d in days 
        for s in shifts 
        for st in stores 
        if store_preferences.get(e) and st not in store_preferences[e]
    ])
    
    prob += understaffing_cost + preference_violations

    # Each store/shift/day: assigned + understaff >= required staff, and don't overstaff
    for d in days:
        for s in shifts:
            for st in stores:
                required_staff = store_staffing[st]
                prob += lpSum([x[e, d, s, st] for e in employees]) + understaff[d, s, st] >= required_staff
                prob += lpSum([x[e, d, s, st] for e in employees]) <= required_staff
                
                # Ensure minimum staffing: stores with 1 required must have exactly 1, stores with 2 required must have at least 1
                if required_staff == 1:
                    # Single-staff stores must have exactly 1 employee
                    prob += lpSum([x[e, d, s, st] for e in employees]) == 1
                else:
                    # Multi-staff stores must have at least 1 employee (can be understaffed to 1, but not 0)
                    prob += lpSum([x[e, d, s, st] for e in employees]) >= 1

    # Each employee: max shifts per week
    for e in employees:
        prob += lpSum([x[e, d, s, st] for d in days for s in shifts for st in stores]) <= max_shifts

    # Each employee: no more than one shift per day
    for e in employees:
        for d in days:
            prob += lpSum([x[e, d, s, st] for s in shifts for st in stores]) <= 1

    # Only assign if available
    for e in employees:
        for d in days:
            for s in shifts:
                for st in stores:
                    if (d, s, st) not in availability[e]:
                        prob += x[e, d, s, st] == 0

    # Hard preference constraints: employees with hard preferences can ONLY work at their preferred stores
    for e in employees:
        if hard_preferences.get(e):
            preferred_stores = hard_preferences[e]
            for d in days:
                for s in shifts:
                    for st in stores:
                        if st not in preferred_stores:
                            prob += x[e, d, s, st] == 0

    prob.solve()
    return x, understaff, LpStatus[prob.status]

def build_schedule_output(x, understaff, employees, days, shifts, stores, store_staffing, store_preferences, hard_preferences):
    schedule = []
    for d in days:
        for s in shifts:
            for st in stores:
                assigned = [e for e in employees if x[e, d, s, st].varValue is not None and abs(x[e, d, s, st].varValue - 1) < 1e-3]
                num_assigned = len(assigned)
                missing = int(understaff[d, s, st].varValue) if understaff[d, s, st].varValue is not None else 0
                
                # Check for soft preference violations
                soft_violations = []
                hard_violations = []
                for emp in assigned:
                    # Check soft preference violations
                    if store_preferences.get(emp) and st not in store_preferences[emp]:
                        pref_str = ", ".join(store_preferences[emp])
                        soft_violations.append(f"{emp} (prefers {pref_str})")
                    
                    # Check hard preference violations (shouldn't happen if constraints work properly)
                    if hard_preferences.get(emp) and st not in hard_preferences[emp]:
                        hard_pref_str = ", ".join(hard_preferences[emp])
                        hard_violations.append(f"{emp} (HARD: {hard_pref_str})")
                
                soft_flag = "; ".join(soft_violations) if soft_violations else "None"
                hard_flag = "; ".join(hard_violations) if hard_violations else "None"
                
                schedule.append({
                    "Day": d,
                    "Shift": s,
                    "Store": st,
                    "Employees Assigned": ", ".join(assigned),
                    "Total Assigned": num_assigned,
                    "Missing Staff": missing,
                    "Soft Preference Violations": soft_flag,
                    "Hard Preference Violations": hard_flag
                })
    return schedule

def main():
    df = pd.read_excel("availability.xlsx")
    print("Loaded employee availability:")
    print(df)
    
    # Check if preference columns exist
    if STORE_PREFERENCE_COL not in df.columns:
        print(f"Warning: Column '{STORE_PREFERENCE_COL}' not found. Assuming no soft store preferences.")
        df[STORE_PREFERENCE_COL] = None
    
    if HARD_PREFERENCE_COL not in df.columns:
        print(f"Warning: Column '{HARD_PREFERENCE_COL}' not found. Assuming no hard store preferences.")
        df[HARD_PREFERENCE_COL] = None
    
    availability, store_preferences, hard_preferences = parse_availability(df)
    employees = df[EMPLOYEE_COL].tolist()
    
    # Print preferences for debugging
    print("\nSoft Store Preferences:")
    for emp in employees:
        pref = store_preferences.get(emp, "None")
        if pref:
            pref_str = ", ".join(pref)
            print(f"{emp}: {pref_str}")
        else:
            print(f"{emp}: None")
    
    print("\nHard Store Preferences:")
    for emp in employees:
        hard_pref = hard_preferences.get(emp, "None")
        if hard_pref:
            hard_pref_str = ", ".join(hard_pref)
            print(f"{emp}: {hard_pref_str}")
        else:
            print(f"{emp}: None")

    # Try increasing max_shifts per employee
    max_shifts = 5
    max_shifts_upper_bound = 10
    found = False
    while max_shifts <= max_shifts_upper_bound:
        print(f"Trying with max_shifts = {max_shifts} ...")
        x, understaff, status = schedule(availability, store_preferences, hard_preferences, employees, DAYS, SHIFTS, STORES, STORE_STAFFING, max_shifts)
        if status == "Optimal":
            print(f"Feasible schedule found with max_shifts = {max_shifts}.")
            found = True
            break
        else:
            print(f"No feasible schedule with max_shifts = {max_shifts}.")
            max_shifts += 1

    if not found:
        print("No feasible schedule found even after relaxing max_shifts constraint.")
        print("This might be due to hard preference constraints that cannot be satisfied.")
        return

    print(f"Feasible schedule found. (max_shifts = {max_shifts})")
    schedule_result = build_schedule_output(x, understaff, employees, DAYS, SHIFTS, STORES, STORE_STAFFING, store_preferences, hard_preferences)
    out_df = pd.DataFrame(schedule_result, columns=["Day", "Shift", "Store", "Employees Assigned", "Total Assigned", "Missing Staff", "Soft Preference Violations", "Hard Preference Violations"])
    print("Generated Schedule:")
    print(out_df)
    out_df.to_excel("schedule.xlsx", index=False)
    print("Schedule saved to schedule.xlsx")

if __name__ == "__main__":
    main()