import streamlit as st
import pandas as pd
import io
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpBinary, LpStatus
from openpyxl.styles import PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows

DAYS = ["M", "T", "W", "TH", "F", "SAT", "SUN"]
SHIFTS = ["4am-12pm", "11am-7pm", "6pm-2am"]
EMPLOYEE_COL = "Employee"
STORE_PREFERENCE_COL = "Store Preference"  # Column E - soft preference
HARD_PREFERENCE_COL = "Hard Preference"    # Column F - strict constraint

def parse_availability(df, stores):
    availability = {}
    store_preferences = {}
    hard_preferences = {}
    for _, row in df.iterrows():
        emp = row[EMPLOYEE_COL]
        availability[emp] = {}
        preference = row[STORE_PREFERENCE_COL] if STORE_PREFERENCE_COL in df.columns else None
        if pd.notna(preference) and str(preference).strip():
            pref_list = [p.strip() for p in str(preference).split(",") if p.strip()]
            store_preferences[emp] = pref_list if pref_list else None
        else:
            store_preferences[emp] = None
        hard_pref = row[HARD_PREFERENCE_COL] if HARD_PREFERENCE_COL in df.columns else None
        if pd.notna(hard_pref) and str(hard_pref).strip():
            hard_pref_list = [p.strip() for p in str(hard_pref).split(",") if p.strip()]
            hard_preferences[emp] = hard_pref_list if hard_pref_list else None
        else:
            hard_preferences[emp] = None
        for shift in SHIFTS:
            days = str(row[shift]).replace(" ","").split(",") if pd.notna(row[shift]) else []
            for day in days:
                if day:
                    for store in stores:
                        availability[emp][(day, shift, store)] = 1
    return availability, store_preferences, hard_preferences

def schedule(availability, store_preferences, hard_preferences, employees, days, shifts, stores, store_staffing, max_shifts=5):
    prob = LpProblem("StoreShiftScheduling", LpMinimize)
    x = LpVariable.dicts("assign", [(e, d, s, st) for e in employees for d in days for s in shifts for st in stores], 0, 1, LpBinary)
    understaff = LpVariable.dicts("understaff", [(d, s, st) for d in days for s in shifts for st in stores], 0, None, cat='Integer')
    understaffing_penalty = 1000
    preference_penalty = 1
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
    for d in days:
        for s in shifts:
            for st in stores:
                required_staff = store_staffing[st]
                prob += lpSum([x[e, d, s, st] for e in employees]) + understaff[d, s, st] >= required_staff
                prob += lpSum([x[e, d, s, st] for e in employees]) <= required_staff
                if required_staff == 1:
                    prob += lpSum([x[e, d, s, st] for e in employees]) == 1
                else:
                    prob += lpSum([x[e, d, s, st] for e in employees]) >= 1
    for e in employees:
        prob += lpSum([x[e, d, s, st] for d in days for s in shifts for st in stores]) <= max_shifts
    for e in employees:
        for d in days:
            prob += lpSum([x[e, d, s, st] for s in shifts for st in stores]) <= 1
    for e in employees:
        for d in days:
            for s in shifts:
                for st in stores:
                    if (d, s, st) not in availability[e]:
                        prob += x[e, d, s, st] == 0
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
                soft_violations = []
                hard_violations = []
                for emp in assigned:
                    if store_preferences.get(emp) and st not in store_preferences[emp]:
                        pref_str = ", ".join(store_preferences[emp])
                        soft_violations.append(f"{emp} (prefers {pref_str})")
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

def build_user_friendly_schedule(x, employees, days, shifts, stores):
    row_tuples = []
    for store in stores:
        for shift in shifts:
            row_tuples.append((store, shift))
    row_index = pd.MultiIndex.from_tuples(row_tuples, names=["Store", "Shift"])
    columns = days
    schedule_df = pd.DataFrame(index=row_index, columns=columns)
    for store in stores:
        for shift in shifts:
            for day in days:
                assigned = [e for e in employees if x[e, day, shift, store].varValue is not None and abs(x[e, day, shift, store].varValue - 1) < 1e-3]
                schedule_df.loc[(store, shift), day] = ", ".join(assigned) if assigned else ""
    return schedule_df

def build_employee_summary(x, employees, days, shifts, stores):
    """Build a summary of total shifts per employee"""
    employee_shifts = {}
    for emp in employees:
        total_shifts = 0
        for day in days:
            for shift in shifts:
                for store in stores:
                    if x[emp, day, shift, store].varValue is not None and abs(x[emp, day, shift, store].varValue - 1) < 1e-3:
                        total_shifts += 1
        employee_shifts[emp] = total_shifts
    
    summary_df = pd.DataFrame({
        "Employee": list(employee_shifts.keys()),
        "Total Shifts": list(employee_shifts.values())
    })
    return summary_df

def get_availability_template(stores):
    columns = [EMPLOYEE_COL, STORE_PREFERENCE_COL, HARD_PREFERENCE_COL] + SHIFTS
    df = pd.DataFrame(columns=columns)
    return df

def run_scheduler(uploaded_file, store_staffing, max_shifts, stores):
    df = pd.read_excel(uploaded_file)
    if STORE_PREFERENCE_COL not in df.columns:
        df[STORE_PREFERENCE_COL] = None
    if HARD_PREFERENCE_COL not in df.columns:
        df[HARD_PREFERENCE_COL] = None
    availability, store_preferences, hard_preferences = parse_availability(df, stores)
    employees = df[EMPLOYEE_COL].tolist()
    max_shifts_val = max_shifts
    max_shifts_upper_bound = max_shifts_val + 5
    found = False
    while max_shifts_val <= max_shifts_upper_bound:
        x, understaff, status = schedule(availability, store_preferences, hard_preferences, employees, DAYS, SHIFTS, stores, store_staffing, max_shifts_val)
        if status == "Optimal":
            found = True
            break
        else:
            max_shifts_val += 1
    if not found:
        return None, None, None, "No feasible schedule found. Try adjusting preferences or availability."
    schedule_result = build_schedule_output(x, understaff, employees, DAYS, SHIFTS, stores, store_staffing, store_preferences, hard_preferences)
    out_df = pd.DataFrame(schedule_result, columns=["Day", "Shift", "Store", "Employees Assigned", "Total Assigned", "Missing Staff", "Soft Preference Violations", "Hard Preference Violations"])
    user_friendly_df = build_user_friendly_schedule(x, employees, DAYS, SHIFTS, stores)
    employee_summary_df = build_employee_summary(x, employees, DAYS, SHIFTS, stores)
    return out_df, user_friendly_df, employee_summary_df, None

def main():
    st.title("Store Staff Scheduler")
    st.write("Upload your availability spreadsheet to generate a staff schedule.")
    
    # Store Configuration Section
    st.header("Store Configuration")
    st.write("Configure your stores and staffing requirements:")
    
    # Store input
    store_input = st.text_area(
        "Enter store names (one per line):",
        value="AK&CO\nBookstore\nAK Mercantile\nCabin\nMoosetique",
        help="Enter each store name on a separate line"
    )
    
    # Parse stores from input
    stores = [store.strip() for store in store_input.split('\n') if store.strip()]
    
    if not stores:
        st.error("Please enter at least one store name.")
        return
    
    # Staffing configuration
    st.subheader("Staffing Requirements")
    st.write("Set the number of employees needed per shift at each store:")
    store_staffing = {}
    for store in stores:
        store_staffing[store] = st.number_input(
            f"{store} staff needed per shift", 
            min_value=1, 
            max_value=10, 
            value=2 if store in ["AK&CO", "Bookstore", "AK Mercantile"] else 1,  # Default values
            step=1
        )
    
    # Other parameters
    max_shifts = st.number_input("Maximum shifts per employee per week", min_value=1, max_value=21, value=5, step=1)
    
    # Template download
    st.header("Availability Template")
    st.write("Download a template for your employees to fill out their availability:")
    template_df = get_availability_template(stores)
    template_bytes = io.BytesIO()
    template_df.to_excel(template_bytes, index=False)
    template_bytes.seek(0)
    st.download_button(
        label="Download availability template (Excel)",
        data=template_bytes,
        file_name="availability_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # File upload
    st.header("Generate Schedule")
    uploaded_file = st.file_uploader("Upload availability.xlsx", type=["xlsx"])
    
    if uploaded_file is not None:
        with st.spinner("Generating schedule..."):
            schedule_df, user_friendly_df, employee_summary_df, error = run_scheduler(uploaded_file, store_staffing, max_shifts, stores)
        if error:
            st.error(error)
        else:
            st.success("Schedule generated!")
            st.subheader("Detailed Schedule")
            st.dataframe(schedule_df)
            st.subheader("User-Friendly Schedule")
            st.dataframe(user_friendly_df)
            st.subheader("Employee Summary")
            st.dataframe(employee_summary_df)
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
                schedule_df.to_excel(writer, index=False, sheet_name="Detailed Schedule")
                user_friendly_df.to_excel(writer, sheet_name="User-Friendly Schedule")
                employee_summary_df.to_excel(writer, index=False, sheet_name="Employee Summary")
                
                # Apply conditional formatting to Employee Summary sheet
                workbook = writer.book
                worksheet = workbook["Employee Summary"]
                
                # Create red fill pattern for cells with more than 5 shifts
                red_fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
                
                # Apply red fill to cells in column B (Total Shifts) where value > 5
                for row in range(2, len(employee_summary_df) + 2):  # Start from row 2 (skip header)
                    cell = worksheet[f"B{row}"]
                    if cell.value and cell.value > 5:
                        cell.fill = red_fill
            towrite.seek(0)
            st.download_button(
                label="Download schedule.xlsx",
                data=towrite,
                file_name="schedule.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main() 
