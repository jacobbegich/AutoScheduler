import streamlit as st
import pandas as pd
import io
from scheduler import parse_availability, schedule, build_schedule_output, DAYS, SHIFTS, STORES, STORE_STAFFING, EMPLOYEE_COL, STORE_PREFERENCE_COL, HARD_PREFERENCE_COL
import inspect

def run_scheduler(uploaded_file, store_staffing, max_shifts):
    df = pd.read_excel(uploaded_file)
    # Ensure preference columns exist
    if STORE_PREFERENCE_COL not in df.columns:
        df[STORE_PREFERENCE_COL] = None
    if HARD_PREFERENCE_COL not in df.columns:
        df[HARD_PREFERENCE_COL] = None
    availability, store_preferences, hard_preferences = parse_availability(df)
    employees = df[EMPLOYEE_COL].tolist()
    max_shifts_val = max_shifts
    max_shifts_upper_bound = max_shifts_val + 5
    found = False
    while max_shifts_val <= max_shifts_upper_bound:
        x, understaff, status = schedule(availability, store_preferences, hard_preferences, employees, DAYS, SHIFTS, STORES, store_staffing, max_shifts_val)
        if status == "Optimal":
            found = True
            break
        else:
            max_shifts_val += 1
    if not found:
        return None, "No feasible schedule found. Try adjusting preferences or availability."
    schedule_result = build_schedule_output(x, understaff, employees, DAYS, SHIFTS, STORES, store_staffing, store_preferences, hard_preferences)
    out_df = pd.DataFrame(schedule_result, columns=["Day", "Shift", "Store", "Employees Assigned", "Total Assigned", "Missing Staff", "Soft Preference Violations", "Hard Preference Violations"])
    return out_df, None

def get_availability_template():
    # Columns: Employee, Store Preference, Hard Preference, then one for each shift
    columns = [EMPLOYEE_COL, STORE_PREFERENCE_COL, HARD_PREFERENCE_COL] + SHIFTS
    df = pd.DataFrame(columns=columns)
    return df

def main():
    st.title("Store Staff Scheduler")
    st.write("Upload your availability spreadsheet to generate a staff schedule.")

    # Download template button
    template_df = get_availability_template()
    template_bytes = io.BytesIO()
    template_df.to_excel(template_bytes, index=False)
    template_bytes.seek(0)
    st.download_button(
        label="Download availability template (Excel)",
        data=template_bytes,
        file_name="availability_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.header("Staffing Parameters")
    st.write("Set the number of employees needed per shift at each store:")
    store_staffing = {}
    for store in STORES:
        store_staffing[store] = st.number_input(f"{store} staff needed per shift", min_value=1, max_value=10, value=STORE_STAFFING[store], step=1)

    max_shifts = st.number_input("Maximum shifts per employee per week", min_value=1, max_value=21, value=5, step=1)

    uploaded_file = st.file_uploader("Upload availability.xlsx", type=["xlsx"])
    if uploaded_file is not None:
        with st.spinner("Generating schedule..."):
            schedule_df, error = run_scheduler(uploaded_file, store_staffing, max_shifts)
        if error:
            st.error(error)
        else:
            st.success("Schedule generated!")
            st.dataframe(schedule_df)
            # Download button
            towrite = io.BytesIO()
            schedule_df.to_excel(towrite, index=False)
            towrite.seek(0)
            st.download_button(
                label="Download schedule.xlsx",
                data=towrite,
                file_name="schedule.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main() 