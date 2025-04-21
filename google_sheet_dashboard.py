# Full Dashboard Code with Season Filter in YOY Trends
# and Quarter Filter REMOVED from YOY Trends

import streamlit as st
import pandas as pd
import warnings
import plotly.express as px
import datetime
import calendar
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import re
import io
import gspread
from google.oauth2.service_account import Credentials
import os
import traceback # Added for better error logging

# Filter warnings for a clean output
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(page_title="YOY Dashboard", page_icon=":chart_with_upwards_trend:", layout="wide")

# --- Google Sheet Configuration ---
# <<< EDIT HERE >>>
# Replace with your actual Google Sheet file name
GOOGLE_SHEET_NAME = "RA_sales_dashboard_data"
# Replace with the exact name of the worksheet (tab) within your Google Sheet
WORKSHEET_NAME = "SAP_DATA"
# Path to your service account key file for local development.
# Change if the file is not in the same directory as the script.
LOCAL_KEY_PATH = 'service_account.json'
# <<< END EDIT HERE >>>

# --- Define Google API Scopes ---
# These define the permissions the script requests
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive.file' # Often needed for discovery/listing
]

# --- Title and Logo ---
col1_title, col2_logo = st.columns([3, 1])
with col1_title:
    st.title("YOY Dashboard 📊")
with col2_logo:
    # Check if logo file exists before trying to display it
    if os.path.exists("logo.png"):
        st.image("logo.png", width=300)
    else:
        st.write(" ") # Placeholder if logo not found

# =============================================================================
# Function to Load Data from Google Sheets (Corrected Error Handling)
# =============================================================================
@st.cache_data(ttl=600, show_spinner="Fetching data from Google Sheet...") # Cache data for 10 minutes
def load_data_from_gsheet():
    """Loads data from the specified Google Sheet worksheet, handling missing secrets file."""
    creds = None
    secrets_used = False # Flag to check if secrets were successfully used

    # --- Authentication ---
    # Try using Streamlit secrets first
    try:
        # Check if the specific key exists in secrets IF the secrets file itself can be accessed
        if hasattr(st, 'secrets') and "gcp_service_account" in st.secrets:
            creds_json_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(creds_json_dict, scopes=SCOPES)
            secrets_used = True
            # st.info("Using credentials from Streamlit secrets.") # Optional feedback
        # else: # This would run if secrets file exists but key is missing
            # st.info("Secrets file found, but 'gcp_service_account' key missing. Trying local file.")
            # pass # Proceed to local file check below

    except FileNotFoundError:
        # This specifically catches the error if secrets.toml doesn't exist
        #st.info("No Streamlit secrets file found. Trying local key file...")
        pass # Proceed to local file check below
    except Exception as e_secrets:
        # Catch other potential errors during secrets processing
        # st.warning(f"Error processing Streamlit secrets: {e_secrets}. Trying local key file.")
        pass # Proceed to local file check below


    # Fallback to local JSON file if secrets weren't successfully used
    if not secrets_used:
        if os.path.exists(LOCAL_KEY_PATH):
            try:
                creds = Credentials.from_service_account_file(LOCAL_KEY_PATH, scopes=SCOPES)
                #st.info(f"Using credentials from local file: '{LOCAL_KEY_PATH}'.") # Feedback
            except Exception as e_local:
                # Handle errors loading from the local file specifically
                st.error(f"Error loading credentials from local file '{LOCAL_KEY_PATH}': {e_local}")
                st.stop() # Stop if local file exists but can't be read
        else:
            # This error occurs if secrets failed AND local file doesn't exist
            st.error(f"Authentication Error: Neither Streamlit Secrets nor local key file '{LOCAL_KEY_PATH}' were found or configured correctly.")
            st.stop() # Stop execution if no credentials found

    # Final check if credentials object was successfully created by either method
    if not creds:
        st.error("Authentication failed. Could not load credentials object.")
        st.stop()

    # --- Authorize and Open Sheet (Rest of the function remains the same) ---
    try:
        client = gspread.authorize(creds)
        try:
            # Using the name first, fallback to key if name fails or key is preferred
            # spreadsheet = client.open(GOOGLE_SHEET_NAME)
            # Or use the specific key if needed:
            spreadsheet = client.open_by_key('1p9MkE7pSF5WiZjPJCFrGwKAHQf3aZglzUKjMHMDprlo') # Using Key from original code
            # st.success(f"Successfully opened Google Sheet: '{spreadsheet.title}'") # Use spreadsheet.title
        except gspread.exceptions.SpreadsheetNotFound:
            st.error(f"Error: Google Sheet '{GOOGLE_SHEET_NAME}' not found. Please check the name/key and ensure it's shared with the service account email: {creds.service_account_email}")
            st.stop()
        except gspread.exceptions.APIError as api_error:
             st.error(f"Google Sheets API Error while opening spreadsheet: {api_error}")
             st.info("Check API permissions and sharing settings.")
             st.stop()


        try:
            worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
        except gspread.exceptions.WorksheetNotFound:
            st.error(f"Error: Worksheet '{WORKSHEET_NAME}' not found in the spreadsheet '{spreadsheet.title}'. Please check the worksheet name (tab name).")
            st.stop()

        #st.success(f"Connected to Google Sheet: '{spreadsheet.title}', Worksheet: '{WORKSHEET_NAME}'")

        # Read Data into Pandas DataFrame
        data = worksheet.get_all_values()
        if not data or len(data) < 2:
            st.warning(f"No data found in worksheet '{WORKSHEET_NAME}' or only headers present.")
            return pd.DataFrame()

        headers = data.pop(0)
        df = pd.DataFrame(data, columns=headers)
        #st.info(f"Read {len(df)} rows from Google Sheet (before type conversion).")

        # --- Data Type Conversion section ---
        numeric_cols = [
            "Revenue", "Week", "Order Quantity", "Sales Value in Transaction Currency",
            "Sales Value (£)", "Year"
            # Add/Remove based on your actual columns
        ]
        date_cols = ["Date"] # Assuming "Date" is the primary date column

        for col in numeric_cols:
            if col in df.columns:
                # Convert to string first to handle various formats before cleaning
                df[col] = df[col].astype(str).str.replace(r'[£,]', '', regex=True).str.strip()
                # Replace empty strings resulting from cleaning with NA before numeric conversion
                df[col] = df[col].replace('', pd.NA)
                # Convert to numeric, coercing errors (invalid formats become NaN)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # else: st.warning(f"Numeric column '{col}' specified for conversion not found.")

        for col in date_cols:
            if col in df.columns:
                 # Replace empty strings with NaT before datetime conversion
                df[col] = df[col].replace('', pd.NaT)
                # Convert to datetime, coercing errors (invalid formats become NaT)
                # Inferring format can be slow, specify if known e.g., format='%Y-%m-%d'
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            # else: st.warning(f"Date column '{col}' specified for conversion not found.")

        # Convert any remaining empty strings in the DataFrame to None (or NaN/NaT)
        df = df.replace('', None)
        #st.success("Data successfully loaded and initial types converted from Google Sheet.")
        return df

    # Catch specific gspread/auth errors or general exceptions during sheet access/reading
    except gspread.exceptions.APIError as e_api:
        st.error(f"Google Sheets API Error after authentication: {e_api}")
        #st.info("This might be due to insufficient permissions for the service account on the Sheet (needs 'Viewer') or API not enabled correctly.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while accessing Google Sheets after authentication: {e}")
        st.error(traceback.format_exc())
        st.stop()

# =============================================================================
# Helper Functions for Sales Data
# =============================================================================

def compute_custom_week(date):
    """Computes the custom week number (Sat-Fri), year, start, and end dates."""
    # Ensure input is a date object
    if not isinstance(date, datetime.date):
        # Handle cases where input might be NaT or other non-date types
        return None, None, None, None
    try:
        custom_dow = (date.weekday() + 2) % 7  # Saturday=0, Sunday=1, ..., Friday=6
        week_start = date - datetime.timedelta(days=custom_dow)
        week_end = week_start + datetime.timedelta(days=6)
        custom_year = week_end.year # Year is defined by the week's end date

        # Calculate the start of the first week of the custom_year
        first_day_of_year = datetime.date(custom_year, 1, 1)
        first_day_of_year_custom_dow = (first_day_of_year.weekday() + 2) % 7
        first_week_start_of_year = first_day_of_year - datetime.timedelta(days=first_day_of_year_custom_dow)

        # Handle edge case where the week_start might fall in the *previous* calendar year
        # but belongs to the first week of the custom_year if first_week_start_of_year is also in the previous year.
        if week_start < first_week_start_of_year:
                # This week likely belongs to the previous year's week numbering
                custom_year -= 1
                first_day_of_year = datetime.date(custom_year, 1, 1)
                first_day_of_year_custom_dow = (first_day_of_year.weekday() + 2) % 7
                first_week_start_of_year = first_day_of_year - datetime.timedelta(days=first_day_of_year_custom_dow)


        # Calculate the custom week number
        custom_week = ((week_start - first_week_start_of_year).days // 7) + 1

        # Sanity check for week 53 potentially belonging to the next year if Jan 1st starts mid-week
        # This logic might need refinement depending on specific ISO week definitions if required
        if custom_week == 53:
            last_day_of_year = datetime.date(custom_year, 12, 31)
            last_day_custom_dow = (last_day_of_year.weekday() + 2) % 7
            if last_day_custom_dow < 3: # If year ends on Sat, Sun, Mon (custom dow 0, 1, 2) - week 53 might belong to next year
                # Check if Jan 1st of next year falls within this week
                next_year_first_day = datetime.date(custom_year + 1, 1, 1)
                if week_start <= next_year_first_day <= week_end:
                        # It's arguably week 1 of the next year
                        # This specific handling depends on exact business rules for week 53/1 transition
                        # For simplicity here, we keep it as week 53 of custom_year
                        pass # Or potentially adjust to year+1, week 1

        return custom_week, custom_year, week_start, week_end
    except Exception as e:
        # Log or handle error appropriately
        st.error(f"Error computing custom week for date {date}: {e}")
        return None, None, None, None


def get_custom_week_date_range(week_year, week_number):
    """Gets the start and end date for a given custom week year and number."""
    try:
        week_year = int(week_year)
        week_number = int(week_number)
        # Calculate the start of the first week of the week_year
        first_day = datetime.date(week_year, 1, 1)
        first_day_custom_dow = (first_day.weekday() + 2) % 7
        first_week_start = first_day - datetime.timedelta(days=first_day_custom_dow)
        # Calculate the start date of the requested week
        week_start = first_week_start + datetime.timedelta(weeks=week_number - 1)
        week_end = week_start + datetime.timedelta(days=6)
        return week_start, week_end
    except (ValueError, TypeError) as e:
        # Handle cases where year/week number are invalid
        # st.warning(f"Invalid input for get_custom_week_date_range: Year={week_year}, Week={week_number}. Error: {e}")
        return None, None


def get_quarter(week):
    """Determines the quarter based on the custom week number."""
    if pd.isna(week): return None
    try:
        week = int(week)
        if 1 <= week <= 13:
            return "Q1"
        elif 14 <= week <= 26:
            return "Q2"
        elif 27 <= week <= 39:
            return "Q3"
        elif 40 <= week <= 53: # Assuming up to 53 weeks possible
            return "Q4"
        else:
            return None
    except (ValueError, TypeError):
        return None # Handle non-integer weeks


def format_currency(value):
    """Formats a numeric value as currency."""
    if pd.isna(value): return "£N/A" # Handle NaN values
    try:
        return f"£{float(value):,.2f}"
    except (ValueError, TypeError):
        return "£Error" # Handle non-numeric input


def format_currency_int(value):
    """Formats a numeric value as integer currency."""
    if pd.isna(value): return "£N/A" # Handle NaN values
    try:
        # Round before converting to int to handle decimals properly
        return f"£{int(round(float(value))):,}"
    except (ValueError, TypeError):
        return "£Error" # Handle non-numeric input


@st.cache_data(show_spinner="Preprocessing data...")
def preprocess_data(data):
    """Preprocesses the loaded data: converts types, calculates custom weeks/quarters."""
    df = data.copy() # Work on a copy to avoid modifying cached data
    #st.info("Starting data preprocessing...")

    # --- Initial Data Validation ---
    # Check for essential columns needed *before* processing
    required_input_cols = {"Date", "Sales Value (£)"} # Adjust if other raw cols are essential
    if not required_input_cols.issubset(set(df.columns)):
        missing_cols = required_input_cols.difference(set(df.columns))
        st.error(f"Input data is missing essential columns required for preprocessing: {missing_cols}")
        st.stop()

    # --- Type Conversion and Cleaning (Safeguard) ---
    # Ensure 'Date' is datetime (should be handled by load_data, but double-check)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        initial_rows = len(df)
        df.dropna(subset=["Date"], inplace=True)
        if len(df) < initial_rows:
            st.warning(f"Removed {initial_rows - len(df)} rows during preprocessing due to invalid 'Date' values.")
    else:
        st.error("Essential column 'Date' not found during preprocessing.")
        st.stop()

    # Ensure 'Sales Value (£)' is numeric (should be handled by load_data)
    if "Sales Value (£)" in df.columns:
        df["Sales Value (£)"] = pd.to_numeric(df["Sales Value (£)"], errors='coerce')
        initial_rows_sales = len(df)
        df.dropna(subset=["Sales Value (£)"], inplace=True)
        if len(df) < initial_rows_sales:
            st.warning(f"Removed {initial_rows_sales - len(df)} rows during preprocessing due to invalid 'Sales Value (£)' values.")
    else:
        st.error("Essential column 'Sales Value (£)' not found during preprocessing.")
        st.stop()

    if df.empty:
        st.error("No valid data remaining after initial cleaning during preprocessing.")
        st.stop()

    # --- Feature Engineering ---
    # Calculate custom week details
    try:
        # Ensure we are applying to the .date part of datetime objects
        week_results = df["Date"].apply(lambda d: compute_custom_week(d.date()) if pd.notnull(d) else (None, None, None, None))
        # Unpack results into new columns
        df[["Custom_Week", "Custom_Week_Year", "Custom_Week_Start", "Custom_Week_End"]] = pd.DataFrame(week_results.tolist(), index=df.index)

    except Exception as e:
        st.error(f"Error calculating custom week details during preprocessing: {e}")
        st.error(traceback.format_exc())
        st.stop()

    # Assign 'Week' and 'Quarter' based on calculated custom week
    df["Week"] = df["Custom_Week"] # Assign Custom_Week to Week
    df["Quarter"] = df["Week"].apply(get_quarter)

    # Convert calculated columns to appropriate integer types (allowing NAs)
    # Use Int64 to handle potential NaNs gracefully
    df["Week"] = pd.to_numeric(df["Week"], errors='coerce').astype('Int64')
    df["Custom_Week_Year"] = pd.to_numeric(df["Custom_Week_Year"], errors='coerce').astype('Int64')
    df["Custom_Week"] = pd.to_numeric(df["Custom_Week"], errors='coerce').astype('Int64') # Also convert Custom_Week

    # Convert week start/end dates back to datetime if needed for consistency
    df["Custom_Week_Start"] = pd.to_datetime(df["Custom_Week_Start"], errors='coerce')
    df["Custom_Week_End"] = pd.to_datetime(df["Custom_Week_End"], errors='coerce')


    # --- Final Validation ---
    # Drop rows where crucial calculated fields are missing
    initial_rows_final = len(df)
    # Keep Quarter for potential use in other tabs, even if removed from YOY filter
    df.dropna(subset=["Week", "Custom_Week_Year", "Quarter"], inplace=True)
    if len(df) < initial_rows_final:
            st.warning(f"Removed {initial_rows_final - len(df)} rows during preprocessing due to missing calculated week/year/quarter.")

    # Check if essential columns for the dashboard exist AFTER processing
    required_output_cols = {"Week", "Custom_Week_Year", "Sales Value (£)", "Date", "Quarter"} # Add others like Listing, Product, etc. if needed downstream
    if not required_output_cols.issubset(set(df.columns)):
            missing_cols = required_output_cols.difference(set(df.columns))
            st.error(f"Preprocessing did not produce all required output columns: {missing_cols}")
            # Potentially stop execution or return None/empty df depending on desired behavior
            st.stop()

    if df.empty:
        st.error("No data remaining after full preprocessing.")
        st.stop()

    #st.success(f"Preprocessing complete. {len(df)} rows ready for analysis.")
    return df


# =============================================================================
# Charting and Table Functions
# =============================================================================

# <<< MODIFIED: Removed selected_quarters parameter and filtering logic >>>
def create_yoy_trends_chart(data, selected_years,
                            selected_channels=None, selected_listings=None,
                            selected_products=None, time_grouping="Week",
                            week_range=None, selected_season=None):
    """Creates the YOY Trends line chart, incorporating week range and season filter."""
    filtered = data.copy()
    # Apply filters
    if selected_years:
        filtered = filtered[filtered["Custom_Week_Year"].isin([int(y) for y in selected_years])]

    # <<< REMOVED: Quarter filtering block >>>
    # if selected_quarters:
    #     filtered = filtered[filtered["Quarter"].isin(selected_quarters)]

    # Season Filter Logic
    if selected_season and selected_season != "ALL":
        if "Season" in filtered.columns:
            filtered = filtered[filtered["Season"] == selected_season]
        else:
            st.warning("Column 'Season' not found for filtering.")

    # Other existing filters...
    if selected_channels and len(selected_channels) > 0:
        if "Sales Channel" in filtered.columns:
            filtered = filtered[filtered["Sales Channel"].isin(selected_channels)]
        else: st.warning("Column 'Sales Channel' not found for filtering.")
    if selected_listings and len(selected_listings) > 0:
        if "Listing" in filtered.columns:
            filtered = filtered[filtered["Listing"].isin(selected_listings)]
        else: st.warning("Column 'Listing' not found for filtering.")
    if selected_products and len(selected_products) > 0:
        if "Product" in filtered.columns:
            filtered = filtered[filtered["Product"].isin(selected_products)]
        else: st.warning("Column 'Product' not found for filtering.")

    if week_range:
        start_week, end_week = week_range
        if "Week" in filtered.columns:
            filtered["Week"] = pd.to_numeric(filtered["Week"], errors='coerce').astype('Int64')
            filtered.dropna(subset=["Week"], inplace=True)
            if not filtered.empty:
               filtered = filtered[(filtered["Week"] >= start_week) & (filtered["Week"] <= end_week)]
        else:
            st.warning("Column 'Week' not found for week range filtering in YOY chart.")

    if filtered.empty:
        st.warning("No data available for YOY Trends chart with selected filters.")
        return go.Figure()

    # Group data based on time_grouping
    # NOTE: The time_grouping logic still exists, but the Quarter filter widget is gone from YOY tab.
    # In the YOY tab, time_grouping is hardcoded to "Week".
    if time_grouping == "Week":
        if "Week" not in filtered.columns:
                st.error("Critical Error: 'Week' column lost during filtering for YOY chart grouping.")
                return go.Figure()
        grouped = filtered.groupby(["Custom_Week_Year", "Week"])["Sales Value (£)"].sum().reset_index()
        x_col = "Week"
        x_axis_label = "Week"
        if "Custom_Week_Year" in grouped.columns and "Week" in grouped.columns:
                grouped = grouped.sort_values(by=["Custom_Week_Year", "Week"])
        title = "Weekly Revenue Trends by Custom Week Year"
    # This 'else' block for Quarter grouping is now less relevant for the YOY tab
    # as the filter is removed and time_grouping is fixed to Week there.
    # Keep it if the function might be reused elsewhere with time_grouping="Quarter".
    else: # Assume Quarter
        if "Quarter" not in filtered.columns:
                st.error("Critical Error: 'Quarter' column lost during filtering for YOY chart grouping.")
                return go.Figure()
        grouped = filtered.groupby(["Custom_Week_Year", "Quarter"])["Sales Value (£)"].sum().reset_index()
        x_col = "Quarter"
        x_axis_label = "Quarter"
        quarter_order = ["Q1", "Q2", "Q3", "Q4"]
        grouped["Quarter"] = pd.Categorical(grouped["Quarter"], categories=quarter_order, ordered=True)
        if "Custom_Week_Year" in grouped.columns and "Quarter" in grouped.columns:
            grouped = grouped.sort_values(by=["Custom_Week_Year", "Quarter"])
        title = "Quarterly Revenue Trends by Custom Week Year"

    if grouped.empty:
        st.warning("No data available after grouping for YOY Trends chart.")
        return go.Figure()

    grouped["RevenueK"] = grouped["Sales Value (£)"] / 1000
    fig = px.line(grouped, x=x_col, y="Sales Value (£)", color="Custom_Week_Year", markers=True,
                    title=title,
                    labels={"Sales Value (£)": "Revenue (£)", x_col: x_axis_label, "Custom_Week_Year": "Year"},
                    custom_data=["RevenueK"])
    fig.update_traces(hovertemplate=f"{x_axis_label}: %{{x}}<br>Revenue: %{{customdata[0]:.1f}}K<extra></extra>")

    if time_grouping == "Week":
        if not grouped.empty and "Week" in grouped.columns:
            min_week_data = grouped["Week"].min()
            max_week_data = grouped["Week"].max()
            if pd.isna(min_week_data): min_week_data = 1
            if pd.isna(max_week_data): max_week_data = 52
            min_week_data = int(min_week_data)
            max_week_data = int(max_week_data)
        else:
            min_week_data = 1
            max_week_data = 52

        min_week_plot = week_range[0] if week_range else min_week_data
        max_week_plot = week_range[1] if week_range else max_week_data

        fig.update_xaxes(range=[max(0.8, min_week_plot - 0.2), max_week_plot + 0.2], dtick=5)

    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(margin=dict(t=50, b=50), legend_title_text='Year')
    return fig


def create_pivot_table(data, selected_years, selected_quarters, selected_channels,
                       selected_listings, selected_products, grouping_key="Listing"):
    """Creates the pivot table."""
    filtered = data.copy()
    # Apply filters
    if selected_years:
        filtered = filtered[filtered["Custom_Week_Year"].isin([int(y) for y in selected_years])]
    # Keep Quarter filter here as this function might be used elsewhere or
    # the Pivot Table tab itself still uses the Quarter filter.
    if selected_quarters:
        if "Quarter" in filtered.columns:
             filtered = filtered[filtered["Quarter"].isin(selected_quarters)]
        else: st.warning("Column 'Quarter' not found for filtering pivot table.")
    if selected_channels and len(selected_channels) > 0:
        if "Sales Channel" in filtered.columns:
            filtered = filtered[filtered["Sales Channel"].isin(selected_channels)]
        else: st.warning("Column 'Sales Channel' not found for filtering pivot table.")
    if selected_listings and len(selected_listings) > 0:
        if "Listing" in filtered.columns:
            filtered = filtered[filtered["Listing"].isin(selected_listings)]
        else: st.warning("Column 'Listing' not found for filtering pivot table.")
    if grouping_key == "Product" and selected_products and len(selected_products) > 0:
        if "Product" in filtered.columns:
            filtered = filtered[filtered["Product"].isin(selected_products)]
        else: st.warning("Column 'Product' not found for filtering pivot table.")

    if filtered.empty:
        st.warning("No data available for Pivot Table with selected filters.")
        return pd.DataFrame({grouping_key: ["No data"]})

    if grouping_key not in filtered.columns:
         st.error(f"Required grouping column ('{grouping_key}') not found for creating pivot table.")
         return pd.DataFrame({grouping_key: ["Missing grouping column"]})
    if "Week" not in filtered.columns:
         st.error("Required column ('Week') not found for creating pivot table.")
         return pd.DataFrame({grouping_key: ["Missing 'Week' column"]})

    filtered["Sales Value (£)"] = pd.to_numeric(filtered["Sales Value (£)"], errors='coerce')
    filtered.dropna(subset=["Sales Value (£)", "Week", grouping_key], inplace=True)

    if filtered.empty:
         st.warning("No valid data left for Pivot Table after cleaning.")
         return pd.DataFrame({grouping_key: ["No valid data"]})

    pivot = pd.pivot_table(filtered, values="Sales Value (£)", index=grouping_key,
                           columns="Week", aggfunc="sum", fill_value=0)

    if pivot.empty:
        st.warning("Pivot table is empty after grouping.")
        return pd.DataFrame({grouping_key: ["No results"]})

    pivot["Total Revenue"] = pivot.sum(axis=1)
    pivot = pivot.round(0).astype(int)

    new_columns = {}
    for col in pivot.columns:
        if isinstance(col, (int, float, pd.Int64Dtype)) and col != "Total Revenue":
             week_num = int(col) if pd.notna(col) else 'NaN'
             new_columns[col] = f"Week {week_num}"
        elif col == "Total Revenue":
            new_columns[col] = "Total Revenue"
        else:
             new_columns[col] = str(col)
    pivot = pivot.rename(columns=new_columns)

    week_cols = sorted([col for col in pivot.columns if col.startswith("Week ") and col.split()[1].isdigit()],
                       key=lambda x: int(x.split()[1]))
    if "Total Revenue" in pivot.columns:
        pivot = pivot[week_cols + ["Total Revenue"]]
    else:
        pivot = pivot[week_cols]

    return pivot


def create_sku_line_chart(data, sku_text, selected_years, selected_channels=None, week_range=None):
    """Creates the SKU Trends line chart."""
    required_cols = {"Product SKU", "Custom_Week_Year", "Week", "Sales Value (£)", "Order Quantity"}
    if not required_cols.issubset(data.columns):
        missing = required_cols.difference(data.columns)
        st.error(f"Dataset is missing required columns for SKU chart: {missing}")
        return go.Figure().update_layout(title_text=f"Missing data for SKU Chart: {missing}")

    filtered = data.copy()
    if "Product SKU" in filtered.columns:
         filtered["Product SKU"] = filtered["Product SKU"].astype(str)
         filtered = filtered[filtered["Product SKU"].str.contains(sku_text, case=False, na=False)]
    else:
         st.error("Column 'Product SKU' not found.")
         return go.Figure().update_layout(title_text="Column 'Product SKU' not found")

    if selected_years:
        filtered = filtered[filtered["Custom_Week_Year"].isin([int(y) for y in selected_years])]
    if selected_channels and len(selected_channels) > 0:
        if "Sales Channel" in filtered.columns:
            filtered = filtered[filtered["Sales Channel"].isin(selected_channels)]
        else: st.warning("Column 'Sales Channel' not found for filtering SKU chart.")

    if week_range:
        start_week, end_week = week_range
        if "Week" in filtered.columns:
             filtered["Week"] = pd.to_numeric(filtered["Week"], errors='coerce').astype('Int64')
             filtered.dropna(subset=["Week"], inplace=True)
             if not filtered.empty:
                  filtered = filtered[(filtered["Week"] >= start_week) & (filtered["Week"] <= end_week)]
        else: st.warning("Column 'Week' not found for week range filtering in SKU chart.")


    if filtered.empty:
        st.warning(f"No data available for SKU matching '{sku_text}' with selected filters.")
        return go.Figure().update_layout(title_text=f"No data for SKU: '{sku_text}'")

    filtered["Sales Value (£)"] = pd.to_numeric(filtered["Sales Value (£)"], errors='coerce')
    filtered["Order Quantity"] = pd.to_numeric(filtered["Order Quantity"], errors='coerce')
    filtered.dropna(subset=["Sales Value (£)", "Order Quantity", "Custom_Week_Year", "Week"], inplace=True)

    if filtered.empty:
         st.warning(f"No valid numeric data for SKU matching '{sku_text}' after cleaning.")
         return go.Figure().update_layout(title_text=f"No valid data for SKU: '{sku_text}'")

    weekly_sku = filtered.groupby(["Custom_Week_Year", "Week"]).agg({
        "Sales Value (£)": "sum",
        "Order Quantity": "sum"
    }).reset_index().sort_values(by=["Custom_Week_Year", "Week"])

    if weekly_sku.empty:
        st.warning("No data after grouping for SKU chart.")
        return go.Figure().update_layout(title_text=f"No data for SKU: '{sku_text}' after grouping")

    weekly_sku["RevenueK"] = weekly_sku["Sales Value (£)"] / 1000

    min_week_data = 1
    max_week_data = 52
    if not weekly_sku["Week"].empty:
         min_week_data_calc = weekly_sku["Week"].min()
         max_week_data_calc = weekly_sku["Week"].max()
         if pd.notna(min_week_data_calc): min_week_data = int(min_week_data_calc)
         if pd.notna(max_week_data_calc): max_week_data = int(max_week_data_calc)

    if week_range:
        min_week_plot, max_week_plot = week_range
    else:
        min_week_plot = min_week_data
        max_week_plot = max_week_data

    fig = px.line(weekly_sku, x="Week", y="Sales Value (£)", color="Custom_Week_Year", markers=True,
                  title=f"Weekly Revenue Trends for SKU matching: '{sku_text}'",
                  labels={"Sales Value (£)": "Revenue (£)", "Custom_Week_Year": "Year", "Week": "Week"},
                  custom_data=["RevenueK", "Order Quantity"])

    fig.update_traces(hovertemplate="Week: %{x}<br>Revenue: %{customdata[0]:.1f}K<br>Units Sold: %{customdata[1]}<extra></extra>")
    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
            range=[max(0.8, min_week_plot - 0.2), max_week_plot + 0.2],
            dtick=5 if (max_week_plot - min_week_plot) > 10 else 1
            ),
        yaxis=dict(rangemode="tozero"),
        margin=dict(t=50, b=50),
        legend_title_text='Year'
        )
    return fig


def create_daily_price_chart(data, listing, selected_years, selected_quarters, selected_channels, week_range=None):
    """Creates the Daily Average Price line chart."""
    if 'Year' not in data.columns and 'Custom_Week_Year' in data.columns:
         data['Year'] = data['Custom_Week_Year']

    required_cols = {"Date", "Listing", "Year", "Sales Value in Transaction Currency", "Order Quantity", "Week"}
    if not required_cols.issubset(data.columns):
         missing = required_cols.difference(data.columns)
         st.error(f"Dataset is missing required columns for Daily Price chart: {missing}")
         return go.Figure().update_layout(title_text=f"Missing data for Daily Prices: {missing}")


    selected_years_int = [int(y) for y in selected_years]
    df_listing = data[(data["Listing"] == listing) & (data["Year"].isin(selected_years_int))].copy()

    # Apply Quarter filter if selected_quarters is provided and Quarter column exists
    if selected_quarters:
        if "Quarter" not in df_listing.columns:
            st.warning("Column 'Quarter' not found for Daily Price filtering.")
        else:
            df_listing = df_listing[df_listing["Quarter"].isin(selected_quarters)]

    if selected_channels and len(selected_channels) > 0:
        if "Sales Channel" not in df_listing.columns:
            st.warning("Column 'Sales Channel' not found for Daily Price filtering.")
        else:
            df_listing = df_listing[df_listing["Sales Channel"].isin(selected_channels)]

    if week_range:
        start_week, end_week = week_range
        if "Week" not in df_listing.columns:
             st.warning("Column 'Week' not found for Daily Price week range filtering.")
        else:
             df_listing["Week"] = pd.to_numeric(df_listing["Week"], errors='coerce').astype('Int64')
             df_listing.dropna(subset=["Week"], inplace=True)
             if not df_listing.empty:
                  df_listing = df_listing[(df_listing["Week"] >= start_week) & (df_listing["Week"] <= end_week)]

    if df_listing.empty:
        st.warning(f"No data available for '{listing}' with the selected filters.")
        return None

    display_currency = "GBP"
    if "Original Currency" in df_listing.columns and not df_listing["Original Currency"].dropna().empty:
         unique_currencies = df_listing["Original Currency"].dropna().unique()
         if len(unique_currencies) > 0:
            display_currency = unique_currencies[0]
            if len(unique_currencies) > 1:
                 st.info(f"Note: Multiple transaction currencies found ({unique_currencies}). Displaying average price in {display_currency}.")


    df_listing["Date"] = pd.to_datetime(df_listing["Date"], errors='coerce')
    df_listing["Sales Value in Transaction Currency"] = pd.to_numeric(df_listing["Sales Value in Transaction Currency"], errors='coerce')
    df_listing["Order Quantity"] = pd.to_numeric(df_listing["Order Quantity"], errors='coerce')

    df_listing.dropna(subset=["Date", "Sales Value in Transaction Currency", "Order Quantity", "Year"], inplace=True)
    df_listing = df_listing[df_listing["Order Quantity"] > 0]


    if df_listing.empty:
        st.warning(f"No valid sales/quantity data for '{listing}' to calculate daily price after cleaning.")
        return None

    grouped = df_listing.groupby([df_listing["Date"].dt.date, "Year"]).agg(
        Total_Sales_Value=("Sales Value in Transaction Currency", "sum"),
        Total_Order_Quantity=("Order Quantity", "sum")
    ).reset_index()
    # Check if 'level_0' exists before renaming (depends on pandas version/grouping behavior)
    if 'level_0' in grouped.columns:
        grouped = grouped.rename(columns={'level_0': 'Date'})

    grouped["Average Price"] = grouped["Total_Sales_Value"] / grouped["Total_Order_Quantity"]
    grouped["Date"] = pd.to_datetime(grouped["Date"])

    dfs_processed = []
    for yr in selected_years_int:
        df_year = grouped[grouped["Year"] == yr].copy()
        if df_year.empty:
            continue

        df_year["Day"] = df_year["Date"].dt.dayofyear
        if df_year["Day"].empty or df_year["Day"].isna().all():
            continue
        start_day = int(df_year["Day"].min())
        end_day = int(df_year["Day"].max())


        df_year = df_year.set_index("Day").reindex(range(start_day, end_day + 1))
        df_year.index.name = "Day"

        df_year["Average Price"] = df_year["Average Price"].ffill()

        df_year["Average Price"] = pd.to_numeric(df_year["Average Price"], errors='coerce')
        df_year.dropna(subset=["Average Price"], inplace=True)
        if df_year.empty: continue

        prices = df_year["Average Price"].values.copy()
        last_valid_price = None
        for i in range(len(prices)):
                current_price = prices[i]
                if pd.notna(current_price):
                    if last_valid_price is not None:
                        if current_price < 0.75 * last_valid_price:
                            prices[i] = last_valid_price
                        elif current_price > 1.25 * last_valid_price:
                             prices[i] = last_valid_price
                    last_valid_price = prices[i]

        df_year["Smoothed Average Price"] = prices

        df_year["Year"] = yr
        df_year = df_year.reset_index()
        df_year.dropna(subset=["Smoothed Average Price"], inplace=True)

        if not df_year.empty:
             dfs_processed.append(df_year)

    if not dfs_processed:
        st.warning("No data available after processing for the Daily Price chart.")
        return None

    combined = pd.concat(dfs_processed, ignore_index=True)

    if combined.empty:
        st.warning("Combined data is empty for the Daily Price chart.")
        return None

    fig = px.line(
        combined,
        x="Day",
        y="Smoothed Average Price",
        color="Year",
        title=f"Daily Average Price for {listing}",
        labels={"Day": "Day of Year", "Smoothed Average Price": f"Avg Price ({display_currency})", "Year": "Year"},
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(margin=dict(t=50, b=50), legend_title_text='Year')
    return fig

# =============================================================================
# Main Dashboard Code
# =============================================================================

# --- Load Data from Google Sheets ---
df_raw = load_data_from_gsheet()

# Check if data loading was successful
if df_raw is None or df_raw.empty:
    st.warning("Failed to load data from Google Sheet or the sheet is empty. Dashboard cannot proceed.")
    st.info(f"Attempted to load from Sheet: '{GOOGLE_SHEET_NAME}', Worksheet: '{WORKSHEET_NAME}'")
    st.stop()

# --- Preprocess Data ---
try:
    df = preprocess_data(df_raw.copy()) # Pass a copy
except Exception as e:
    st.error(f"An error occurred during data preprocessing: {e}")
    st.error("Please check the 'preprocess_data' function and the structure/content of your Google Sheet.")
    st.error(traceback.format_exc())
    st.stop()

# Check if preprocessing returned valid data
if df is None or df.empty:
    st.error("Data is empty after preprocessing. Please check the 'preprocess_data' function logic and the source data.")
    st.stop()

# --- Prepare Filters and Variables based on Processed Data 'df' ---
if "Custom_Week_Year" not in df.columns:
    st.error("Critical Error: 'Custom_Week_Year' column not found after preprocessing.")
    st.stop()

available_custom_years = sorted(pd.to_numeric(df["Custom_Week_Year"], errors='coerce').dropna().unique().astype(int))

if not available_custom_years:
    st.error("No valid 'Custom_Week_Year' data found after preprocessing. Check calculations and sheet content.")
    st.stop()

current_custom_year = available_custom_years[-1]
if len(available_custom_years) >= 2:
    prev_custom_year = available_custom_years[-2]
    yoy_default_years = [prev_custom_year, current_custom_year]
else:
    prev_custom_year = None
    yoy_default_years = [current_custom_year]

default_current_year = [current_custom_year]


# --- Define Dashboard Tabs ---
tabs = st.tabs([
    "KPIs",
    "YOY Trends",
    "Daily Prices",
    "SKU Trends",
    "Pivot Table",
    "Unrecognised Sales"
])

# =============================================================================
# Tab Implementations
# =============================================================================

# -------------------------------
# Tab 1: KPIs
# -------------------------------
with tabs[0]:
    st.markdown("### Key Performance Indicators")
    with st.expander("KPI Filters", expanded=True):
        today = datetime.date.today()
        if "Custom_Week_Year" not in df.columns or "Week" not in df.columns:
            st.error("Missing 'Custom_Week_Year' or 'Week' for KPI calculations.")
            selected_week = None
        else:
            df["Week"] = pd.to_numeric(df["Week"], errors='coerce').astype('Int64')
            current_year_weeks = df[df["Custom_Week_Year"] == current_custom_year]["Week"].dropna()

            if not current_year_weeks.empty:
                 available_weeks = sorted(current_year_weeks.unique())
            else:
                 available_weeks = []

            full_weeks = []
            if available_weeks:
                for wk in available_weeks:
                    if pd.notna(wk):
                        try:
                            wk_int = int(wk)
                            week_start_dt, week_end_dt = get_custom_week_date_range(current_custom_year, wk_int)
                            if week_end_dt and week_end_dt < today:
                                full_weeks.append(wk_int)
                        except (ValueError, TypeError):
                            continue

            default_week = full_weeks[-1] if full_weeks else (available_weeks[-1] if available_weeks else 1)

            if available_weeks:
                selected_week = st.selectbox(
                    "Select Week for KPI Calculation",
                    options=available_weeks,
                    index=available_weeks.index(default_week) if default_week in available_weeks else 0,
                    key="kpi_week",
                    help="Select the week to calculate KPIs for. (Defaults to the last full week if available)"
                )

                if pd.notna(selected_week):
                     try:
                         selected_week_int = int(selected_week)
                         week_start_custom, week_end_custom = get_custom_week_date_range(current_custom_year, selected_week_int)
                         if week_start_custom and week_end_custom:
                            st.info(f"Selected Week {selected_week_int}: {week_start_custom.strftime('%d %b')} - {week_end_custom.strftime('%d %b, %Y')}")
                         else:
                            st.warning(f"Could not determine date range for Week {selected_week_int}, Year {current_custom_year}.")
                     except (ValueError, TypeError):
                         st.warning(f"Invalid week selected: {selected_week}")
                         selected_week = None
                else:
                     selected_week = None

            else:
                st.warning(f"No weeks found for the current year ({current_custom_year}) to calculate KPIs.")
                selected_week = None

    if selected_week is not None and pd.notna(selected_week):
        try:
             selected_week_int = int(selected_week)

             kpi_data = df[df["Week"] == selected_week_int].copy()

             if kpi_data.empty:
                 st.info(f"No sales data found for Week {selected_week_int} to calculate KPIs.")
             else:
                 revenue_summary = kpi_data.groupby("Custom_Week_Year")["Sales Value (£)"].sum()

                 if "Order Quantity" in kpi_data.columns:
                     kpi_data["Order Quantity"] = pd.to_numeric(kpi_data["Order Quantity"], errors='coerce')
                     units_summary = kpi_data.groupby("Custom_Week_Year")["Order Quantity"].sum().fillna(0)
                 else:
                     units_summary = None
                     st.info("Column 'Order Quantity' not found, units and AOV KPIs will not be shown.")

                 all_custom_years_in_df = sorted(pd.to_numeric(df["Custom_Week_Year"], errors='coerce').dropna().unique().astype(int))
                 kpi_cols = st.columns(len(all_custom_years_in_df))

                 for idx, year in enumerate(all_custom_years_in_df):
                     with kpi_cols[idx]:
                         revenue = revenue_summary.get(year, 0)

                         numeric_delta_rev = None
                         delta_rev_str = None
                         delta_rev_color = "off"

                         if idx > 0:
                             prev_year = all_custom_years_in_df[idx - 1]
                             prev_year_week_data = df[(df["Custom_Week_Year"] == prev_year) & (df["Week"] == selected_week_int)]
                             prev_rev = prev_year_week_data["Sales Value (£)"].sum() if not prev_year_week_data.empty else 0

                             if prev_rev != 0 or revenue != 0:
                                 numeric_delta_rev = revenue - prev_rev
                                 delta_rev_str = f"{int(round(numeric_delta_rev)):,}"
                                 delta_rev_color = "normal"

                         st.metric(
                             label=f"Revenue {year} (Week {selected_week_int})",
                             value=format_currency_int(revenue),
                             delta=delta_rev_str,
                             delta_color=delta_rev_color
                         )

                         if units_summary is not None:
                             total_units = units_summary.get(year, 0)

                             delta_units_str = None
                             delta_units_color = "off"
                             if idx > 0:
                                 prev_year = all_custom_years_in_df[idx - 1]
                                 prev_year_week_data = df[(df["Custom_Week_Year"] == prev_year) & (df["Week"] == selected_week_int)]
                                 prev_units = pd.to_numeric(prev_year_week_data['Order Quantity'], errors='coerce').sum() if not prev_year_week_data.empty else 0
                                 prev_units = prev_units if pd.notna(prev_units) else 0

                                 if prev_units != 0:
                                     delta_units_percent = ((total_units - prev_units) / prev_units) * 100
                                     delta_units_str = f"{delta_units_percent:.1f}%"
                                     delta_units_color = "normal"
                                 elif total_units != 0:
                                     delta_units_str = "vs 0"
                                     delta_units_color = "normal"

                             st.metric(
                                 label=f"Units Sold {year} (Week {selected_week_int})",
                                 value=f"{int(total_units):,}" if pd.notna(total_units) else "N/A",
                                 delta=delta_units_str,
                                 delta_color=delta_units_color
                             )

                             aov = revenue / total_units if total_units != 0 else 0
                             delta_aov_str = None
                             delta_aov_color = "off"
                             if idx > 0:
                                 prev_aov = prev_rev / prev_units if prev_units != 0 else 0
                                 if prev_aov != 0:
                                     delta_aov_percent = ((aov - prev_aov) / prev_aov) * 100
                                     delta_aov_str = f"{delta_aov_percent:.1f}%"
                                     delta_aov_color = "normal"
                                 elif aov != 0:
                                     delta_aov_str = "vs £0"
                                     delta_aov_color = "normal"

                             st.metric(
                                 label=f"AOV {year} (Week {selected_week_int})",
                                 value=format_currency(aov),
                                 delta=delta_aov_str,
                                 delta_color=delta_aov_color
                             )
        except (ValueError, TypeError):
             st.error(f"Invalid week number encountered: {selected_week}. Cannot calculate KPIs.")
    elif selected_week is None:
         st.info("Select a valid week from the filters above to view KPIs.")

# -----------------------------------------
# Tab 2: YOY Trends (UPDATED - Quarter filter removed)
# -----------------------------------------
with tabs[1]:
    st.markdown("### YOY Weekly Revenue Trends")
    # --- Filters ---
    with st.expander("Chart Filters", expanded=True):
        # <<< MODIFIED: Reduced columns back to 6 >>>
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            yoy_years = st.multiselect("Year(s)", options=available_custom_years, default=yoy_default_years, key="yoy_years")

        # <<< REMOVED: col2 block for Quarter filter >>>

        # <<< MODIFIED: Season Filter moved to col2 >>>
        with col2:
            selected_season = "ALL" # Default value
            if "Season" in df.columns:
                season_options_data = sorted(df["Season"].dropna().unique())
                filtered_season_data = [season for season in season_options_data if season != "AYR"]
                season_options = ["ALL"] + filtered_season_data
                selected_season = st.selectbox(
                    "Season",
                    options=season_options,
                    index=0,
                    key="yoy_season",
                    help="Filter data by season. Select ALL to include all seasons."
                )
            else:
                st.caption("Season filter unavailable (column missing)")

        # <<< MODIFIED: Channel, Listing, Product, Slider moved to subsequent columns >>>
        with col3: # Was col4
            if "Sales Channel" in df.columns:
                channel_options = sorted(df["Sales Channel"].dropna().unique())
                selected_channels = st.multiselect("Channel(s)", options=channel_options, default=[], key="yoy_channels")
            else:
                selected_channels = []
                st.caption("Sales Channel filter unavailable (column missing)")
        with col4: # Was col5
            if "Listing" in df.columns:
                listing_options = sorted(df["Listing"].dropna().unique())
                selected_listings = st.multiselect("Listing(s)", options=listing_options, default=[], key="yoy_listings")
            else:
                selected_listings = []
                st.caption("Listing filter unavailable (column missing)")
        with col5: # Was col6
            if "Product" in df.columns:
                if selected_listings:
                    product_options = sorted(df[df["Listing"].isin(selected_listings)]["Product"].dropna().unique())
                else:
                    product_options = sorted(df["Product"].dropna().unique())
                selected_products = st.multiselect("Product(s)", options=product_options, default=[], key="yoy_products")
            else:
                selected_products = []
                st.caption("Product filter unavailable (column missing)")

        # Slider placed in the 6th column
        with col6: # Was col7
            week_range_yoy = st.slider(
                "Select Week Range",
                min_value=1,
                max_value=53,
                value=(1, 53),
                step=1,
                key="yoy_week_range",
                help="Filter the YOY chart and summary table by week number."
            )

        # Time grouping option
        time_grouping = "Week" # Fixed to Week for YOY

    # --- Create and Display Chart ---
    if not yoy_years:
        st.warning("Please select at least one year in the filters to display the YOY chart.")
    else:
        # <<< MODIFIED: Call the updated chart function (removed selected_quarters) >>>
        fig_yoy = create_yoy_trends_chart(
            df,
            yoy_years,
            # selected_quarters, # Removed
            selected_channels,
            selected_listings,
            selected_products,
            time_grouping=time_grouping,
            week_range=week_range_yoy,
            selected_season=selected_season
        )
        st.plotly_chart(fig_yoy, use_container_width=True)

    # --- Revenue Summary Table (UPDATED Filtering) ---
    st.markdown("### Revenue Summary")
    st.markdown("") # Add space

    # Filter DataFrame based on selections for the summary table
    filtered_df_summary = df.copy()
    if yoy_years:
        filtered_df_summary = filtered_df_summary[filtered_df_summary["Custom_Week_Year"].isin([int(y) for y in yoy_years])]

    # <<< REMOVED: Quarter Filtering for Summary Table >>>
    # if selected_quarters:
    #     filtered_df_summary = filtered_df_summary[filtered_df_summary["Quarter"].isin(selected_quarters)]

    # Season Filtering for Summary Table
    if "Season" in filtered_df_summary.columns and selected_season != "ALL":
        filtered_df_summary = filtered_df_summary[filtered_df_summary["Season"] == selected_season]

    # Other existing filters...
    if selected_channels:
        if "Sales Channel" in filtered_df_summary.columns:
            filtered_df_summary = filtered_df_summary[filtered_df_summary["Sales Channel"].isin(selected_channels)]
    if selected_listings:
        if "Listing" in filtered_df_summary.columns:
            filtered_df_summary = filtered_df_summary[filtered_df_summary["Listing"].isin(selected_listings)]
    if selected_products:
        if "Product" in filtered_df_summary.columns:
            filtered_df_summary = filtered_df_summary[filtered_df_summary["Product"].isin(selected_products)]

    if 'week_range_yoy' in locals() and week_range_yoy:
        start_week, end_week = week_range_yoy
        if "Week" in filtered_df_summary.columns:
            filtered_df_summary["Week"] = pd.to_numeric(filtered_df_summary["Week"], errors='coerce').astype('Int64')
            filtered_df_summary.dropna(subset=["Week"], inplace=True)
            if not filtered_df_summary.empty:
                filtered_df_summary = filtered_df_summary[(filtered_df_summary["Week"] >= start_week) & (filtered_df_summary["Week"] <= end_week)]
        else:
            st.warning("Column 'Week' not found for week range filtering in Revenue Summary.")

    # Revenue summary table calculation code...
    if filtered_df_summary.empty:
        st.info("No data available for the selected filters (including season and week range) to build the revenue summary table.")
    else:
        filtered_df_summary["Custom_Week_Year"] = pd.to_numeric(filtered_df_summary["Custom_Week_Year"], errors='coerce').astype('Int64')
        filtered_df_summary["Week"] = pd.to_numeric(filtered_df_summary["Week"], errors='coerce').astype('Int64')
        filtered_df_summary["Sales Value (£)"] = pd.to_numeric(filtered_df_summary["Sales Value (£)"], errors='coerce')
        filtered_df_summary.dropna(subset=["Custom_Week_Year", "Week", "Sales Value (£)"], inplace=True)

        if filtered_df_summary.empty:
            st.info("No valid week/year/sales data after type conversion for summary table.")
        else:
            filtered_years_present = sorted(filtered_df_summary["Custom_Week_Year"].unique())
            if not filtered_years_present:
                st.info("No valid years found in filtered data for summary.")
            else:
                filtered_current_year = filtered_years_present[-1]
                df_revenue_current = filtered_df_summary[filtered_df_summary["Custom_Week_Year"] == filtered_current_year].copy()

                if "Custom_Week_Start" not in df_revenue_current.columns or "Custom_Week_End" not in df_revenue_current.columns:
                    try:
                        week_results_current = df_revenue_current.apply(lambda row: get_custom_week_date_range(row['Custom_Week_Year'], row['Week']) if pd.notna(row['Custom_Week_Year']) and pd.notna(row['Week']) else (None, None), axis=1)
                        if all(isinstance(item, tuple) and len(item) == 2 for item in week_results_current):
                            df_revenue_current[["Custom_Week_Start", "Custom_Week_End"]] = pd.DataFrame(week_results_current.tolist(), index=df_revenue_current.index)
                        else:
                            st.warning("Could not consistently derive week start/end dates for summary.")
                            df_revenue_current["Custom_Week_Start"] = pd.NaT
                            df_revenue_current["Custom_Week_End"] = pd.NaT
                    except Exception as e:
                        st.warning(f"Error deriving week start/end dates for summary: {e}")
                        df_revenue_current["Custom_Week_Start"] = pd.NaT
                        df_revenue_current["Custom_Week_End"] = pd.NaT


                if df_revenue_current.empty:
                    st.info(f"No data found for the latest filtered year ({filtered_current_year}) for summary.")
                else:
                    today = datetime.date.today()
                    df_revenue_current["Custom_Week_End_Date"] = pd.to_datetime(df_revenue_current["Custom_Week_End"], errors='coerce').dt.date
                    df_full_weeks_current = df_revenue_current.dropna(subset=["Custom_Week_End_Date", "Week"])
                    df_full_weeks_current = df_full_weeks_current[df_full_weeks_current["Custom_Week_End_Date"] < today].copy()

                    if not df_full_weeks_current.empty:
                          unique_weeks_current = (df_full_weeks_current.dropna(subset=["Week", "Custom_Week_Year"])
                                              .groupby(["Custom_Week_Year", "Week"])
                                              .agg(Week_End=("Custom_Week_End_Date", "first"))
                                              .reset_index()
                                              .sort_values("Week_End", na_position='first'))
                    else:
                         unique_weeks_current = pd.DataFrame(columns=["Custom_Week_Year", "Week", "Week_End"])


                    if unique_weeks_current.empty or unique_weeks_current['Week_End'].isna().all():
                        st.info("Not enough complete week data in the filtered current year to build the revenue summary table.")
                    else:
                        last_complete_week_row_current = unique_weeks_current.dropna(subset=['Week_End']).iloc[-1]
                        last_week_number = int(last_complete_week_row_current["Week"])
                        last_week_year = int(last_complete_week_row_current["Custom_Week_Year"])

                        last_4_weeks_current = unique_weeks_current.drop_duplicates(subset=["Week"]).dropna(subset=['Week_End']).tail(4)
                        last_4_week_numbers = last_4_weeks_current["Week"].astype(int).tolist()

                        grouping_key = None
                        if "Product" in filtered_df_summary.columns and selected_listings and len(selected_listings) == 1:
                            grouping_key = "Product"
                        elif "Listing" in filtered_df_summary.columns:
                            grouping_key = "Listing"
                        else:
                            st.warning("Cannot determine grouping key (Listing/Product) for summary table.")

                        if grouping_key:
                            rev_last_4_current = (df_full_weeks_current[df_full_weeks_current["Week"].isin(last_4_week_numbers)]
                                                    .groupby(grouping_key)["Sales Value (£)"].sum()
                                                    .rename(f"Last 4 Weeks Revenue ({last_week_year})").round(0))

                            rev_last_1_current = (df_full_weeks_current[df_full_weeks_current["Week"] == last_week_number]
                                                    .groupby(grouping_key)["Sales Value (£)"].sum()
                                                    .rename(f"Last Week Revenue ({last_week_year})").round(0))

                            rev_last_4_last_year = pd.Series(dtype='float64', name="Last 4 Weeks Revenue (Prev Year)")
                            rev_last_1_last_year = pd.Series(dtype='float64', name="Last Week Revenue (Prev Year)")
                            prev_year_label = "Prev Year"
                            last_year = None

                            if last_week_year in filtered_years_present:
                                current_year_index = filtered_years_present.index(last_week_year)
                                if current_year_index > 0:
                                    last_year = filtered_years_present[current_year_index - 1]
                                    prev_year_label = str(last_year)

                            if last_year is not None:
                                df_revenue_last_year = filtered_df_summary[filtered_df_summary["Custom_Week_Year"] == last_year].copy()
                                if not df_revenue_last_year.empty:
                                    df_revenue_last_year["Week"] = pd.to_numeric(df_revenue_last_year["Week"], errors='coerce').astype('Int64')
                                    df_revenue_last_year.dropna(subset=["Week", "Sales Value (£)"], inplace=True)

                                    if not df_revenue_last_year.empty:
                                        rev_last_1_last_year = (df_revenue_last_year[df_revenue_last_year["Week"] == last_week_number]
                                                                    .groupby(grouping_key)["Sales Value (£)"].sum()
                                                                    .rename(f"Last Week Revenue ({last_year})").round(0))

                                        rev_last_4_last_year = (df_revenue_last_year[df_revenue_last_year["Week"].isin(last_4_week_numbers)]
                                                                    .groupby(grouping_key)["Sales Value (£)"].sum()
                                                                    .rename(f"Last 4 Weeks Revenue ({last_year})").round(0))

                            all_keys = pd.Series(sorted(filtered_df_summary[grouping_key].dropna().unique()), name=grouping_key)
                            revenue_summary = pd.DataFrame({grouping_key: all_keys}).set_index(grouping_key)


                            revenue_summary = revenue_summary.join(rev_last_4_current)\
                                                            .join(rev_last_1_current)\
                                                            .join(rev_last_4_last_year.rename(f"Last 4 Weeks Revenue ({prev_year_label})"))\
                                                            .join(rev_last_1_last_year.rename(f"Last Week Revenue ({prev_year_label})"))


                            revenue_summary = revenue_summary.fillna(0)


                            current_4wk_col = f"Last 4 Weeks Revenue ({last_week_year})"
                            prev_4wk_col = f"Last 4 Weeks Revenue ({prev_year_label})"
                            current_1wk_col = f"Last Week Revenue ({last_week_year})"
                            prev_1wk_col = f"Last Week Revenue ({prev_year_label})"

                            if current_4wk_col in revenue_summary.columns and prev_4wk_col in revenue_summary.columns:
                                revenue_summary["Last 4 Weeks Diff"] = revenue_summary[current_4wk_col] - revenue_summary[prev_4wk_col]
                            else: revenue_summary["Last 4 Weeks Diff"] = 0

                            if current_1wk_col in revenue_summary.columns and prev_1wk_col in revenue_summary.columns:
                                revenue_summary["Last Week Diff"] = revenue_summary[current_1wk_col] - revenue_summary[prev_1wk_col]
                            else: revenue_summary["Last Week Diff"] = 0

                            revenue_summary["Last 4 Weeks % Change"] = revenue_summary.apply(
                                lambda row: (row["Last 4 Weeks Diff"] / row[prev_4wk_col] * 100)
                                if prev_4wk_col in row and row[prev_4wk_col] != 0 else
                                (100.0 if "Last 4 Weeks Diff" in row and row["Last 4 Weeks Diff"] > 0 else 0.0), axis=1)

                            revenue_summary["Last Week % Change"] = revenue_summary.apply(
                                lambda row: (row["Last Week Diff"] / row[prev_1wk_col] * 100)
                                if prev_1wk_col in row and row[prev_1wk_col] != 0 else
                                (100.0 if "Last Week Diff" in row and row["Last Week Diff"] > 0 else 0.0), axis=1)

                            revenue_summary = revenue_summary.reset_index()

                            desired_order = [grouping_key,
                                                current_4wk_col, prev_4wk_col, "Last 4 Weeks Diff", "Last 4 Weeks % Change",
                                                current_1wk_col, prev_1wk_col, "Last Week Diff", "Last Week % Change"]
                            desired_order = [col for col in desired_order if col in revenue_summary.columns]
                            revenue_summary = revenue_summary[desired_order]

                            summary_row = {}
                            for col in desired_order:
                                if col != grouping_key and pd.api.types.is_numeric_dtype(revenue_summary[col]):
                                    summary_row[col] = revenue_summary[col].sum()
                                else:
                                    summary_row[col] = ''

                            summary_row[grouping_key] = "Total"

                            total_last4_last_year = summary_row.get(prev_4wk_col, 0)
                            total_last_week_last_year = summary_row.get(prev_1wk_col, 0)
                            total_diff_4wk = summary_row.get("Last 4 Weeks Diff", 0)
                            total_diff_1wk = summary_row.get("Last Week Diff", 0)

                            summary_row["Last 4 Weeks % Change"] = (total_diff_4wk / total_last4_last_year * 100) if total_last4_last_year != 0 else (100.0 if total_diff_4wk > 0 else 0.0)
                            summary_row["Last Week % Change"] = (total_diff_1wk / total_last_week_last_year * 100) if total_last_week_last_year != 0 else (100.0 if total_diff_1wk > 0 else 0.0)

                            total_df = pd.DataFrame([summary_row])[desired_order]

                            def color_diff(val):
                                try:
                                    val = float(val)
                                    if val < -0.001: return 'color: red'
                                    elif val > 0.001: return 'color: green'
                                    else: return ''
                                except (ValueError, TypeError):
                                    return ''

                            formats = {}
                            if current_4wk_col in revenue_summary.columns: formats[current_4wk_col] = "£{:,.0f}"
                            if prev_4wk_col in revenue_summary.columns: formats[prev_4wk_col] = "£{:,.0f}"
                            if current_1wk_col in revenue_summary.columns: formats[current_1wk_col] = "£{:,.0f}"
                            if prev_1wk_col in revenue_summary.columns: formats[prev_1wk_col] = "£{:,.0f}"
                            if "Last 4 Weeks Diff" in revenue_summary.columns: formats["Last 4 Weeks Diff"] = "{:,.0f}"
                            if "Last Week Diff" in revenue_summary.columns: formats["Last Week Diff"] = "{:,.0f}"
                            if "Last 4 Weeks % Change" in revenue_summary.columns: formats["Last 4 Weeks % Change"] = "{:.1f}%"
                            if "Last Week % Change" in revenue_summary.columns: formats["Last Week % Change"] = "{:.1f}%"

                            color_cols = [col for col in ["Last 4 Weeks Diff", "Last Week Diff", "Last 4 Weeks % Change", "Last Week % Change"] if col in revenue_summary.columns]

                            styled_total = total_df.style.format(formats, na_rep='-').apply(lambda x: x.map(color_diff), subset=color_cols)\
                                                        .set_properties(**{'font-weight': 'bold'}) \
                                                        .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                                            {'selector': 'td', 'props': [('text-align', 'right')]}])

                            st.markdown("##### Total Summary")
                            st.dataframe(styled_total, use_container_width=True, hide_index=True)

                            value_cols_to_int = [col for col in [current_4wk_col, prev_4wk_col, current_1wk_col, prev_1wk_col, "Last 4 Weeks Diff", "Last Week Diff"] if col in revenue_summary.columns]
                            revenue_summary[value_cols_to_int] = revenue_summary[value_cols_to_int].astype(int)

                            styled_main = revenue_summary.style.format(formats, na_rep='-').apply(lambda x: x.map(color_diff), subset=color_cols) \
                                                        .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                                            {'selector': 'td', 'props': [('text-align', 'right')]}])

                            st.markdown("##### Detailed Summary")
                            st.dataframe(styled_main, use_container_width=True, hide_index=True)
                        else:
                            st.info("Summary table requires 'Listing' or 'Product' column and valid selections.")


# -------------------------------
# Tab 3: Daily Prices
# -------------------------------
with tabs[2]:
    st.markdown("### Daily Prices for Top Listings")
    # --- Filters ---
    with st.expander("Daily Price Filters", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            today_year = datetime.date.today().year
            default_daily_years = [year for year in available_custom_years if year >= today_year - 1]
            if not default_daily_years: default_daily_years = default_current_year
            elif len(default_daily_years) > 2: default_daily_years = default_daily_years[-2:]

            selected_daily_years = st.multiselect(
                "Select Year(s)",
                options=available_custom_years,
                default=default_daily_years,
                key="daily_years",
                help="Select year(s) to display daily prices."
                )
        with col2:
            # Keep Quarter filter available for this tab
            quarter_options_daily = ["All Quarters", "Q1", "Q2", "Q3", "Q4", "Custom..."]
            quarter_selection_daily = st.selectbox("Quarter(s)", options=quarter_options_daily, index=0, key="quarter_dropdown_daily_prices")

            if quarter_selection_daily == "Custom...":
                if "Quarter" in df.columns:
                    quarter_opts_daily = sorted(df["Quarter"].dropna().unique())
                else: quarter_opts_daily = ["Q1", "Q2", "Q3", "Q4"]
                selected_daily_quarters = st.multiselect("Select Quarter(s)", options=quarter_opts_daily, default=[], key="daily_quarters_custom", help="Select one or more quarters to filter.")
            elif quarter_selection_daily == "All Quarters":
                # Need to use the actual quarter names present in the data if available
                 selected_daily_quarters = df["Quarter"].dropna().unique().tolist() if "Quarter" in df.columns else ["Q1", "Q2", "Q3", "Q4"]
            else:
                selected_daily_quarters = [quarter_selection_daily]
        with col3:
             if "Sales Channel" in df.columns:
                 channel_options_daily = sorted(df["Sales Channel"].dropna().unique())
                 selected_daily_channels = st.multiselect("Select Sales Channel(s)", options=channel_options_daily, default=[], key="daily_channels", help="Select one or more sales channels to filter the daily price data.")
             else:
                 selected_daily_channels = []
                 st.caption("Sales Channel filter unavailable")
        with col4:
             daily_week_range = st.slider(
                 "Select Week Range",
                 min_value=1,
                 max_value=53,
                 value=(1, 53),
                 step=1,
                 key="daily_week_range",
                 help="Select the range of weeks to display in the Daily Prices section."
                 )

    # --- Display Charts for Main Listings ---
    main_listings = ["Pattern Pants", "Pattern Shorts", "Solid Pants", "Solid Shorts", "Patterned Polos"]

    if "Listing" not in df.columns:
        st.error("Column 'Listing' not found. Cannot display Daily Price charts.")
    else:
        available_main_listings = [l for l in main_listings if l in df["Listing"].unique()]
        if not available_main_listings:
             st.warning("None of the specified main listings found in the data.")
        else:
            if not selected_daily_years:
                 st.warning("Please select at least one year in the filters to view Daily Price charts.")
            else:
                for listing in available_main_listings:
                    st.subheader(listing)
                    # Pass selected_daily_quarters to the function
                    fig_daily = create_daily_price_chart(df, listing, selected_daily_years, selected_daily_quarters, selected_daily_channels, week_range=daily_week_range)
                    if fig_daily:
                        st.plotly_chart(fig_daily, use_container_width=True)

    # --- Daily Prices Comparison Section ---
    st.markdown("### Daily Prices Comparison")
    with st.expander("Comparison Chart Filters", expanded=False):
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        with comp_col1:
            comp_years = st.multiselect("Select Year(s)", options=available_custom_years, default=default_daily_years, key="comp_years", help="Select the year(s) for the comparison chart.")
        with comp_col2:
            # Keep Quarter filter available for comparison section
            comp_quarter_options = ["All Quarters", "Q1", "Q2", "Q3", "Q4", "Custom..."]
            comp_quarter_selection = st.selectbox("Quarter(s)", options=comp_quarter_options, index=0, key="quarter_dropdown_comp_prices")
            if comp_quarter_selection == "Custom...":
                if "Quarter" in df.columns:
                    comp_quarter_opts = sorted(df["Quarter"].dropna().unique())
                else: comp_quarter_opts = ["Q1", "Q2", "Q3", "Q4"]
                comp_quarters = st.multiselect("Select Quarter(s)", options=comp_quarter_opts, default=[], key="comp_quarters_custom", help="Select one or more quarters for comparison.")
            elif comp_quarter_selection == "All Quarters":
                 comp_quarters = df["Quarter"].dropna().unique().tolist() if "Quarter" in df.columns else ["Q1", "Q2", "Q3", "Q4"]
            else:
                comp_quarters = [comp_quarter_selection]
        with comp_col3:
            if "Sales Channel" in df.columns:
                comp_channel_opts = sorted(df["Sales Channel"].dropna().unique())
                comp_channels = st.multiselect("Select Sales Channel(s)", options=comp_channel_opts, default=[], key="comp_channels", help="Select the sales channel(s) for the comparison chart.")
            else:
                comp_channels = []
                st.caption("Sales Channel filter unavailable")

        if "Listing" in df.columns:
             comp_listing_opts = sorted(df["Listing"].dropna().unique())
             comp_listing_default_index = 0 if comp_listing_opts else -1
             comp_listing = st.selectbox("Select Listing", options=comp_listing_opts, index=comp_listing_default_index , key="comp_listing", help="Select a listing for daily prices comparison.")
        else:
             comp_listing = None
             st.warning("Listing selection unavailable (column missing)")

    if comp_listing and comp_years:
        # Pass comp_quarters to the function
        fig_comp = create_daily_price_chart(df, comp_listing, comp_years, comp_quarters, comp_channels, week_range=None) # No week range slider here
        if fig_comp:
            st.plotly_chart(fig_comp, use_container_width=True)
    elif not comp_listing and "Listing" in df.columns:
        st.info("Select a listing in the comparison filters above to view the comparison chart.")
    elif not comp_years:
        st.info("Select at least one year in the comparison filters above to view the comparison chart.")

# -------------------------------
# Tab 4: SKU Trends
# -------------------------------
with tabs[3]:
    st.markdown("### SKU Trends")
    if "Product SKU" not in df.columns:
        st.error("The dataset does not contain a 'Product SKU' column. SKU Trends cannot be displayed.")
    else:
        with st.expander("Chart Filters", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                sku_text = st.text_input("Enter Product SKU", value="", key="sku_input", help="Enter a SKU (or part of it) to display its weekly revenue trends.")
            with col2:
                sku_years = st.multiselect("Select Year(s)", options=available_custom_years, default=yoy_default_years, key="sku_years", help="Default includes current and previous custom week year.")
            with col3:
                if "Sales Channel" in df.columns:
                    sku_channel_opts = sorted(df["Sales Channel"].dropna().unique())
                    sku_channels = st.multiselect("Select Sales Channel(s)", options=sku_channel_opts, default=[], key="sku_channels", help="Select one or more sales channels to filter SKU trends. If empty, all channels are shown.")
                else:
                    sku_channels = []
                    st.caption("Sales Channel filter unavailable")
            with col4:
                week_range_sku = st.slider("Select Week Range", min_value=1, max_value=53, value=(1, 53), step=1, key="sku_week_range", help="Select the range of weeks to display.")

        if sku_text.strip() == "":
            st.info("Please enter a Product SKU in the filters above to view its trends.")
        elif not sku_years:
            st.warning("Please select at least one year in the filters to view SKU trends.")
        else:
            fig_sku = create_sku_line_chart(df, sku_text, sku_years, selected_channels=sku_channels, week_range=week_range_sku)
            if fig_sku is not None:
                st.plotly_chart(fig_sku, use_container_width=True)

            filtered_sku_data = df.copy()
            if "Product SKU" in filtered_sku_data.columns:
                 filtered_sku_data["Product SKU"] = filtered_sku_data["Product SKU"].astype(str)
                 filtered_sku_data = filtered_sku_data[filtered_sku_data["Product SKU"].str.contains(sku_text, case=False, na=False)]
            else:
                 filtered_sku_data = pd.DataFrame() # Empty df if SKU col missing

            if sku_years:
                filtered_sku_data = filtered_sku_data[filtered_sku_data["Custom_Week_Year"].isin(sku_years)]
            if sku_channels and len(sku_channels) > 0:
                 if "Sales Channel" in filtered_sku_data.columns:
                    filtered_sku_data = filtered_sku_data[filtered_sku_data["Sales Channel"].isin(sku_channels)]
            if week_range_sku:
                if "Week" in filtered_sku_data.columns:
                     filtered_sku_data["Week"] = pd.to_numeric(filtered_sku_data["Week"], errors='coerce').astype('Int64')
                     filtered_sku_data.dropna(subset=["Week"], inplace=True)
                     if not filtered_sku_data.empty:
                           filtered_sku_data = filtered_sku_data[(filtered_sku_data["Week"] >= week_range_sku[0]) & (filtered_sku_data["Week"] <= week_range_sku[1])]
                else: st.warning("Week column missing for week range filter in SKU summary.")


            if "Order Quantity" in filtered_sku_data.columns:
                filtered_sku_data["Order Quantity"] = pd.to_numeric(filtered_sku_data["Order Quantity"], errors='coerce')
                filtered_sku_data.dropna(subset=["Order Quantity", "Custom_Week_Year", "Product SKU"], inplace=True)

                if not filtered_sku_data.empty:
                    total_units = filtered_sku_data.groupby("Custom_Week_Year")["Order Quantity"].sum().reset_index()
                    if not total_units.empty:
                        total_units_summary = total_units.pivot(index=None, columns="Custom_Week_Year", values="Order Quantity")
                        total_units_summary.index = ["Total Units Sold (All Matching SKUs)"]
                        st.markdown("##### Total Units Sold Summary")
                        st.dataframe(total_units_summary.fillna(0).astype(int).style.format("{:,}"), use_container_width=True)
                    # else: st.info("No units sold data for total summary.") # Can be noisy

                    sku_units = filtered_sku_data.groupby(["Product SKU", "Custom_Week_Year"])["Order Quantity"].sum().reset_index()
                    if not sku_units.empty:
                         sku_pivot = sku_units.pivot(index="Product SKU", columns="Custom_Week_Year", values="Order Quantity")
                         sku_pivot = sku_pivot.fillna(0).astype(int)
                         st.markdown("##### SKU Breakdown (Units Sold by Custom Week Year)")
                         st.dataframe(sku_pivot.style.format("{:,}"), use_container_width=True)
                    # else: st.info("No data for SKU breakdown table.") # Can be noisy
                # else: # Covered by chart warning
                #     st.info("No valid 'Order Quantity' data found for the selected SKU filters after cleaning.")
            else:
                st.info("Column 'Order Quantity' not found, cannot show units sold summary.")

# -------------------------------
# Tab 5: Pivot Table: Revenue by Week
# -------------------------------
with tabs[4]:
    st.markdown("### Pivot Table: Revenue by Week")
    with st.expander("Pivot Table Filters", expanded=False):
        pivot_years = st.multiselect("Select Year(s) for Pivot Table", options=available_custom_years, default=default_current_year, key="pivot_years", help="Default is the current custom week year.")

        # Keep Quarter filter available for Pivot Table tab
        if "Quarter" in df.columns:
             pivot_quarter_opts = sorted(df["Quarter"].dropna().unique())
             # Default to empty list unless user selects quarters
             pivot_quarters = st.multiselect("Select Quarter(s)", options=pivot_quarter_opts, default=[], key="pivot_quarters_tab", help="Select one or more quarters to filter by. Default shows all.")
        else:
             pivot_quarters = []
             st.caption("Quarter filter unavailable")

        if "Sales Channel" in df.columns:
             pivot_channel_opts = sorted(df["Sales Channel"].dropna().unique())
             pivot_channels = st.multiselect("Select Sales Channel(s)", options=pivot_channel_opts, default=[], key="pivot_channels_tab", help="Select one or more channels to filter. If empty, all channels are shown.")
        else:
             pivot_channels = []
             st.caption("Sales Channel filter unavailable")

        if "Listing" in df.columns:
             pivot_listing_opts = sorted(df["Listing"].dropna().unique())
             pivot_listings = st.multiselect("Select Listing(s)", options=pivot_listing_opts, default=[], key="pivot_listings_tab", help="Select one or more listings to filter. If empty, all listings are shown.")
        else:
             pivot_listings = []
             st.caption("Listing filter unavailable")

        pivot_product_options = []
        if "Product" in df.columns:
             if pivot_listings and len(pivot_listings) > 0:
                 pivot_product_options = sorted(df[df["Listing"].isin(pivot_listings)]["Product"].dropna().unique())
             else:
                 pivot_product_options = sorted(df["Product"].dropna().unique())
             pivot_products = st.multiselect("Select Product(s)", options=pivot_product_options, default=[], key="pivot_products_tab", help="Select one or more products to filter. Options depend on selected listings.")
        else:
             pivot_products = []
             st.caption("Product filter unavailable")


    if not pivot_years:
        st.warning("Please select at least one year for the Pivot Table.")
    else:
        grouping_key = "Product" if (pivot_listings and len(pivot_listings) == 1 and "Product" in df.columns) else ("Listing" if "Listing" in df.columns else None)

        if grouping_key:
            effective_products = pivot_products if grouping_key == "Product" else []
            # Pass the selected pivot_quarters to the function
            # If pivot_quarters is empty, the function should handle it (treat as no filter)
            effective_quarters = pivot_quarters if pivot_quarters else df['Quarter'].dropna().unique().tolist() if 'Quarter' in df.columns else [] # Pass all if empty, but ensure Quarter exists


            pivot = create_pivot_table(
                df,
                selected_years=pivot_years,
                selected_quarters=effective_quarters, # Pass selected quarters
                selected_channels=pivot_channels,
                selected_listings=pivot_listings,
                selected_products=effective_products,
                grouping_key=grouping_key
                )

            is_real_pivot = not pivot.index.name or (pivot.index.name == grouping_key and pivot.index.name in df.columns and not pivot.empty and pivot.index[0] not in ["No data", "Missing columns", "No results", "No valid data", "Missing grouping column", "Missing 'Week' column"])


            if len(pivot_years) == 1 and is_real_pivot and isinstance(pivot.columns, pd.Index):
                try:
                    year_for_date = int(pivot_years[0])
                    new_columns_tuples = []
                    for col_name in pivot.columns:
                        if col_name == "Total Revenue":
                            new_columns_tuples.append(("Total Revenue", ""))
                        elif isinstance(col_name, str) and col_name.startswith("Week "):
                            try:
                                week_number = int(col_name.split()[1])
                                mon, fri = get_custom_week_date_range(year_for_date, week_number)
                                date_range = f"{mon.strftime('%d %b')} - {fri.strftime('%d %b')}" if mon and fri else ""
                                new_columns_tuples.append((col_name, date_range))
                            except (IndexError, ValueError, TypeError):
                                new_columns_tuples.append((col_name, ""))
                        else:
                            new_columns_tuples.append((str(col_name), ""))

                    if new_columns_tuples:
                        pivot.columns = pd.MultiIndex.from_tuples(new_columns_tuples, names=["Metric", "Date Range"])

                except Exception as e:
                     st.warning(f"Could not create multi-index header for pivot table: {e}")

            if is_real_pivot:
                 format_dict = {}
                 if isinstance(pivot.columns, pd.MultiIndex):
                      format_dict = {col: "{:,}" for col in pivot.columns if col[0] != 'Total Revenue'}
                 elif isinstance(pivot.columns, pd.Index): # Handle single index case
                      format_dict = {col: "{:,}" for col in pivot.columns if col != 'Total Revenue'}

                 st.dataframe(pivot.style.format(format_dict, na_rep='0'), use_container_width=True)
            else:
                 st.dataframe(pivot, use_container_width=True) # Display message dataframe

        else:
             st.error("Cannot create pivot table: Required grouping column ('Listing' or 'Product') not found in data.")


# -------------------------------
# Tab 6: Unrecognised Sales
# -------------------------------
with tabs[5]:
    st.markdown("### Unrecognised Sales")
    if "Listing" not in df.columns:
        st.error("Column 'Listing' not found. Cannot identify unrecognised sales.")
    else:
        df["Listing"] = df["Listing"].astype(str)
        unrecognised_sales = df[df["Listing"].str.contains("unrecognised", case=False, na=False)].copy()

        columns_to_drop_orig = ["Year", "Weekly Sales Value (£)", "YOY Growth (%)", "Custom_Week", "Custom_Week_Year", "Custom_Week_Start", "Custom_Week_End", "Quarter"]
        columns_to_drop_existing = [col for col in columns_to_drop_orig if col in unrecognised_sales.columns]

        if columns_to_drop_existing:
            unrecognised_sales = unrecognised_sales.drop(columns=columns_to_drop_existing, errors='ignore')

        if unrecognised_sales.empty:
            st.info("No unrecognised sales found based on 'Listing' column containing 'unrecognised'.")
        else:
            st.info(f"Found {len(unrecognised_sales)} rows potentially related to unrecognised sales.")

            display_cols_order = [
                "Date", "Week", "Sales Channel", "Listing", "Product SKU", "Product",
                "Sales Value (£)", "Order Quantity", "Sales Value in Transaction Currency", "Original Currency"
            ]
            display_cols_existing = [col for col in display_cols_order if col in unrecognised_sales.columns]
            remaining_cols = [col for col in unrecognised_sales.columns if col not in display_cols_existing]
            final_display_cols = display_cols_existing + remaining_cols

            style_format = {}
            if "Sales Value (£)" in final_display_cols:
                 style_format["Sales Value (£)"] = "£{:,.2f}"
            if "Sales Value in Transaction Currency" in final_display_cols:
                 # Attempt to get the actual currency symbol if consistent, else just format number
                 # This might be complex if multiple currencies exist in unrecognized sales
                 style_format["Sales Value in Transaction Currency"] = "{:,.2f}"

            st.dataframe(unrecognised_sales[final_display_cols].style.format(style_format, na_rep='-'), use_container_width=True, hide_index=True)
