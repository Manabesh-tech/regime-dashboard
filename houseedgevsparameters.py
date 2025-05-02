pythonimport streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import pytz

# Page configuration
st.set_page_config(
    page_title="House Edge Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.auto_update = False  # Start with auto-update disabled
    st.session_state.simulated_data_mode = True  # Force simulated mode
    st.session_state.last_update_time = datetime.now()

# Title
st.title('House Edge Parameter Adjustment Dashboard')
st.markdown("This dashboard monitors house edge and dynamically adjusts parameters.")

# Current time display
now = datetime.now()
st.write(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")

# Simple update button
if st.button("Update Now"):
    st.session_state.last_update_time = datetime.now()
    st.success("Updated successfully!")

# Display when last updated
st.write(f"Last updated: {st.session_state.last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Sidebar controls
st.sidebar.title("Dashboard Controls")

# Auto-update toggle (disabled for debugging)
auto_update = st.sidebar.checkbox("Enable Auto-Update", value=False, disabled=True)
st.sidebar.warning("Auto-update temporarily disabled for debugging")

# Add some test data visualization
st.subheader("Test Chart")
x = np.linspace(0, 10, 100)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
st.pyplot(fig)

# Show session state for debugging
st.subheader("Debug: Session State")
st.write({k: v for k, v in st.session_state.items() if not callable(v)})