# tws_connector.py - Separate file to handle TWS connection for Streamlit

import streamlit as st
import threading
import time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class StreamlitTWSApp(EWrapper, EClient):
    def __init__(self, tracker):
        EClient.__init__(self, self)
        self.tracker = tracker
        self.req_id_map = {}
        self.next_req_id = 1000
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in [2104, 2106, 2158, 2176]:  # Ignore common info messages
            st.error(f"TWS Error {errorCode}: {errorString}")
    
    def connectAck(self):
        st.success("✅ Connected to TWS successfully!")
        
    def nextValidId(self, orderId):
        self.start_data_requests()
        
    def tickPrice(self, reqId, tickType, price, attrib):
        """Handle incoming price data"""
        if reqId in self.req_id_map:
            symbol = self.req_id_map[reqId]['symbol']
            tick_types = {1: "BID", 2: "ASK", 4: "LAST"}
            
            if tickType in tick_types:
                # Update tracker data
                self.tracker.data[symbol][tick_types[tickType]] = price
                
                # Add price point for volatility calculation
                if tickType in [1, 2]:  # BID or ASK
                    bid = self.tracker.data[symbol]["BID"]
                    ask = self.tracker.data[symbol]["ASK"]
                    if bid > 0 and ask > 0:
                        mid_price = (bid + ask) / 2
                        self.tracker.add_price_update(symbol, mid_price)
                elif tickType == 4:  # LAST trade price
                    self.tracker.add_price_update(symbol, price)
    
    def tickSize(self, reqId, tickType, size):
        """Handle incoming size data"""
        if reqId in self.req_id_map:
            symbol = self.req_id_map[reqId]['symbol']
            tick_types = {0: "BID_SIZE", 3: "ASK_SIZE", 5: "LAST_SIZE", 8: "VOLUME"}
            
            if tickType in tick_types:
                self.tracker.data[symbol][tick_types[tickType]] = size
    
    def start_data_requests(self):
        """Start requesting data for selected stocks"""
        if 'selected_stocks' in st.session_state:
            for symbol in st.session_state.selected_stocks:
                # Initialize stock in tracker if not exists
                if symbol not in self.tracker.data:
                    self.tracker.initialize_stock(symbol)
                
                # Create contract
                contract = Contract()
                contract.symbol = symbol
                contract.secType = "STK"
                contract.exchange = "SMART"
                contract.currency = "USD"
                
                # Request market data
                req_id = self.next_req_id
                self.req_id_map[req_id] = {'symbol': symbol, 'type': 'market_data'}
                self.reqMktData(req_id, contract, "", False, False, [])
                
                self.next_req_id += 1
                time.sleep(0.1)  # Small delay between requests

class TWSConnectionManager:
    def __init__(self, tracker):
        self.tracker = tracker
        self.app = None
        self.connection_thread = None
        self.volatility_thread = None
        self.is_connected = False
        
    def connect(self, host="127.0.0.1", port=7497, client_id=1):
        """Connect to TWS"""
        try:
            self.app = StreamlitTWSApp(self.tracker)
            self.app.connect(host, port, client_id)
            
            # Start connection thread
            self.connection_thread = threading.Thread(target=self.app.run)
            self.connection_thread.daemon = True
            self.connection_thread.start()
            
            # Start volatility calculation thread
            self.start_volatility_calculator()
            
            self.is_connected = True
            return True
            
        except Exception as e:
            st.error(f"Failed to connect to TWS: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from TWS"""
        if self.app:
            self.app.disconnect()
        self.is_connected = False
    
    def start_volatility_calculator(self):
        """Start background volatility calculation"""
        def volatility_worker():
            while self.is_connected:
                time.sleep(10)  # Calculate every 10 seconds
                
                for symbol in list(self.tracker.data.keys()):
                    self.tracker.process_10s_window(symbol)
        
        self.volatility_thread = threading.Thread(target=volatility_worker)
        self.volatility_thread.daemon = True
        self.volatility_thread.start()

# Integration function for Streamlit
def get_tws_manager():
    """Get or create TWS connection manager"""
    if 'tws_manager' not in st.session_state:
        st.session_state.tws_manager = TWSConnectionManager(st.session_state.tracker)
    
    return st.session_state.tws_manager

# Usage example for the main Streamlit app:
"""
# In your main streamlit app, replace the connection button code with:

if st.sidebar.button("🔌 Connect to TWS"):
    tws_manager = get_tws_manager()
    if tws_manager.connect(tws_host, tws_port, client_id):
        st.session_state.is_connected = True
        st.sidebar.success("Connected to TWS!")
    else:
        st.session_state.is_connected = False

if st.sidebar.button("❌ Disconnect"):
    tws_manager = get_tws_manager()
    tws_manager.disconnect()
    st.session_state.is_connected = False
    st.sidebar.info("Disconnected from TWS")
"""