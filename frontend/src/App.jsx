import { useState, useEffect } from 'react';
import './App.css';

// ==========================================
// 1. KSU PARKING MAP COORDINATES
// ==========================================
// This object acts as the glue between the Artificial Intelligence and the Web UI.
// The backend OpenCV script detects shapes and labels them "A1", "A2", etc.
// This structure maps those arbitrary labels to exact X/Y percentage coordinates 
// on top of the 'ksu_map.png' background image, creating the final real-world overlay.
const mapCoordinates = {
  // Lot A - Left vertical column (Closest to Hornet Village Building 100)
  "A1": { top: '25%', left: '42%' },
  "A2": { top: '30%', left: '42%' },
  "A3": { top: '35%', left: '42%' },
  "A4": { top: '40%', left: '42%' },
  "A5": { top: '45%', left: '42%' },
  "A6": { top: '50%', left: '42%' },
  "A7": { top: '55%', left: '42%' },
  "A8": { top: '60%', left: '42%' },
  "A9": { top: '65%', left: '42%' },
  "A10": { top: '70%', left: '42%' },
  // Lot B - Right vertical column next to Lot A
  "B1": { top: '25%', left: '50%' },
  "B2": { top: '30%', left: '50%' },
  "B3": { top: '35%', left: '50%' },
  "B4": { top: '40%', left: '50%' },
  "B5": { top: '45%', left: '50%' },
  "B6": { top: '50%', left: '50%' },
  "B7": { top: '55%', left: '50%' },
  "B8": { top: '60%', left: '50%' },
  "B9": { top: '65%', left: '50%' },
  "B10": { top: '70%', left: '50%' },
};

// ==========================================
// 2. MAIN REACT COMPONENT
// ==========================================
function App() {
  // React State Hooks:
  // `spots` holds the live array data beamed down from the Python OpenCV AI.
  // Example: [{id: "A1", isOccupied: true}, {id: "A2", isOccupied: false}]
  const [spots, setSpots] = useState([]);
  
  // `isConnected` tracks if the WebSocket successfully tunneled to the Python 8000 port.
  const [isConnected, setIsConnected] = useState(false);

  // The `useEffect` hook runs exactly once when the web app first loads in the browser.
  useEffect(() => {
    // Attempt to open a live, real-time funnel directly to the Python backend
    const ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => {
      console.log('Connected to Smart Parking Server');
      setIsConnected(true);
    };

    // This block fires every single time Python detects a new frame on the CCTV camera (1x/sec).
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'update') {
          // Immediately overwrite the React UI state with the fresh AI payload.
          // This triggers React to re-render the screen instantly.
          setSpots(data.spots);
        }
      } catch (err) {
        console.error("Error parsing websocket message", err);
      }
    };

    ws.onclose = () => {
      console.log('Disconnected from server');
      setIsConnected(false);
    };

    // Cleanup: If the user closes the browser tab, terminate the connection gracefully.
    return () => {
      ws.close();
    };
  }, []); // Empty dependency array means this WebSocket logic only bootstraps once.

  // ==========================================
  // 3. UI DATA CALCULATIONS
  // ==========================================
  // We compute total numbers from the raw array the AI sends us mathematically
  const availableSpots = spots.filter(s => !s.isOccupied).length;
  const occupiedSpots = spots.length - availableSpots;

  // ==========================================
  // 4. BROWSER JSX RENDERING
  // ==========================================
  return (
    <div className="dashboard-container">
      {/* Top Header Navigation */}
      <header className="dashboard-header">
        <div className="logo-container">
          <h1 className="text-gradient">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="url(#gradient)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <defs>
                <linearGradient id="gradient" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stopColor="var(--accent-primary)" />
                  <stop offset="100%" stopColor="var(--accent-secondary)" />
                </linearGradient>
              </defs>
              <rect x="4" y="4" width="16" height="16" rx="2" ry="2" />
              <path d="M9 16V8h3.5a2.5 2.5 0 0 1 0 5H9" />
            </svg>
            KSU Smart Parking
          </h1>
        </div>
        
        {/* Dynamic Status Indicator: Green if connected, Gray if server is dead */}
        <div 
          className="status-indicator" 
          style={{ filter: isConnected ? 'none' : 'grayscale(100%)' }}
        >
          <div className="status-dot"></div>
          {isConnected ? 'LIVE BACKEND CONNECTED' : 'BACKEND DISCONNECTED'}
        </div>
      </header>

      {/* Top Metric Cards (Calculated from state on lines 82-83) */}
      <div className="stats-grid">
        <div className="stat-card glass-panel">
          <span className="stat-label">Available Spots</span>
          <span className="stat-value value-available">{availableSpots}</span>
        </div>
        <div className="stat-card glass-panel">
          <span className="stat-label">Occupied</span>
          <span className="stat-value value-occupied">{occupiedSpots}</span>
        </div>
        <div className="stat-card glass-panel">
          <span className="stat-label">Total Capacity</span>
          {/* Default to 20 spots if the AI payload drops out to prevent UI jumping */}
          <span className="stat-value value-total">{spots.length || 20}</span>
        </div>
      </div>

      {/* Main Map Visualization Area */}
      <main className="parking-lot-section">
        <h2 className="section-title text-gradient">Campus Lot A - Live Map</h2>
        
        {/*
          This container draws the 'ksu_map.png' background. Every spot is drawn as a child of this div
          using `position: absolute` so it can 'float' on top of the real map correctly.
        */}
        <div className="glass-panel map-container">
           {/* Fallback Warning Box if the connection failed or dataset is empty */}
           {!isConnected && spots.length === 0 && (
             <div className="map-overlay-message">
                <h3>Waiting for Python AI Backend...</h3>
                <p>Ensure `server.py` and `create_mask.py` are running properly.</p>
             </div>
           )}

           {/* 
             The Core Loop: For every parking spot the AI says exists, render a dot.
             Lookup its physical X/Y location from `mapCoordinates` (Line 9).
             Color it Green ('available') or Red ('occupied') based on OpenCV analysis.
           */}
           {spots.map(spot => {
             const coords = mapCoordinates[spot.id];
             if (!coords) return null; // Safety check in case the AI emits a spot not on our map

             return (
               <div 
                 key={spot.id} 
                 className={`map-pin ${spot.isOccupied ? 'occupied' : 'available'}`}
                 style={{ top: coords.top, left: coords.left }}
                 title={`Spot ${spot.id} is ${spot.isOccupied ? 'Occupied' : 'Available'}`}
               >
                 <div className="pin-pulse"></div>
                 <span className="pin-label">{spot.id}</span>
               </div>
             );
           })}
        </div>
      </main>
    </div>
  );
}

export default App;
