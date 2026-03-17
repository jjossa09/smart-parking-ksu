import { useState, useEffect } from 'react';
import './App.css';

// For the hackathon demo, we map the backend IDs to specific X,Y percentage coordinates
// overlaid on top of the KSU map screenshot.
const mapCoordinates = {
  // Lot A Top Row
  "A1": { top: '35%', left: '20%' },
  "A2": { top: '35%', left: '26%' },
  "A3": { top: '35%', left: '32%' },
  "A4": { top: '35%', left: '38%' },
  "A5": { top: '35%', left: '44%' },
  "A6": { top: '35%', left: '50%' },
  "A7": { top: '35%', left: '56%' },
  "A8": { top: '35%', left: '62%' },
  "A9": { top: '35%', left: '68%' },
  "A10": { top: '35%', left: '74%' },
  // Lot B Bottom Row
  "B1": { top: '65%', left: '20%' },
  "B2": { top: '65%', left: '26%' },
  "B3": { top: '65%', left: '32%' },
  "B4": { top: '65%', left: '38%' },
  "B5": { top: '65%', left: '44%' },
  "B6": { top: '65%', left: '50%' },
  "B7": { top: '65%', left: '56%' },
  "B8": { top: '65%', left: '62%' },
  "B9": { top: '65%', left: '68%' },
  "B10": { top: '65%', left: '74%' },
};

function App() {
  const [spots, setSpots] = useState([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Connect to the Python FastAPI WebSocket server
    const ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => {
      console.log('Connected to Smart Parking Server');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'update') {
          // Replace our local react state perfectly with the latest OpenCV data
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

    return () => {
      ws.close();
    };
  }, []);

  const availableSpots = spots.filter(s => !s.isOccupied).length;
  const occupiedSpots = spots.length - availableSpots;

  return (
    <div className="dashboard-container">
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
        <div 
          className="status-indicator" 
          style={{ filter: isConnected ? 'none' : 'grayscale(100%)' }}
        >
          <div className="status-dot"></div>
          {isConnected ? 'LIVE BACKEND CONNECTED' : 'BACKEND DISCONNECTED'}
        </div>
      </header>

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
          <span className="stat-value value-total">{spots.length || 20}</span>
        </div>
      </div>

      <main className="parking-lot-section">
        <h2 className="section-title text-gradient">Campus Lot A - Live Map</h2>
        
        <div className="glass-panel map-container">
           {/* Fallback text if backend is disconnected */}
           {!isConnected && spots.length === 0 && (
             <div className="map-overlay-message">
                <h3>Waiting for Python AI Backend...</h3>
                <p>Ensure `server.py` and `create_mask.py` are running properly.</p>
             </div>
           )}

           {/* The mapped spots overlaid on the generic map container */}
           {spots.map(spot => {
             const coords = mapCoordinates[spot.id];
             if (!coords) return null; // Safety check

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
