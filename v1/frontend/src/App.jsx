import { useState, useEffect } from 'react';
import './App.css';

// We start with an empty layout and let the server populate it.
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
          <span className="stat-value value-total">{spots.length}</span>
        </div>
      </div>

      <main className="parking-lot-section">
        <h2 className="section-title text-gradient">Campus Lot A - Live Map</h2>
        
        {!isConnected && spots.length === 0 ? (
           <div className="glass-panel" style={{padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)'}}>
              <h3>Waiting for Python AI Backend...</h3>
              <p>Please ensure `server.py` is running to send video detection data.</p>
           </div>
        ) : (
          <div className="glass-panel parking-grid">
            <div className="parking-row">
              {spots.slice(0, 10).map(spot => (
                <div key={spot.id} className={`parking-spot ${spot.isOccupied ? 'occupied' : 'available'}`}>
                  <span className="spot-label">{spot.id}</span>
                  <span className="spot-icon">{spot.isOccupied ? '🚗' : '✨'}</span>
                </div>
              ))}
            </div>

            <div className="roadway"></div>

            <div className="parking-row">
              {spots.slice(10, 20).map(spot => (
                <div key={spot.id} className={`parking-spot ${spot.isOccupied ? 'occupied' : 'available'}`}>
                  <span className="spot-label">{spot.id}</span>
                  <span className="spot-icon">{spot.isOccupied ? '🚗' : '✨'}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
