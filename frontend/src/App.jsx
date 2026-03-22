import { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [lotData, setLotData] = useState({
    lotName: "Campus Lot A - Live Map",
    frameImage: null,
    totalSpots: 0,
    availableSpots: 0,
    occupiedSpots: 0,
    spots: [],
    timestamp: null
  });
  
  const [isConnected, setIsConnected] = useState(false);
  const [imgDim, setImgDim] = useState({ w: 1280, h: 720 });
  const [hoveredSpot, setHoveredSpot] = useState(null);

  useEffect(() => {
    // Connect to the local FastAPI WebSocket
    const wsUrl = 'ws://localhost:8000/ws';
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('Connected to Smart Parking Server');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'update') {
          setLotData({
            lotName: data.lotName || "Parking Lot",
            frameImage: data.frameImage,
            totalSpots: data.totalSpots || 0,
            availableSpots: data.availableSpots || 0,
            occupiedSpots: data.occupiedSpots || 0,
            spots: data.spots || [],
            timestamp: data.timestamp
          });
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

  const handleImageLoad = (e) => {
    // This allows the SVG viewBox to match the image's original coordinate system perfectly!
    // We only update if dimensions are truly different to prevent infinite looping re-renders 
    // since the img src base64 gets updated 10 times a second perfectly synced with WebSockets.
    if (imgDim.w !== e.target.naturalWidth || imgDim.h !== e.target.naturalHeight) {
      setImgDim({
        w: e.target.naturalWidth,
        h: e.target.naturalHeight
      });
    }
  };

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
          {isConnected ? 'LIVE INTERFACE' : 'BACKEND DISCONNECTED'}
        </div>
      </header>

      <div className="stats-grid">
        <div className="stat-card glass-panel">
          <span className="stat-label">Available Spaces</span>
          <span className="stat-value value-available">{lotData.availableSpots}</span>
        </div>
        <div className="stat-card glass-panel">
          <span className="stat-label">Occupied</span>
          <span className="stat-value value-occupied">{lotData.occupiedSpots}</span>
        </div>
        <div className="stat-card glass-panel">
          <span className="stat-label">Total Capacity</span>
          <span className="stat-value value-total">{lotData.totalSpots}</span>
        </div>
      </div>

      <main className="parking-lot-section">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 className="section-title text-gradient">{lotData.lotName}</h2>
          {lotData.timestamp && (
            <span style={{color: 'var(--text-secondary)', fontSize: '0.875rem'}}>
              Data Last Updated: {new Date(lotData.timestamp).toLocaleTimeString()}
            </span>
          )}
        </div>
        
        {!isConnected && lotData.spots.length === 0 ? (
           <div className="glass-panel" style={{padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)'}}>
              <h3>Waiting for Python AI Backend...</h3>
              <p>Please ensure backend is running using `uvicorn app.main:app` and ML predictions have started.</p>
           </div>
        ) : (
          <div className="map-container glass-panel">
            {lotData.frameImage ? (
              <img 
                src={lotData.frameImage} 
                className="parking-image" 
                alt="Live Parking Lot Feed" 
                onLoad={handleImageLoad}
              />
            ) : (
              <div className="image-placeholder">Loading Camera Feed...</div>
            )}
            
            <svg 
              className="map-overlay" 
              viewBox={`0 0 ${imgDim.w} ${imgDim.h}`}
              preserveAspectRatio="xMidYMid slice"
            >
              <defs>
                <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
              </defs>

              {lotData.spots.map(spot => {
                // Formatting [[x1,y1], [x2,y2]] into "x1,y1 x2,y2"
                const pointsStr = spot.polygon.map(coord => coord.join(',')).join(' ');
                
                // Calculate centroid for placing the label beautifully in the center of the polygon
                let cx = 0; let cy = 0;
                spot.polygon.forEach(coord => { cx += coord[0]; cy += coord[1]; });
                cx /= spot.polygon.length;
                cy /= spot.polygon.length;

                return (
                  <g key={spot.id}>
                    <polygon 
                      points={pointsStr}
                      className={`spot-polygon ${spot.isOccupied ? 'occupied' : 'available'}`}
                      onMouseEnter={() => setHoveredSpot(spot.id)}
                      onMouseLeave={() => setHoveredSpot(null)}
                    />
                    
                    {/* Render ID Label if available, or if hovered */}
                    {!spot.isOccupied && (
                      <text 
                        x={cx} 
                        y={cy} 
                        className="spot-label-svg"
                        dominantBaseline="middle"
                        textAnchor="middle"
                      >
                        {spot.label}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>

            {hoveredSpot && (
               <div className="hover-toast glass-panel">
                 <span style={{fontWeight: 700}}>Spot {hoveredSpot}</span>: 
                 <span style={{marginLeft: '0.5rem', color: lotData.spots.find(s => s.id === hoveredSpot)?.isOccupied ? 'var(--status-occupied)' : 'var(--status-available)'}}>
                     {lotData.spots.find(s => s.id === hoveredSpot)?.isOccupied ? 'Occupied 🚗' : 'Available ✨'}
                 </span>
               </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
