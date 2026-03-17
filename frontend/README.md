# Smart Parking System - Frontend

This is the user interface for the Smart Parking System, built with React and Vite. It provides a real-time, dynamic dashboard showing the availability of parking spots.

## Technology Stack
- **React**: For building the component-based UI.
- **Vite**: As a fast build tool and development server.
- **WebSockets**: To receive live updates from the Python backend instantly without polling.
- **Vanilla CSS**: For styling, utilizing CSS variables, responsive grids, and modern glassmorphic design principles.

## Structure
- `src/App.jsx`: The main component containing the state logic and WebSocket connection. It renders the Dashboard, Status Indicators, and the Parking Grid.
- `src/App.css` and `src/index.css`: The styling files. They define the color palette, animations (like glowing effects when spots are available), and the layout of the parking lot graphic.

## How it Works (The Process)
1. **Initialization**: When the application loads, it initializes with an empty layout and attempts to connect to the backend WebSocket server at `ws://localhost:8000/ws`.
2. **WebSocket Connection**: 
   - While disconnected, it shows a warning and a grayscale status indicator.
   - Once connected, it listens precisely for `onmessage` events from the server.
3. **Real-time Updates**: The Python AI server is constantly analyzing video frames and broadcasting the state of the parking lot (e.g., Spot A1: Occupied, Spot B4: Available) every second.
4. **React State Mapping**: When the frontend receives this JSON payload, it updates its `spots` state array. React automatically re-renders the specific `div` elements representing the parking spots, instantly turning them red/green based on the newest data.

## Running Locally

1. Ensure you have Node.js installed.
2. From this `frontend` directory, install dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
4. Open your browser to the local address provided (usually `http://localhost:5173` or `http://localhost:5174`).
