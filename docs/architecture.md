# System Architecture
Smart Parking KSU

## 1. System Overview

The Smart Parking KSU system is designed to help students quickly identify available parking spots in campus parking lots. Parking at Kennesaw State University can become congested during peak semesters, causing students to spend significant time searching for parking.

This system uses computer vision and machine learning to analyze video input from a parking lot and determine whether individual parking spots are occupied or available.

The processed information is then displayed through a web interface where users can see parking availability and visualize the parking lot layout.

The architecture focuses on a simple and modular design suitable for a prototype implementation while remaining scalable for future expansion.

---

## 2. Architecture Style

The system follows a **layered architecture** pattern. Each layer has a specific responsibility and communicates with adjacent layers.

The main layers are:

- **Presentation Layer** – User interface displayed in the web application.
- **Application Layer** – Backend server handling logic and communication.
- **Computer Vision / Machine Learning Layer** – Processes video input and performs parking spot classification.
- **Data State Layer** – Maintains the current parking spot status.

This separation ensures the system remains modular, maintainable, and scalable.

---

## 3. System Architecture Diagram

The following diagram illustrates the high-level architecture of the system.

```
+---------------------------+
|        User Browser       |
|        (Frontend)         |
|       React Website       |
+------------+--------------+
             |
             | HTTP Request
             v
+---------------------------+
|        Backend API        |
|        (Python)           |
|        Flask Server       |
+------------+--------------+
             |
             | Calls ML Module
             v
+---------------------------+
|  Computer Vision Pipeline |
|          OpenCV           |
|  Parking Spot Extraction  |
+------------+--------------+
             |
             v
+---------------------------+
|  Machine Learning Model   |
| Random Forest / SVM / CNN |
|   Binary Classification   |
+------------+--------------+
             |
             v
+---------------------------+
|     Parking Spot State    |
|   (Available / Occupied)  |
+------------+--------------+
             |
             v
+---------------------------+
|        Frontend UI        |
| Parking Lot Visualization |
+---------------------------+
```
---

## 4. System Components

### 4.1 Frontend (Web Interface)

The frontend is responsible for displaying parking information to the user.

Users will interact with a website that provides:

- A list of available parking lots
- The number of available parking spots in each lot
- The total number of parking spots
- A visual layout of the parking lot

Parking spots will be represented visually using colors:

- **Green** – Available parking spot
- **Red** – Occupied parking spot

The frontend will periodically request updates from the backend API to display current parking availability.

---

### 4.2 Backend API

The backend server acts as the central controller of the system.

Responsibilities include:

- Receiving requests from the frontend
- Processing parking data
- Running the computer vision and machine learning pipeline
- Returning parking availability data to the frontend

The backend will expose endpoints such as:

`GET/parking-status`

Example response:

```
{
"lot_name": "Central Lot",
"total_spots": 20,
"available_spots": 7,
"spots": [0,1,1,0,0,1,1,0,0,1]
}
```

Where:

- `0` represents an available spot
- `1` represents an occupied spot

---

### 4.3 Computer Vision Module

The computer vision module processes frames extracted from the parking lot video.

Responsibilities:

- Capture frames from the parking lot video feed
- Identify predefined parking spot regions
- Extract image crops corresponding to each parking spot
- Send cropped images to the machine learning classifier

This module will be implemented using **OpenCV**.

Example pipeline:
```
Video Frame
↓
Detect parking spot regions
↓
Crop each parking spot
↓
Send image to classifier
```


---

### 4.4 Machine Learning Classifier

The machine learning component performs binary classification for each parking spot image.

The classifier determines whether a parking spot is:

- **Empty**
- **Occupied**

Multiple models will be tested and evaluated for performance:

- Random Forest
- Support Vector Machine (SVM)
- Convolutional Neural Network (CNN)

Each cropped parking spot image will be processed and classified to produce a binary output.

Example output:
```
Spot 1 → 0 (empty)
Spot 2 → 1 (occupied)
Spot 3 → 0 (empty)
Spot 4 → 1 (occupied)
```

The results are then stored in the parking spot state.

---

### 4.5 Parking State Data

The system maintains the current parking spot occupancy state.

This state contains:

- Total number of spots
- Number of available spots
- Occupancy status for each spot

Example structure:
```
Parking Lot A

Total Spots: 10

Spot Status:
[0,1,0,1,1,0,0,1,0,1]
```

Where:

- `0` = Available
- `1` = Occupied

This information is updated continuously as the video feed is processed.

---

## 5. Data Flow

The system processes parking data through the following sequence:

### Step 1 – Video Input

A video of a parking lot is used as the input source.

The system captures frames from this video feed.
```
Parking Lot Video → Frame Capture
```

---

### Step 2 – Parking Spot Detection

Each frame is processed to extract predefined parking spot regions.

```
Frame → Crop Spot 1
Frame → Crop Spot 2
Frame → Crop Spot 3
```

---

### Step 3 – Machine Learning Classification

Each cropped parking spot image is passed to the classifier.
```
Spot Image → ML Model → Prediction
```

The classifier outputs a binary result indicating whether the spot is occupied.

---

### Step 4 – Parking State Update

The system updates the internal parking state based on predictions.
```
Spot Predictions → Update Parking State
```

Example: 
```
[0,1,0,1,0,0,1,0]
```


---

### Step 5 – Frontend Request

The frontend periodically requests updated parking information from the backend.

```
GET /parking-status
```

---

### Step 6 – User Interface Update

The frontend displays updated parking availability and visualizes the lot layout.
```
Available Spots → Green
Occupied Spots → Red
```

---

## 6. Technology Stack

### Frontend

- React
- Tailwind CSS

### Backend

- Python
- Flask

### Computer Vision

- OpenCV

### Machine Learning

- Random Forest
- Support Vector Machine (SVM)
- Convolutional Neural Network (CNN)

---

## 7. Future Improvements

Future versions of the system may include additional capabilities such as:

- Monitoring multiple parking lots simultaneously
- Integration with live camera feeds
- Predicting parking availability based on historical data
- Identifying busiest parking periods
- Providing recommendations for the best parking lot

These improvements would enhance system scalability and usability for a full campus deployment.


