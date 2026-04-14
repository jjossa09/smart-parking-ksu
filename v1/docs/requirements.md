# Software Requirements Specification
Smart Parking KSU

## 1. Overview

Parking at Kennesaw State University becomes highly congested during peak semesters. Students often spend significant time searching for available parking spaces.

The goal of this project is to develop a system that detects available parking spaces in a parking lot and presents this information through a web interface.

The system will use computer vision and machine learning to classify parking spots as either occupied or available using video input.

---

## 2. Objectives

The main objectives of this project are:

- Detect parking spot occupancy using video input
- Classify parking spots using machine learning
- Display parking availability through a web interface
- Help students quickly identify available parking

---

## 3. System Scope

The first prototype will focus on a single parking lot.

The system will:

- Process a video feed of a parking lot
- Identify individual parking spots
- Classify each spot as occupied (1) or empty (0)
- Display parking availability on a web dashboard

Future versions may support:

- Multiple parking lots
- Live camera feeds
- Parking demand predictions
- Historical parking analytics

---

## 4. Functional Requirements

FR1  
The system shall process video input of a parking lot.

FR2  
The system shall identify predefined parking spot regions within the video frame.

FR3  
The system shall classify each parking spot as either:

- Occupied (1)
- Available (0)

FR4  
The system shall update parking availability periodically.

FR5  
The system shall display a list of parking lots.

FR6  
Each parking lot shall display:

- total number of spots
- number of available spots

FR7  
Users shall be able to select a parking lot in the menu.

FR8  
The system shall display a graphical layout of the parking lot.

FR9  
Parking spots shall be visually represented as:

- Green → available
- Red → occupied

---

## 5. Non-Functional Requirements

NFR1  
The system should update parking availability within 5 seconds.

NFR2  
The classification model should achieve at least 85% accuracy.

NFR3  
The system should support multiple users accessing the website simultaneously.

NFR4  
The system interface should be simple and easy to understand.

---

## 6. Machine Learning Models

The project will experiment with multiple models:

- Random Forest
- Support Vector Machine (SVM) - To compare accuracy

The models will classify cropped images of parking spots as occupied or empty.

---

## 7. Future Features

Possible phase 2 features include:

- Parking availability predictions
- Busiest parking time analysis
- Multiple parking lot monitoring
- Live camera feeds
- Best parking lot recommendations