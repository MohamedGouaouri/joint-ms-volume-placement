
from graph.base import BaseMicroserviceEdgeEnvironment
import networkx as nx

# Camera Feed Ingestion:
#  - Collects video feeds from traffic cameras placed across intersections and roads.
#  - Streamlining camera data, compressing video, and forwarding relevant frames to analysis services.
#  - Higher bandwidth for video streaming and moderate CPU for pre-processing.

# Traffic Flow Analyzer
#  - Analyzes live video feeds and sensor data to estimate traffic density, flow speed, and congestion points.
#  -  Using machine learning or computer vision models to detect vehicle counts, classify vehicle types, and predict traffic patterns.
#  - High CPU and memory usage due to complex analysis tasks.

# Incident Detector
#  - Identifies accidents, roadblocks, and unusual driving patterns.
#  - Detecting collisions, stalled vehicles, or illegal maneuvers and triggering alerts.
#  - Needs substantial memory and bandwidth for real-time incident detection.

# Signal Controller
#  - Manages traffic lights based on live traffic data and incident reports.
#  - Dynamically adjusting signal timings to optimize flow, prioritize emergency vehicles, or clear congested intersections.
#  - Moderate CPU and memory for real-time signal adjustment logic.


# Parking Availability Manager
#  - Tracks available parking spaces and guides vehicles to free spots.
#  - Integrating with smart parking sensors or camera feeds, providing occupancy updates, and pushing notifications to drivers.
#  - Moderate CPU and bandwidth for frequent space updates.

# Public Transport Tracker
#  - Monitors buses, trains, or trams to provide real-time location data and estimated arrival times.
#  - Collecting GPS data, calculating ETAs, and updating transport schedules dynamically.
#  - Needs CPU for continuous location updates and bandwidth for communication with multiple vehicles.

# Emergency Vehicle Router
#  - Finds the fastest route for ambulances, fire trucks, or police vehicles.
#  - Rerouting traffic signals, calculating alternative paths, and notifying nearby vehicles to clear lanes.
#  - High CPU and memory for route calculation and live traffic analysis.

class SmartTrafficMicroserviceGraph(BaseMicroserviceEdgeEnvironment):
    def __init__(self, num_edge_servers=10, block_size=1):
        super().__init__(num_edge_servers, block_size)
        self.app_name = "Smart Home Microservices"
        self.microservices = {
            "Camera Feed Ingestion": 0,
            "Traffic Flow Analyzer": 1,
            "Incident Detector": 2,
            "Signal Controller": 3,
            "Parking Availability Manager": 4,
            "Public Transport Tracker": 5,
            "Emergency Vehicle Router": 6,
            
        }
        self.microservice_graph = nx.DiGraph()
        self._build_ms_graph()

    def _build_ms_graph(self):
        self._add_service(self.microservices["Camera Feed Ingestion"], 400, 1024, 2, 300)
        self._add_service(self.microservices["Traffic Flow Analyzer"], 600, 1536, 2, 200)
        self._add_service(self.microservices["Incident Detector"], 500, 1024, 1, 150)
        self._add_service(self.microservices["Signal Controller"], 350, 512, 1, 50)
        self._add_service(self.microservices["Parking Availability Manager"], 300, 512, 1, 80)
        self._add_service(self.microservices["Public Transport Tracker"], 450, 768, 2, 120)
        self._add_service(self.microservices["Emergency Vehicle Router"], 500, 1024, 1, 150)
        
        self._add_dependency(self.microservices["Camera Feed Ingestion"], self.microservices["Traffic Flow Analyzer"], 200)
        self._add_dependency(self.microservices["Traffic Flow Analyzer"], self.microservices["Incident Detector"], 150)
        self._add_dependency(self.microservices["Incident Detector"], self.microservices["Signal Controller"], 50)
        self._add_dependency(self.microservices["Camera Feed Ingestion"], self.microservices["Parking Availability Manager"], 80)
        self._add_dependency(self.microservices["Traffic Flow Analyzer"], self.microservices["Public Transport Tracker"], 120)
        self._add_dependency(self.microservices["Incident Detector"], self.microservices["Emergency Vehicle Router"], 150)

