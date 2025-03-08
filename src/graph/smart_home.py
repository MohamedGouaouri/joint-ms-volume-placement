import networkx as nx
import matplotlib.pyplot as plt
from .base import BaseMicroserviceEdgeEnvironment
import random

class SmartHomeMicroserviceGraph(BaseMicroserviceEdgeEnvironment):
    def __init__(self, num_edge_servers=10, block_size=1):
        super().__init__(num_edge_servers, block_size)
        self.app_name = "Smart Home Microservices"
        self.microservices = {
            "Device Manager": 0,
            "Event Processor": 1,
            "Video Stream Analyzer": 2,
            "Notification Service": 3,
            "Rule Engine": 4,
        }
        self.microservice_graph = nx.DiGraph()
        self._build_ms_graph()


    def _build_ms_graph(self):
        self._add_service(self.microservices["Device Manager"], 300, 512, 1, 50)
        self._add_service(self.microservices["Event Processor"], 400, 768, 2, 75)
        self._add_service(self.microservices["Video Stream Analyzer"], 600, 1024, 2, 200)
        self._add_service(self.microservices["Notification Service"], 200, 512, 1, 20)
        self._add_service(self.microservices["Rule Engine"], 350, 512, 1, 30)
        
        self._add_dependency(self.microservices["Device Manager"], self.microservices["Event Processor"], 50)
        self._add_dependency(self.microservices["Event Processor"], self.microservices["Rule Engine"], 30)
        self._add_dependency(self.microservices["Video Stream Analyzer"], self.microservices["Event Processor"], 100)
        self._add_dependency(self.microservices["Rule Engine"], self.microservices["Notification Service"], 20)
