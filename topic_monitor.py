import rclpy
from rclpy.node import Node
from threading import Thread, Lock
import time

class TopicMonitor(Node):
    def __init__(self):
        super().__init__('web_topic_monitor')
        self.topics = []
        self.lock = Lock()
        self.timer = self.create_timer(2.0, self.update_topics)  # Poll every 2 sec

    def update_topics(self):
        with self.lock:
            self.topics = [topic[0] for topic in self.get_topic_names_and_types()]

    def get_topics(self):
        with self.lock:
            return list(self.topics)

# Global instance and spin thread
rclpy.init()
topic_monitor_node = TopicMonitor()

def spin_thread():
    rclpy.spin(topic_monitor_node)

thread = Thread(target=spin_thread, daemon=True)
thread.start()

def get_active_topics():
    return topic_monitor_node.get_topics()
