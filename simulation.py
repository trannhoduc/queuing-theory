import heapq
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass(order=True)
class Event:
    time: float
    priority: int
    event_type: str = field(compare=False)
    packet_id: Optional[int] = field(default=None, compare=False)
    meta: Optional[dict] = field(default=None, compare=False)


@dataclass
class Packet:
    packet_id: int
    traffic_class: str
    t_arrival_net: float

    t_arrival_n1: Optional[float] = None
    t_start_n1: Optional[float] = None
    t_depart_n1: Optional[float] = None

    t_arrival_n2: Optional[float] = None
    t_depart_n2: Optional[float] = None

    dropped: bool = False


class StatsCollector:

    def __init__(self, warmup: float, sim_time: float, urlcc_threshold: float):

        self.warmup = warmup
        self.sim_time = sim_time
        self.measurement_time = max(sim_time - warmup, 1e-12)
        self.urlcc_threshold = urlcc_threshold

        self.arrivals = {"URLLC": 0, "eMBB": 0}
        self.losses = {"URLLC": 0, "eMBB": 0}
        self.completed = {"URLLC": 0, "eMBB": 0}

        self.delay_total = {"URLLC": 0.0, "eMBB": 0.0}
        self.delay_n1 = {"URLLC": 0.0, "eMBB": 0.0}
        self.delay_n2 = {"URLLC": 0.0, "eMBB": 0.0}

        self.wait_n1 = {"URLLC": 0.0, "eMBB": 0.0}

        # ⭐ store all delays for percentile calculation
        self.delay_samples = {"URLLC": [], "eMBB": []}

        self.urlcc_violations = 0

    def _count_packet(self, pkt: Packet):
        return pkt.t_arrival_net >= self.warmup

    def record_arrival(self, pkt: Packet):

        if self._count_packet(pkt):
            self.arrivals[pkt.traffic_class] += 1

    def record_loss(self, pkt: Packet):

        if self._count_packet(pkt):
            self.losses[pkt.traffic_class] += 1

    def record_completion(self, pkt: Packet):

        if not self._count_packet(pkt):
            return

        cls = pkt.traffic_class

        self.completed[cls] += 1

        total_delay = pkt.t_depart_n2 - pkt.t_arrival_net
        node1_delay = pkt.t_depart_n1 - pkt.t_arrival_n1
        node2_delay = pkt.t_depart_n2 - pkt.t_arrival_n2
        node1_wait = pkt.t_start_n1 - pkt.t_arrival_n1

        self.delay_total[cls] += total_delay
        self.delay_n1[cls] += node1_delay
        self.delay_n2[cls] += node2_delay
        self.wait_n1[cls] += node1_wait

        # ⭐ store delay samples
        self.delay_samples[cls].append(total_delay)

        if cls == "URLLC" and total_delay > self.urlcc_threshold:
            self.urlcc_violations += 1

    def results(self):

        def safe_div(a, b):
            return a / b if b > 0 else math.nan

        return {

            "URLLC": {

                "mean_wait_node1": safe_div(self.wait_n1["URLLC"], self.completed["URLLC"]),
                "mean_delay_total": safe_div(self.delay_total["URLLC"], self.completed["URLLC"]),
                "mean_delay_node1": safe_div(self.delay_n1["URLLC"], self.completed["URLLC"]),
                "mean_delay_node2": safe_div(self.delay_n2["URLLC"], self.completed["URLLC"]),
                "delay_violation_prob_1ms": safe_div(self.urlcc_violations, self.completed["URLLC"]),
                "arrivals": self.arrivals["URLLC"],
                "completed": self.completed["URLLC"],
                "losses": self.losses["URLLC"],
            },

            "eMBB": {

                "mean_wait_node1": safe_div(self.wait_n1["eMBB"], self.completed["eMBB"]),
                "mean_delay_total": safe_div(self.delay_total["eMBB"], self.completed["eMBB"]),
                "mean_delay_node1": safe_div(self.delay_n1["eMBB"], self.completed["eMBB"]),
                "mean_delay_node2": safe_div(self.delay_n2["eMBB"], self.completed["eMBB"]),
                "mean_throughput": self.completed["eMBB"] / self.measurement_time,
                "loss_prob": safe_div(self.losses["eMBB"], self.arrivals["eMBB"]),
                "arrivals": self.arrivals["eMBB"],
                "completed": self.completed["eMBB"],
                "losses": self.losses["eMBB"],
            }
        }


class NetworkSimulator:

    def __init__(self,
                 lambda_urllc,
                 lambda_embb,
                 mu_r,
                 mu_b,
                 sim_time,
                 warmup=0,
                 delay_threshold_urllc=1e-3,
                 buffer_size_node1=None,
                 seed=None):

        self.lambda_urllc = lambda_urllc
        self.lambda_embb = lambda_embb
        self.mu_r = mu_r
        self.mu_b = mu_b
        self.sim_time = sim_time
        self.warmup = warmup
        self.delay_threshold_urllc = delay_threshold_urllc
        self.buffer_size_node1 = buffer_size_node1

        self.rng = random.Random(seed)

        self.event_queue = []
        self.next_packet_id = 0
        self.packets = {}

        self.stats = StatsCollector(warmup, sim_time, delay_threshold_urllc)

        self.n1_server_busy = False
        self.n1_current_packet_id = None

        self.n1_q_urllc = deque()
        self.n1_q_embb = deque()

        self.n2_active_packets = set()
        self.n2_departure_version = 0

    def expovariate(self, rate):

        return self.rng.expovariate(rate)

    def schedule_event(self, time, priority, event_type, packet_id=None, meta=None):

        heapq.heappush(
            self.event_queue,
            Event(time, priority, event_type, packet_id, meta)
        )

    def create_packet(self, traffic_class, t_now):

        pkt = Packet(self.next_packet_id, traffic_class, t_now)
        self.packets[pkt.packet_id] = pkt
        self.next_packet_id += 1

        return pkt

    def node1_population(self):

        in_service = 1 if self.n1_server_busy else 0

        return in_service + len(self.n1_q_urllc) + len(self.n1_q_embb)

    def node1_has_capacity(self):

        if self.buffer_size_node1 is None:
            return True

        return self.node1_population() < self.buffer_size_node1

    def start_service_node1(self, packet_id, t_now):

        self.n1_server_busy = True
        self.n1_current_packet_id = packet_id

        pkt = self.packets[packet_id]
        pkt.t_start_n1 = t_now

        service_time = self.expovariate(self.mu_r)

        self.schedule_event(t_now + service_time, 1, "depart_node1", packet_id)

    def try_start_next_node1_service(self, t_now):

        if self.n1_server_busy:
            return

        if self.n1_q_urllc:
            self.start_service_node1(self.n1_q_urllc.popleft(), t_now)

        elif self.n1_q_embb:
            self.start_service_node1(self.n1_q_embb.popleft(), t_now)

    def handle_external_arrival(self, traffic_class, t_now):

        pkt = self.create_packet(traffic_class, t_now)
        pkt.t_arrival_n1 = t_now

        self.stats.record_arrival(pkt)

        if not self.node1_has_capacity():

            pkt.dropped = True
            self.stats.record_loss(pkt)

            return

        if not self.n1_server_busy:

            self.start_service_node1(pkt.packet_id, t_now)

        else:

            if traffic_class == "URLLC":
                self.n1_q_urllc.append(pkt.packet_id)
            else:
                self.n1_q_embb.append(pkt.packet_id)

    def handle_depart_node1(self, packet_id, t_now):

        if self.n1_current_packet_id != packet_id:
            return

        pkt = self.packets[packet_id]
        pkt.t_depart_n1 = t_now

        self.n1_current_packet_id = None
        self.n1_server_busy = False

        self.handle_arrival_node2(packet_id, t_now)

        self.try_start_next_node1_service(t_now)

    def reschedule_node2_departure(self, t_now):

        self.n2_departure_version += 1
        version = self.n2_departure_version

        n = len(self.n2_active_packets)

        if n == 0:
            return

        dt = self.expovariate(n * self.mu_b)

        self.schedule_event(t_now + dt, 2, "depart_node2", meta={"version": version})

    def handle_arrival_node2(self, packet_id, t_now):

        pkt = self.packets[packet_id]
        pkt.t_arrival_n2 = t_now

        self.n2_active_packets.add(packet_id)

        self.reschedule_node2_departure(t_now)

    def handle_depart_node2(self, t_now, version):

        if version != self.n2_departure_version:
            return

        n = len(self.n2_active_packets)

        if n == 0:
            return

        departing_id = self.rng.choice(tuple(self.n2_active_packets))

        self.n2_active_packets.remove(departing_id)

        pkt = self.packets[departing_id]
        pkt.t_depart_n2 = t_now

        self.stats.record_completion(pkt)

        self.reschedule_node2_departure(t_now)

    def schedule_initial_arrivals(self):

        self.schedule_event(self.expovariate(self.lambda_urllc),0,"arrival_urllc")
        self.schedule_event(self.expovariate(self.lambda_embb),0,"arrival_embb")

    def schedule_next_arrival(self, traffic_class, t_now):

        if traffic_class == "URLLC":

            next_t = t_now + self.expovariate(self.lambda_urllc)

            if next_t <= self.sim_time:
                self.schedule_event(next_t,0,"arrival_urllc")

        else:

            next_t = t_now + self.expovariate(self.lambda_embb)

            if next_t <= self.sim_time:
                self.schedule_event(next_t,0,"arrival_embb")

    def run(self):

        self.schedule_initial_arrivals()

        while self.event_queue:

            event = heapq.heappop(self.event_queue)

            if event.time > self.sim_time:
                break

            t_now = event.time

            if event.event_type == "arrival_urllc":

                self.handle_external_arrival("URLLC", t_now)
                self.schedule_next_arrival("URLLC", t_now)

            elif event.event_type == "arrival_embb":

                self.handle_external_arrival("eMBB", t_now)
                self.schedule_next_arrival("eMBB", t_now)

            elif event.event_type == "depart_node1":

                self.handle_depart_node1(event.packet_id, t_now)

            elif event.event_type == "depart_node2":

                self.handle_depart_node2(t_now, event.meta["version"])

        return self.stats.results()