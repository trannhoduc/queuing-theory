# Queuing Theory Project

In this project, you will model a simplified 5G access system consisting of:
1. Radio access node (gNB): where URLLC and eMBB packets arrive and contend for
transmission. URLLC is given strict non-preemptive priority.
2. Backhaul / MEC node: which forwards traffic to the core network. This is modeled as a
Processor Sharing server, appropriate for multiplexed flows.
The goal is to combine analytic derivations and simulation to assess whether URLLC requirements can be met under given traffic loads, and to understand how eMBB performance is affected
by prioritization. By the end of this project, you will have carried out a small but complete research study — from modeling and derivation to simulation and interpretation.

## System Model.
- Arrivals: Independent Poisson processes for URLLC and eMBB with rates λU and λE.
- Node 1 (Radio): M/M/1 with strict non-preemptive priority for URLLC over eMBB. Service
rate µR.
- Node 2 (Backhaul/MEC): M/M/1 with Processor Sharing (PS). Service rate µB.
- Routing: All traffic passes through Node 1 → Node 2, then exits.
- Buffers: Analyze both infinite-buffer and finite-buffer (capacity B = 100 packets) cases at
Node 1.
