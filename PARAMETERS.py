LAMBDA_URLLC = 200.0          # packets/s
LAMBDA_EMBB = 3000.0           # packets/s

MU_R = 5000.0                # Node 1 service rate, packets/s
MU_B = 10000.0                # Node 2 service rate, packets/s

T_SIM = 500.0                 # total simulation time, seconds
WARMUP = 2.0                 # warm-up duration, seconds
SEED = 42

URLLC_DELAY_THRESHOLD = 1e-3 # 1 ms
BUFFER_SIZE_NODE1 = 100      # finite capacity at Node 1