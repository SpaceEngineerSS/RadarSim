import time
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.network_manager import NetworkManager, RadarNode, NetworkTrack

def run_benchmark():
    print("=" * 50)
    print("NetworkManager Performance Benchmark")
    print("=" * 50)

    # Initialize a 5-node network
    manager = NetworkManager()
    
    # Create 5 radar nodes
    nodes = []
    node_ids = []
    for i in range(5):
        nid = f"radar_{i}"
        node = manager.register_node(nid, position_xy=np.array([i*1000, i*1000]))
        nodes.append(node)
        node_ids.append(nid)

    # Generate random tracks for each node to simulate a busy environment
    # 20 tracks per node = 100 tracks total to associate
    for i, nid in enumerate(node_ids):
        tracks = []
        for j in range(20):
            track = NetworkTrack(
                track_id=f"{nid}_track_{j}",
                node_id=nid,
                state=np.array([np.random.uniform(-50000, 50000), 
                                np.random.uniform(-50000, 50000), 
                                np.random.uniform(-300, 300), 
                                np.random.uniform(-300, 300)]),
                covariance=np.eye(4) * 100,
                timestamp=time.time(),
                snr_db=20.0
            )
            tracks.append(track)
        manager.submit_tracks(nid, tracks, current_time=1.0)

    print("Simulating T2TA for 5-node network (100 tracks total)...")
    
    # Run association 100 times to get average latency
    latencies = []
    for _ in range(100):
        start_time = time.time()
        # Trigger fusion/association logic
        manager.fuse(current_time=1.0)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # ms

    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Max Latency:     {max_latency:.2f} ms")
    
    if avg_latency > 10.0:
        print("\nWARNING: Latency exceeds 10ms threshold!")
        sys.exit(1)
    else:
        print("\nSUCCESS: Latency is within 10ms threshold.")
        sys.exit(0)

if __name__ == "__main__":
    run_benchmark()
