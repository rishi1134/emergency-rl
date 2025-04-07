def reward_fn(traffic_signal):
    return -1 * traffic_signal.get_total_queued() 