def reward_fn(traffic_signal):
    return -1 * traffic_signal.get_total_queued() 

def emergency_reward_fn(traffic_signal):
    alpha = 0.6
    reward = 0
    vehicle_ids = traffic_signal.sumo.vehicle.getIDList()
    if len(vehicle_ids) != 0:
        for veh_id in vehicle_ids:
            if traffic_signal.sumo.vehicle.getTypeID(veh_id) == "rescue":                  
                laneID = traffic_signal.sumo.vehicle.getLaneID(veh_id)
                queue_length = traffic_signal.sumo.lane.getLastStepVehicleNumber(laneID)
                reward += (-1 * queue_length)
    # print(alpha * reward, (-1 * ((1 - alpha) * traffic_signal.get_total_queued())))
    # print((alpha * reward) + (-1 * ((1 - alpha) * traffic_signal.get_total_queued())), alpha * reward, (-1 * ((1 - alpha) * traffic_signal.get_total_queued())))
    if reward == 0:
        # print(-1 * traffic_signal.get_total_queued())
        max_q = 0
        n_t = 0
        w_t = 0
        for lane in traffic_signal.lanes:
            if "n_t" in lane:
                n_t += traffic_signal.sumo.lane.getLastStepVehicleNumber(lane)
            if "w_t" in lane:
                w_t += traffic_signal.sumo.lane.getLastStepVehicleNumber(lane)
        return -1 * (traffic_signal.get_total_queued() + max(n_t, w_t))
        # return -1 * (max(n_t, w_t))
    # print((alpha * reward) + (-1 * ((1 - alpha) * traffic_signal.get_total_queued())))
    # return (alpha * reward) + (-1 * ((1 - alpha) * traffic_signal.get_total_queued()))
    # print(reward)
    return reward