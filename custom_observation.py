'''
https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/observations.py
'''

# get_total_queued

import numpy as np
from gymnasium import spaces

from sumo_rl.environment.observations import TrafficSignal
from sumo_rl.environment.observations import ObservationFunction


class CustomObservationFunction(ObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        lanes_queue = [
            self.ts.sumo.lane.getLastStepVehicleNumber(lane)
            for lane in self.ts.lanes
        ]
        current_phase_index = [self.ts.sumo.trafficlight.getPhase(self.ts.id)]
        observation = np.array(lanes_queue + current_phase_index, dtype=np.int32)
        return observation

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=np.array([0]*self.ts.lanes + [0]),
            high=np.array([self.ts.lanes_length[lane] / (self.ts.MIN_GAP + 5.0) for lane in self.ts.lanes] + [3]), 
            dtype=np.int32
        )


class CustomEmergencyObservationFunction(ObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        lanes_queue = [
            self.ts.sumo.lane.getLastStepVehicleNumber(lane)
            for lane in self.ts.lanes
        ]
        current_phase_index = [self.ts.sumo.trafficlight.getPhase(self.ts.id)]
        emer_lane = [0]*len(self.ts.lanes)

        vehicle_ids = self.ts.sumo.vehicle.getIDList()
        if len(vehicle_ids) != 0:
            for veh_id in vehicle_ids:
                laneID = self.ts.sumo.vehicle.getLaneID(veh_id)
                if self.ts.sumo.vehicle.getTypeID(veh_id) == "rescue" and laneID in self.ts.lanes:        
                    laneID = self.ts.lanes.index(laneID)
                    emer_lane[laneID] = 1
        else:
            emer_lane = [0]*len(self.ts.lanes)
        observation = np.array(lanes_queue + current_phase_index + emer_lane, dtype=np.int32)
        print(observation)
        return observation

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=np.array([0]*len(self.ts.lanes) + [0] + [0]*len(self.ts.lanes)),
            high=np.array([self.ts.lanes_length[lane] / (self.ts.MIN_GAP + 5.0) for lane in self.ts.lanes] + [3] + [1]*len(self.ts.lanes)), 
            dtype=np.int32
        )