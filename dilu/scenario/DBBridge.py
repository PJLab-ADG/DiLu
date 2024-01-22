import sqlite3
import numpy as np
from typing import List
from highway_env.envs import AbstractEnv
from highway_env.road.road import RoadNetwork, LaneIndex
from highway_env.road.lane import StraightLane, CircularLane
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle


class DBBridge:
    def __init__(self, database: str, env: AbstractEnv) -> None:
        self.database = database
        self.env = env
        self.ego: MDPVehicle = env.vehicle
        self.network: RoadNetwork = env.road.network

    def createTable(self):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS simINFO(
                envType TEXT,
                seed INT
            );"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS networkINFO(
                laneIndexO TEXT,
                laneIndexD TEXT,
                laneIndexI INT,
                laneType TEXT,
                wayPoint TEXT,
                width REAL,
                speedLimit REAL,
                PRIMARY KEY (laneIndexO, laneIndexD, laneIndexI)
            );"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS vehINFO(
                decisionFrame INT,
                vehicleID INT,
                length REAL,
                width REAL,
                posx REAL,
                posy REAL,
                speed REAL,
                acceleration REAL,
                heading REAL,
                steering REAL,
                laneIndexO TEXT,
                laneIndexD TEXT,
                laneIndexI INT
            );"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS promptsINFO(
                decisionFrame INT PRIMARY KEY,
                vectorID TEXT,
                done BOOL,
                description TEXT,
                fewshots TEXT,
                thoughtsAndAction TEXT,
                editedTA TEXT,
                editTimes INT
            );"""
        )
        conn.commit()
        conn.close()

    def insertSimINFO(self, envType: str, seed: int):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO simINFO(envType, seed) VALUES(?,?);""",
            (envType, seed),
        )
        conn.commit()
        conn.close()

    def getCicularLaneWayPoint(self, cl: CircularLane):
        if cl.direction == 1:
            start_radian, end_radian = cl.end_phase, cl.start_phase
        else:
            start_radian, end_radian = cl.start_phase, cl.end_phase
        theta = np.linspace(start_radian, end_radian, num=50)
        x = cl.center[0] + cl.radius * np.cos(theta)
        y = cl.center[1] + cl.radius * np.sin(theta)
        return ' '.join([f'{x[i]},{y[i]}' for i in range(len(x))])

    def insertNetwork(self):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        for k1, v1 in self.network.graph.items():
            for k2, v2 in v1.items():
                for k3, lane in enumerate(v2):
                    if isinstance(lane, StraightLane):
                        wayPoint = f'{lane.start[0]},{lane.start[1]} {lane.end[0]},{lane.end[1]}'
                        cur.execute(
                            """INSERT INTO networkINFO (
                                laneIndexO, laneIndexD, laneIndexI, laneType, 
                                wayPoint, width, speedLimit
                                ) VALUES (?,?,?,?,?,?,?);""",
                            (
                                k1, k2, k3,
                                "StraightLane",
                                wayPoint, lane.width, lane.speed_limit
                            )
                        )
                    elif isinstance(lane, CircularLane):
                        wayPoint = self.getCicularLaneWayPoint(lane)
                        cur.execute(
                            """INSERT INTO networkINFO(
                                laneIndexO, laneIndexD, laneIndexI, laneType, 
                                wayPoint, width, speedLimit
                                ) VALUES (?,?,?,?,?,?,?);""",
                            (
                                k1, k2, k3, "CircularLane",
                                wayPoint, lane.width, lane.speed_limit
                            )
                        )
                    else:
                        raise NotImplementedError('Lane type not implemented')
        conn.commit()
        conn.close()

    def insertVehicle(self, decisionFrame: int, SVs: List[IDMVehicle]):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        ek1, ek2, ek3 = self.ego.lane_index
        cur.execute(
            """INSERT INTO vehINFO (
                decisionFrame, vehicleID, length, width, posx, posy, speed, 
                acceleration, heading, steering, 
                laneIndexO, laneIndexD, laneIndexI
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?);""",
            (
                decisionFrame, 'ego',
                self.ego.LENGTH, self.ego.WIDTH,
                self.ego.position[0], self.ego.position[1],
                self.ego.speed, self.ego.action['acceleration'],
                self.ego.heading, self.ego.action['steering'],
                ek1, ek2, ek3
            )
        )
        for sv in SVs:
            k1, k2, k3 = sv.lane_index
            cur.execute(
                """INSERT INTO vehINFO (
                    decisionFrame, vehicleID, length, width, posx, posy, speed, 
                    acceleration, heading, steering, 
                    laneIndexO, laneIndexD, laneIndexI
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?);""",
                (
                    decisionFrame, id(sv) % 1000,
                    sv.LENGTH, sv.WIDTH,
                    sv.position[0], sv.position[1],
                    sv.speed, sv.action['acceleration'],
                    sv.heading, sv.action['steering'],
                    k1, k2, k3
                )
            )
        conn.commit()
        conn.close()

    def insertPrompts(
            self, decisionFrame: int, vectorID: str, done: bool,
            description: str, fewshots: str, thoughtsAndAction: str
    ):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO promptsINFO (
                decisionFrame, vectorID, done, description, fewshots, 
                thoughtsAndAction, editedTA, editTimes
                ) VALUES (?,?,?,?,?,?,?,?);""",
            (
                decisionFrame, vectorID, done, description,
                fewshots, thoughtsAndAction, None, 0
            )
        )
        conn.commit()
        conn.close()
