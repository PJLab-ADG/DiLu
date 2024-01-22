from typing import List, Tuple, Optional, Union, Dict
from datetime import datetime
import math
import os

from highway_env.road.road import Road, RoadNetwork, LaneIndex
from highway_env.road.lane import (
    StraightLane, CircularLane, SineLane, PolyLane, PolyLaneFixedWidth
)
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle
import numpy as np

from dilu.scenario.DBBridge import DBBridge
from dilu.scenario.envPlotter import ScePlotter


ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

ACTIONS_DESCRIPTION = {
    0: 'Turn-left - change lane to the left of the current lane',
    1: 'IDLE - remain in the current lane with current speed',
    2: 'Turn-right - change lane to the right of the current lane',
    3: 'Acceleration - accelerate the vehicle',
    4: 'Deceleration - decelerate the vehicle'
}


class EnvScenario:
    def __init__(
            self, env: AbstractEnv, envType: str,
            seed: int, database: str = None
    ) -> None:
        self.env = env
        self.envType = envType

        self.ego: MDPVehicle = env.vehicle
        # 下面的四个变量用来判断车辆是否在 ego 的危险视距内
        self.theta1 = math.atan(3/17.5)
        self.theta2 = math.atan(2/2.5)
        self.radius1 = np.linalg.norm([3, 17.5])
        self.radius2 = np.linalg.norm([2, 2.5])

        self.road: Road = env.road
        self.network: RoadNetwork = self.road.network

        self.plotter = ScePlotter()
        if database:
            self.database = database
        else:
            self.database = datetime.strftime(
                datetime.now(), '%Y-%m-%d_%H-%M-%S'
            ) + '.db'

        if os.path.exists(self.database):
            os.remove(self.database)

        self.dbBridge = DBBridge(self.database, env)

        self.dbBridge.createTable()
        self.dbBridge.insertSimINFO(envType, seed)
        self.dbBridge.insertNetwork()

    def getSurrendVehicles(self, vehicles_count: int) -> List[IDMVehicle]:
        return self.road.close_vehicles_to(
            self.ego, self.env.PERCEPTION_DISTANCE,
            count=vehicles_count-1, see_behind=True,
            sort='sorted'
        )

    def plotSce(self, fileName: str) -> None:
        SVs = self.getSurrendVehicles(10)
        self.plotter.plotSce(self.network, SVs, self.ego, fileName)

    def getUnitVector(self, radian: float) -> Tuple[float]:
        return (
            math.cos(radian), math.sin(radian)
        )

    def isInJunction(self, vehicle: Union[IDMVehicle, MDPVehicle]) -> float:
        if self.envType == 'intersection-v1':
            x, y = vehicle.position
            # 这里交叉口的范围是 -12~12, 这里是为了保证车辆可以检测到交叉口内部的信息
            # 这个时候车辆需要提前减速
            if -20 <= x <= 20 and -20 <= y <= 20:
                return True
            else:
                return False
        else:
            return False

    def getLanePosition(self, vehicle: Union[IDMVehicle, MDPVehicle]) -> float:
        currentLaneIdx = vehicle.lane_index
        currentLane = self.network.get_lane(currentLaneIdx)
        if not isinstance(currentLane, StraightLane):
            raise ValueError(
                "The vehicle is in a junction, can't get lane position"
            )
        else:
            currentLane = self.network.get_lane(vehicle.lane_index)
            return np.linalg.norm(vehicle.position - currentLane.start)

    def availableActionsDescription(self) -> str:
        avaliableActionDescription = 'Your available actions are: \n'
        availableActions = self.env.get_available_actions()
        for action in availableActions:
            avaliableActionDescription += ACTIONS_DESCRIPTION[action] + ' Action_id: ' + str(
                action) + '\n'
        # if 1 in availableActions:
        #     avaliableActionDescription += 'You should check IDLE action as FIRST priority. '
        # if 0 in availableActions or 2 in availableActions:
        #     avaliableActionDescription += 'For change lane action, CAREFULLY CHECK the safety of vehicles on target lane. '
        # if 3 in availableActions:
        #     avaliableActionDescription += 'Consider acceleration action carefully. '
        # if 4 in availableActions:
        #     avaliableActionDescription += 'The deceleration action is LAST priority. '
        # avaliableActionDescription += '\n'
        return avaliableActionDescription

    def processNormalLane(self, lidx: LaneIndex) -> str:
        sideLanes = self.network.all_side_lanes(lidx)
        numLanes = len(sideLanes)
        if numLanes == 1:
            description = "You are driving on a road with only one lane, you can't change lane. "
        else:
            egoLaneRank = lidx[2]
            if egoLaneRank == 0:
                description = f"You are driving on a road with {numLanes} lanes, and you are currently driving in the leftmost lane. "
            elif egoLaneRank == numLanes - 1:
                description = f"You are driving on a road with {numLanes} lanes, and you are currently driving in the rightmost lane. "
            else:
                laneRankDict = {
                    1: 'second',
                    2: 'third',
                    3: 'fourth'
                }
                description = f"You are driving on a road with {numLanes} lanes, and you are currently driving in the {laneRankDict[egoLaneRank]} lane from the left. "

        description += f"Your current position is `({self.ego.position[0]:.2f}, {self.ego.position[1]:.2f})`, speed is {self.ego.speed:.2f} m/s, acceleration is {self.ego.action['acceleration']:.2f} m/s^2, and lane position is {self.getLanePosition(self.ego):.2f} m.\n"
        return description

    def getSVRelativeState(self, sv: IDMVehicle) -> str:
        # CAUTION: 这里有一个问题，pygame 的 y 轴是上下颠倒的，向下是 y 轴的正方向。
        #       因此，在 highway-v0 上，车辆向左换道实际上是向右运动。因此判断车辆相
        #       对自车的位置，不能用向量来算，直接根据车辆在哪条车道上来判断是比较合适
        #       的，向量只能用来判断车辆在 ego 的前方还是后方
        relativePosition = sv.position - self.ego.position
        egoUnitVector = self.getUnitVector(self.ego.heading)
        cosineValue = sum(
            [x*y for x, y in zip(relativePosition, egoUnitVector)]
        )
        if cosineValue >= 0:
            return 'is ahead of you'
        else:
            return 'is behind of you'

    def getVehDis(self, veh: IDMVehicle):
        posA = self.ego.position
        posB = veh.position
        distance = np.linalg.norm(posA - posB)
        return distance

    def getClosestSV(self, SVs: List[IDMVehicle]):
        if SVs:
            closestIdex = -1
            closestDis = 99999999
            for i, sv in enumerate(SVs):
                dis = self.getVehDis(sv)
                if dis < closestDis:
                    closestDis = dis
                    closestIdex = i
            return SVs[closestIdex]
        else:
            return None

    def processSingleLaneSVs(self, SingleLaneSVs: List[IDMVehicle]):
        # 返回当前车道上，前方最近的车辆和后方最近的车辆，如果没有，则为 None
        if SingleLaneSVs:
            aheadSVs = []
            behindSVs = []
            for sv in SingleLaneSVs:
                RSStr = self.getSVRelativeState(sv)
                if RSStr == 'is ahead of you':
                    aheadSVs.append(sv)
                else:
                    behindSVs.append(sv)
            aheadClosestOne = self.getClosestSV(aheadSVs)
            behindClosestOne = self.getClosestSV(behindSVs)
            return aheadClosestOne, behindClosestOne
        else:
            return None, None

    def processSVsNormalLane(
            self, SVs: List[IDMVehicle], currentLaneIndex: LaneIndex
    ):
        # 目前 description 中的车辆有些太多了，需要处理一下，只保留最靠近 ego 的几辆车
        classifiedSVs: Dict[str, List[IDMVehicle]] = {
            'current lane': [],
            'left lane': [],
            'right lane': [],
            'target lane': []
        }
        sideLanes = self.network.all_side_lanes(currentLaneIndex)
        nextLane = self.network.next_lane(
            currentLaneIndex, self.ego.route, self.ego.position
        )
        for sv in SVs:
            lidx = sv.lane_index
            if lidx in sideLanes:
                if lidx == currentLaneIndex:
                    classifiedSVs['current lane'].append(sv)
                else:
                    laneRelative = lidx[2] - currentLaneIndex[2]
                    if laneRelative == 1:
                        classifiedSVs['right lane'].append(sv)
                    elif laneRelative == -1:
                        classifiedSVs['left lane'].append(sv)
                    else:
                        continue
            elif lidx == nextLane:
                classifiedSVs['target lane'].append(sv)
            else:
                continue

        validVehicles: List[IDMVehicle] = []
        existVehicles: Dict[str, bool] = {}
        for k, v in classifiedSVs.items():
            if v:
                existVehicles[k] = True
            else:
                existVehicles[k] = False
            ahead, behind = self.processSingleLaneSVs(v)
            if ahead:
                validVehicles.append(ahead)
            if behind:
                validVehicles.append(behind)

        return validVehicles, existVehicles

    def describeSVNormalLane(self, currentLaneIndex: LaneIndex) -> str:
        # 当 ego 在 StraightLane 上时，车道信息是重要的，需要处理车道信息
        # 首先判断车辆是不是和车辆在同一条 road 上
        #   如果在同一条 road 上，则判断在哪条 lane 上
        #   如果不在同一条 road 上，则判断是否在 next_lane 上
        #      如果不在 nextLane 上，则直接不考虑这辆车的信息
        #      如果在 nextLane 上，则统计这辆车关于 ego 的相对运动状态
        sideLanes = self.network.all_side_lanes(currentLaneIndex)
        nextLane = self.network.next_lane(
            currentLaneIndex, self.ego.route, self.ego.position
        )
        surroundVehicles = self.getSurrendVehicles(10)
        validVehicles, existVehicles = self.processSVsNormalLane(
            surroundVehicles, currentLaneIndex
        )
        if not surroundVehicles:
            SVDescription = "There are no other vehicles driving near you, so you can drive completely according to your own ideas.\n"
            return SVDescription
        else:
            SVDescription = ''
            for sv in surroundVehicles:
                lidx = sv.lane_index
                if lidx in sideLanes:
                    # 车辆和 ego 在同一条 road 上行驶
                    if lidx == currentLaneIndex:
                        # 车辆和 ego 在同一条 lane 上行驶
                        if sv in validVehicles:
                            SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on the same lane as you and {self.getSVRelativeState(sv)}. "
                        else:
                            continue
                    else:
                        laneRelative = lidx[2] - currentLaneIndex[2]
                        if laneRelative == 1:
                            # laneRelative = 1 表示车辆在 ego 的右侧车道上行驶
                            if sv in validVehicles:
                                SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on the lane to your right and {self.getSVRelativeState(sv)}. "
                            else:
                                continue
                        elif laneRelative == -1:
                            # laneRelative = -1 表示车辆在 ego 的左侧车道上行驶
                            if sv in validVehicles:
                                SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on the lane to your left and {self.getSVRelativeState(sv)}. "
                            else:
                                continue
                        else:
                            # laneRelative 是其他的值表示在更远的车道上，不需要考虑
                            continue
                elif lidx == nextLane:
                    # 车辆在 ego 的 nextLane 上行驶
                    if sv in validVehicles:
                        SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on your target lane and {self.getSVRelativeState(sv)}. "
                    else:
                        continue
                else:
                    continue
                if self.envType == 'intersection-v1':
                    SVDescription += f"The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, acceleration is {sv.action['acceleration']:.2f} m/s^2.\n"
                else:
                    SVDescription += f"The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, acceleration is {sv.action['acceleration']:.2f} m/s^2, and lane position is {self.getLanePosition(sv):.2f} m.\n"
            if SVDescription:
                descriptionPrefix = "There are other vehicles driving around you, and below is their basic information:\n"
                return descriptionPrefix + SVDescription
            else:
                SVDescription = 'There are no other vehicles driving near you, so you can drive completely according to your own ideas.\n'
                return SVDescription

    def isInDangerousArea(self, sv: IDMVehicle) -> bool:
        relativeVector = sv.position - self.ego.position
        distance = np.linalg.norm(relativeVector)
        egoUnitVector = self.getUnitVector(self.ego.heading)
        relativeUnitVector = relativeVector / distance
        alpha = np.arccos(
            np.clip(np.dot(egoUnitVector, relativeUnitVector), -1, 1)
        )
        if alpha <= self.theta1:
            if distance <= self.radius1:
                return True
            else:
                return False
        elif self.theta1 < alpha <= self.theta2:
            if distance <= self.radius2:
                return True
            else:
                return False
        else:
            return False

    def describeSVJunctionLane(self, currentLaneIndex: LaneIndex) -> str:
        # 当 ego 在交叉口内部时，车道的信息不再重要，只需要判断车辆和 ego 的相对位置
        # 但是需要判断交叉口内部所有车道关于 ego 的位置
        nextLane = self.network.next_lane(
            currentLaneIndex, self.ego.route, self.ego.position
        )
        surroundVehicles = self.getSurrendVehicles(6)
        if not surroundVehicles:
            SVDescription = "There are no other vehicles driving near you, so you can drive completely according to your own ideas.\n"
            return SVDescription
        else:
            SVDescription = ''
            for sv in surroundVehicles:
                lidx = sv.lane_index
                if self.isInJunction(sv):
                    collisionPoint = self.getCollisionPoint(sv)
                    if collisionPoint:
                        SVDescription += f"- Vehicle `{id(sv) % 1000}` is also in the junction and {self.getSVRelativeState(sv)}. The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, and acceleration is {sv.action['acceleration']:.2f} m/s^2. The potential collision point is `({collisionPoint[0]:.2f}, {collisionPoint[1]:.2f})`.\n"
                    else:
                        SVDescription += f"- Vehicle `{id(sv) % 1000}` is also in the junction and {self.getSVRelativeState(sv)}. The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, and acceleration is {sv.action['acceleration']:.2f} m/s^2. You two are no potential collision.\n"
                elif lidx == nextLane:
                    collisionPoint = self.getCollisionPoint(sv)
                    if collisionPoint:
                        SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on your target lane and {self.getSVRelativeState(sv)}. The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, and acceleration is {sv.action['acceleration']:.2f} m/s^2. The potential collision point is `({collisionPoint[0]:.2f}, {collisionPoint[1]:.2f})`.\n"
                    else:
                        SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on your target lane and {self.getSVRelativeState(sv)}. The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, and acceleration is {sv.action['acceleration']:.2f} m/s^2. You two are no potential collision.\n"
                if self.isInDangerousArea(sv):
                    print(f"Vehicle {id(sv) % 1000} is in dangerous area.")
                    SVDescription += f"- Vehicle `{id(sv) % 1000}` is also in the junction and {self.getSVRelativeState(sv)}. The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, and acceleration is {sv.action['acceleration']:.2f} m/s^2. This car is within your field of vision, and you need to pay attention to its status when making decisions.\n"
                else:
                    continue
            if SVDescription:
                descriptionPrefix = "There are other vehicles driving around you, and below is their basic information:\n"
                return descriptionPrefix + SVDescription
            else:
                'There are no other vehicles driving near you, so you can drive completely according to your own ideas.\n'
                return SVDescription

    def describe(self, decisionFrame: int) -> str:
        surroundVehicles = self.getSurrendVehicles(10)
        self.dbBridge.insertVehicle(decisionFrame, surroundVehicles)
        currentLaneIndex: LaneIndex = self.ego.lane_index
        if self.isInJunction(self.ego):
            roadCondition = "You are driving in an intersection, you can't change lane. "
            roadCondition += f"Your current position is `({self.ego.position[0]:.2f}, {self.ego.position[1]:.2f})`, speed is {self.ego.speed:.2f} m/s, and acceleration is {self.ego.action['acceleration']:.2f} m/s^2.\n"
            SVDescription = self.describeSVJunctionLane(currentLaneIndex)
        else:
            roadCondition = self.processNormalLane(currentLaneIndex)
            SVDescription = self.describeSVNormalLane(currentLaneIndex)

        return roadCondition + SVDescription

    def promptsCommit(
        self, decisionFrame: int, vectorID: str, done: bool,
        description: str, fewshots: str, thoughtsAndAction: str
    ):
        self.dbBridge.insertPrompts(
            decisionFrame, vectorID, done, description,
            fewshots, thoughtsAndAction
        )
