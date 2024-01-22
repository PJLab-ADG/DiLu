import base64
import os
import sqlite3
import numpy as np
from io import BytesIO
from typing import List, Tuple
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.text import Text
from dataclasses import dataclass


@dataclass
class FramePrompts:
    decisionFrame: int
    vectorID: str
    done: bool
    description: str
    fewshots: str
    thoughtsAndAction: str
    editedTA: str
    editTimes: int

    @classmethod
    def createFromCursor(cls, decisionFrame: int, cursorData: Tuple):
        return cls(
            decisionFrame, cursorData[0], cursorData[1],
            cursorData[2], cursorData[3], cursorData[4],
            cursorData[5], cursorData[6]
        )


class EnvScenarioReplay:
    def __init__(self, database: str) -> None:
        self.database = database

    def processWayPoint(self, wayPoint: str) -> List[List[float]]:
        wayList = wayPoint.split(' ')
        wayListSplit = [point.split(',') for point in wayList]
        wayX, wayY = list(zip(*wayListSplit))
        wayX = list(map(float, wayX))
        wayY = list(map(float, wayY))
        return wayX, wayY

    def plotNetwork(self, ax: plt.Axes):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(
            """SELECT laneIndexO, laneIndexD, laneIndexI, 
            wayPoint, width from networkINFO;"""
        )
        networkINFO = cur.fetchall()
        for lane in networkINFO:
            laneIndexO, laneIndexD, laneIndexI, wayPoint, width = lane
            wayX, wayY = self.processWayPoint(wayPoint)
            ax.plot(
                wayX, wayY, linewidth=width * 2.8, color='#c8d6e5', alpha=0.3
            )
        conn.close()

    def getVehShape(
        self, posx: float, posy: float,
        heading: float, length: float, widht: float
    ):
        radian = np.pi - heading
        rotation_matrix = np.array(
            [
                [np.cos(radian), -np.sin(radian)],
                [np.sin(radian), np.cos(radian)]
            ]
        )
        half_length = length / 2
        half_width = widht / 2
        vertices = np.array(
            [
                [half_length, half_width], [half_length, -half_width],
                [-half_length, -half_width], [-half_length, half_width]
            ]
        )
        rotated_vertices = np.dot(vertices, rotation_matrix)
        position = np.array([posx, posy])
        translated_vertices = rotated_vertices + position
        return translated_vertices.tolist()

    def plotSce(self, decisionFrame: int) -> str:
        if not os.path.exists('./temp'):
            os.mkdir('./temp')
        fig, ax = plt.subplots()
        self.plotNetwork(ax)

        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(
            f"""SELECT vehicleID, length, width, posx, posy, heading 
            FROM vehINFO WHERE decisionFrame = {decisionFrame}"""
        )
        vehINFO = cur.fetchall()
        for vehicle in vehINFO:
            vehicleID, length, width, posx, posy, heading = vehicle
            vehVertices = self.getVehShape(
                posx, posy, heading, length, width
            )
            if vehicleID == 'ego':
                egoPosx, egoPosy = posx, posy
                vehRectangle = Polygon(vehVertices, facecolor='#ff9f43')
                vehText = Text(posx, posy, 'ego')
            else:
                vehRectangle = Polygon(vehVertices, facecolor='#1dd1a1')
                vehText = Text(posx, posy, vehicleID)
            ax.add_patch(vehRectangle)
            ax.add_artist(vehText)
        conn.close()

        ax.set_xlim(egoPosx-50, egoPosx+50)
        ax.set_ylim(egoPosy-50, egoPosy+50)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        buffer = BytesIO()

        plt.savefig('./temp/temp.png', bbox_inches='tight')
        plt.close()
        return './temp/temp.png'

    def getPrompts(self, decisionFrame: int):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(
            f"""SELECT vectorID, done, description, fewshots, 
            thoughtsAndAction, editedTA, editTimes
            FROM promptsINFO WHERE decisionFrame = {decisionFrame}"""
        )
        framePrompts = FramePrompts.createFromCursor(
            decisionFrame, cur.fetchone()
        )
        conn.close()
        return framePrompts

    def getMinMaxFrame(self):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(
            """SELECT min(decisionFrame), max(decisionFrame) 
            FROM promptsINFO;"""
        )
        minFrame, maxFrame = cur.fetchone()
        conn.close()
        return int(minFrame), int(maxFrame)

    def editTA(self, decisionFrame: int, editedTA: str):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(
            """SELECT editTimes FROM promptsINFO WHERE decisionFrame =?""",
            (decisionFrame,)
        )
        editTimes = cur.fetchone()[0]
        editTimes += 1
        cur.execute(
            """UPDATE promptsINFO SET editedTA =?, editTimes =? 
            WHERE decisionFrame =?""",
            (editedTA, editTimes, decisionFrame)
        )
        conn.commit()
        conn.close()
