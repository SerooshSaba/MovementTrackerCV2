import cv2
from Vec2f import Vec2f

class Vehicle:

    def __init__(self):
        self.x = 0
        self.y = 0
        self.xaxis = Vec2f(1,0)
        self.yaxis = Vec2f(0,-1)

    def transX(self,amount):
        amount /= 20
        self.x += self.xaxis.x * amount
        self.y += self.xaxis.y * amount

    def transY(self,amount):
        amount /= 20
        self.x += self.yaxis.x * amount
        self.y += self.yaxis.y * amount

    def rotate(self,rad):
        self.xaxis.rotate(rad)
        self.yaxis.rotate(rad)

    def render(self,img,xoffset,yoffset):

        axis_len = 10

        # x - axis
        start = (int(self.x + xoffset), int(self.y + yoffset))
        end = (int(self.x + self.xaxis.x * axis_len + xoffset), int(self.y + self.xaxis.y * axis_len + + yoffset))
        img = cv2.line(img, start, end, (0,0,255), 1 )

        # y - axis
        start = (int(self.x + xoffset), int(self.y + yoffset))
        end = (int(self.x + self.yaxis.x * axis_len + xoffset), int(self.y + self.yaxis.y * axis_len + + yoffset))
        img = cv2.line(img, start, end, (0,255,0), 1 )

        return img