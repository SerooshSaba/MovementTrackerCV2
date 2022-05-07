import math

class Vec2f:

    def __init__(self,x,y):
        self.x = x
        self.y = y

    def rotate(self, rad):
        c = math.cos(rad)
        s = math.sin(rad)
        x_ = self.x * c - self.y * s
        y_ = self.x * s + self.y * c
        self.x = x_
        self.y = y_

    def scale(self,amount):
        self.x *= amount
        self.y *= amount