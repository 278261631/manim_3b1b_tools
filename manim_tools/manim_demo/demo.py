from manimlib import *

class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()
        circle.set_fill(BLUE, opacity=0.5)
        circle.set_stroke(BLUE_E, width=4)
        
        square = Square()
        square.set_fill(RED, opacity=0.5)
        square.set_stroke(RED_E, width=4)
        
        self.play(ShowCreation(square))
        self.wait()
        self.play(Transform(square, circle))
        self.wait()

