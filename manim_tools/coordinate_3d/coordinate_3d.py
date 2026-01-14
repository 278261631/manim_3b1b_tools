from manimlib import *

class ThreeDCoordinateSystem(InteractiveScene):
    def construct(self):
        # Create 3D axes
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-4, 4, 1],
        )
        
        # Create some 3D points with labels
        points_data = [
            (1, 2, 3, RED, "A"),
            (-2, 1, 2, GREEN, "B"),
            (3, -1, 1, BLUE, "C"),
            (0, 3, -2, YELLOW, "D"),
            (-1, -2, 3, PURPLE, "E"),
            (2, 2, 2, ORANGE, "F"),
        ]
        
        dots = Group()
        for x, y, z, color, name in points_data:
            dot = Sphere(radius=0.12, color=color)
            dot.move_to(axes.c2p(x, y, z))
            dots.add(dot)
        
        # Set initial camera orientation
        frame = self.camera.frame
        frame.set_euler_angles(
            theta=45 * DEGREES,
            phi=70 * DEGREES,
        )
        
        # Add everything to scene
        self.add(axes, dots)
        
        # Interactive: drag mouse to rotate, scroll to zoom
        # Press 'd' to enter drag mode, 'f' to reset, 'z' to zoom mode
        self.wait()

