from manimlib import *

class ThreeDCoordinateSystem(InteractiveScene):
    def construct(self):
        # Create colored axes with arrows using Arrow
        x_axis = Arrow(
            start=np.array([-4, 0, 0]),
            end=np.array([4.5, 0, 0]),
            color=RED,
            stroke_width=4,
        )
        y_axis = Arrow(
            start=np.array([0, -4, 0]),
            end=np.array([0, 4.5, 0]),
            color=GREEN,
            stroke_width=4,
        )
        z_axis = Arrow(
            start=np.array([0, 0, -4]),
            end=np.array([0, 0, 4.5]),
            color=BLUE,
            stroke_width=4,
        )

        # Axis labels
        x_label = Text("X", color=RED, font_size=36)
        x_label.move_to([5, 0, 0])
        y_label = Text("Y", color=GREEN, font_size=36)
        y_label.move_to([0, 5, 0])
        z_label = Text("Z", color=BLUE, font_size=36)
        z_label.move_to([0, 0, 5])

        # Create tick marks and labels
        ticks = Group()
        tick_labels = VGroup()

        for i in range(-4, 5):
            if i == 0:
                continue
            # X axis ticks
            x_tick = Line3D(
                start=np.array([i, -0.1, 0]),
                end=np.array([i, 0.1, 0]),
                color=RED,
            )
            ticks.add(x_tick)
            x_num = Text(str(i), font_size=20, color=RED)
            x_num.move_to([i, -0.4, 0])
            tick_labels.add(x_num)

            # Y axis ticks
            y_tick = Line3D(
                start=np.array([-0.1, i, 0]),
                end=np.array([0.1, i, 0]),
                color=GREEN,
            )
            ticks.add(y_tick)
            y_num = Text(str(i), font_size=20, color=GREEN)
            y_num.move_to([-0.4, i, 0])
            tick_labels.add(y_num)

            # Z axis ticks
            z_tick = Line3D(
                start=np.array([-0.1, 0, i]),
                end=np.array([0.1, 0, i]),
                color=BLUE,
            )
            ticks.add(z_tick)
            z_num = Text(str(i), font_size=20, color=BLUE)
            z_num.move_to([-0.4, 0, i])
            tick_labels.add(z_num)

        # Create some 3D points
        points_data = [
            (1, 2, 3, YELLOW, "A"),
            (-2, 1, 2, PURPLE, "B"),
            (3, -1, 1, ORANGE, "C"),
            (0, 3, -2, PINK, "D"),
            (-1, -2, 3, TEAL, "E"),
            (2, 2, 2, WHITE, "F"),
        ]

        dots = Group()
        for x, y, z, color, name in points_data:
            dot = Sphere(radius=0.12, color=color)
            dot.move_to([x, y, z])
            dots.add(dot)

        # Set initial camera orientation
        frame = self.camera.frame
        frame.set_euler_angles(
            theta=45 * DEGREES,
            phi=70 * DEGREES,
        )

        # Add everything to scene
        self.add(x_axis, y_axis, z_axis)
        self.add(x_label, y_label, z_label)
        self.add(ticks, tick_labels)
        self.add(dots)

        # Interactive: drag mouse to rotate, scroll to zoom
        # Press 'd' to enter drag mode, 'f' to reset, 'z' to zoom mode
        self.wait()

