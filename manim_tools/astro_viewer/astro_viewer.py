from manimlib import *
import numpy as np
from PIL import Image
import os

# Data loader functions
def load_from_image(image_path, num_points=50, brightness_threshold=0.7):
    """Extract bright points from an image as 3D coordinates"""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Calculate brightness (grayscale)
    brightness = np.mean(img_array, axis=2) / 255.0
    
    # Find bright pixels above threshold
    bright_mask = brightness > brightness_threshold
    y_coords, x_coords = np.where(bright_mask)
    
    if len(x_coords) == 0:
        # Lower threshold if no points found
        bright_mask = brightness > 0.5
        y_coords, x_coords = np.where(bright_mask)
    
    if len(x_coords) > num_points:
        indices = np.random.choice(len(x_coords), num_points, replace=False)
        x_coords = x_coords[indices]
        y_coords = y_coords[indices]
    
    # Normalize coordinates to [-4, 4] range
    h, w = img_array.shape[:2]
    x_norm = (x_coords / w - 0.5) * 8
    y_norm = (0.5 - y_coords / h) * 8  # Flip Y axis
    
    # Use brightness as Z coordinate
    z_coords = np.array([brightness[y, x] * 4 for x, y in zip(x_coords, y_coords)])
    
    # Get colors from image
    colors = [img_array[y, x] / 255.0 for x, y in zip(x_coords, y_coords)]
    
    return list(zip(x_norm, y_norm, z_coords, colors))


def load_from_fits(fits_path, num_points=50, brightness_threshold=0.7):
    """Extract bright points from a FITS file as 3D coordinates"""
    from astropy.io import fits
    
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(float)
    
    # Normalize data to [0, 1]
    data_min, data_max = np.nanmin(data), np.nanmax(data)
    if data_max > data_min:
        data_norm = (data - data_min) / (data_max - data_min)
    else:
        data_norm = np.zeros_like(data)
    
    # Find bright pixels
    bright_mask = data_norm > brightness_threshold
    y_coords, x_coords = np.where(bright_mask)
    
    if len(x_coords) == 0:
        bright_mask = data_norm > 0.5
        y_coords, x_coords = np.where(bright_mask)
    
    if len(x_coords) > num_points:
        indices = np.random.choice(len(x_coords), num_points, replace=False)
        x_coords = x_coords[indices]
        y_coords = y_coords[indices]
    
    # Normalize coordinates
    h, w = data.shape
    x_norm = (x_coords / w - 0.5) * 8
    y_norm = (0.5 - y_coords / h) * 8
    z_coords = np.array([data_norm[y, x] * 4 for x, y in zip(x_coords, y_coords)])
    
    # Color based on intensity (blue to white)
    colors = []
    for x, y in zip(x_coords, y_coords):
        intensity = data_norm[y, x]
        colors.append([intensity, intensity, 1.0])  # Blue to white
    
    return list(zip(x_norm, y_norm, z_coords, colors))


class AstroViewer3D(InteractiveScene):
    # Configuration
    data_source = "../test-img/sdss.jpg"  # Can be .jpg, .png, or .fits
    num_points = 100
    brightness_threshold = 0.6
    
    def construct(self):
        # Load data based on file type
        if self.data_source.endswith('.fits'):
            points_data = load_from_fits(
                self.data_source, self.num_points, self.brightness_threshold
            )
        else:
            points_data = load_from_image(
                self.data_source, self.num_points, self.brightness_threshold
            )
        
        # Create axes
        x_axis = Arrow(start=np.array([-4, 0, 0]), end=np.array([4.5, 0, 0]),
                       color=RED, stroke_width=4)
        y_axis = Arrow(start=np.array([0, -4, 0]), end=np.array([0, 4.5, 0]),
                       color=GREEN, stroke_width=4)
        z_axis = Arrow(start=np.array([0, 0, -0.5]), end=np.array([0, 0, 4.5]),
                       color=BLUE, stroke_width=4)
        
        # Axis labels
        x_label = Text("X", color=RED, font_size=36).move_to([5, 0, 0])
        y_label = Text("Y", color=GREEN, font_size=36).move_to([0, 5, 0])
        z_label = Text("Brightness", color=BLUE, font_size=28).move_to([0, 0, 5])
        
        # Create tick marks and labels
        ticks = Group()
        tick_labels = VGroup()
        for i in range(-4, 5):
            if i == 0:
                continue
            # X axis ticks
            x_tick = Line3D(start=np.array([i, -0.1, 0]), end=np.array([i, 0.1, 0]), color=RED)
            ticks.add(x_tick)
            x_num = Text(str(i), font_size=20, color=RED)
            x_num.move_to([i, -0.4, 0])
            tick_labels.add(x_num)

            # Y axis ticks
            y_tick = Line3D(start=np.array([-0.1, i, 0]), end=np.array([0.1, i, 0]), color=GREEN)
            ticks.add(y_tick)
            y_num = Text(str(i), font_size=20, color=GREEN)
            y_num.move_to([-0.4, i, 0])
            tick_labels.add(y_num)

        # Z axis ticks (0-4 for brightness)
        for i in range(1, 5):
            z_tick = Line3D(start=np.array([-0.1, 0, i]), end=np.array([0.1, 0, i]), color=BLUE)
            ticks.add(z_tick)
            z_num = Text(str(i), font_size=20, color=BLUE)
            z_num.move_to([-0.4, 0, i])
            tick_labels.add(z_num)
        
        # Create 3D points from loaded data
        dots = Group()
        for x, y, z, color in points_data:
            rgb_color = rgb_to_color(color[:3]) if len(color) >= 3 else WHITE
            dot = Sphere(radius=0.08, color=rgb_color)
            dot.move_to([x, y, z])
            # Store coordinates as custom attribute for click detection
            dot.point_coords = (x, y, z)
            dots.add(dot)

        # Store dots reference for click handler
        self.dots = dots
        self.points_data = points_data
        
        # Info text
        source_name = os.path.basename(self.data_source)
        info_text = Text(f"Source: {source_name} | Points: {len(points_data)}", 
                        font_size=24, color=WHITE)
        info_text.to_corner(UL)
        info_text.fix_in_frame()
        
        # Set camera
        frame = self.camera.frame
        frame.set_euler_angles(theta=45 * DEGREES, phi=70 * DEGREES)
        
        # Add to scene
        self.add(x_axis, y_axis, z_axis)
        self.add(x_label, y_label, z_label)
        self.add(ticks, tick_labels)
        self.add(dots)
        self.add(info_text)
        
        # Store selected dot for highlight effect
        self.selected_dot = None
        self.highlight_ring = None

        self.wait()

    def on_mouse_press(self, point, button, mods):
        """Handle mouse click to detect point selection"""
        super().on_mouse_press(point, button, mods)

        # Get mouse position in world coordinates
        mouse_point = self.mouse_point.get_center()

        # Get camera frame for proper 3D picking
        frame = self.camera.frame

        # Find closest point using camera-relative distance
        min_dist = float('inf')
        closest_dot = None

        for dot in self.dots:
            dot_center = dot.get_center()
            # Use full 3D distance
            dist = np.linalg.norm(mouse_point - dot_center)
            if dist < min_dist:
                min_dist = dist
                closest_dot = dot

        # Print nearest point coordinates and add glow effect
        if closest_dot is not None:
            coords = closest_dot.point_coords
            print(f"[Point] X: {coords[0]:.3f}, Y: {coords[1]:.3f}, Z: {coords[2]:.3f} (dist: {min_dist:.2f})")

            # Remove previous highlight
            if self.highlight_ring is not None:
                self.remove(self.highlight_ring)

            # Create glow effect - multiple rings
            self.highlight_ring = Group()
            center = closest_dot.get_center()
            for i, r in enumerate([0.15, 0.22, 0.30]):
                ring = Circle(radius=r, color=YELLOW, stroke_width=3-i)
                ring.set_stroke(opacity=0.8-i*0.2)
                ring.move_to(center)
                self.highlight_ring.add(ring)

            # Add outer glow sphere
            glow = Sphere(radius=0.18, color=YELLOW)
            glow.set_opacity(0.3)
            glow.move_to(center)
            self.highlight_ring.add(glow)

            self.add(self.highlight_ring)


class AstroViewerFITS(AstroViewer3D):
    """View FITS file data"""
    data_source = "../test-img/galaxies.fits"
    num_points = 150
    brightness_threshold = 0.8


class AstroViewerSDSS(AstroViewer3D):
    """View SDSS image"""
    data_source = "../test-img/sdss.jpg"
    num_points = 80
    brightness_threshold = 0.5


class AstroViewerM44(AstroViewer3D):
    """View M44 star cluster"""
    data_source = "../test-img/m44-1975-01-18.jpg"
    num_points = 100
    brightness_threshold = 0.7

