from manimlib import *
import numpy as np
from PIL import Image
import os

# Data loader functions
def load_from_image(image_path, num_points=-1, brightness_threshold=-1, z_scale=1.0):
    """Extract points from an image as 3D coordinates
    num_points: -1 means no limit, otherwise randomly sample this many points
    brightness_threshold: -1 means no filter, otherwise only include pixels above threshold
    """
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    total_pixels = w * h

    # Calculate brightness (grayscale) - normalized for threshold comparison
    brightness = np.mean(img_array, axis=2) / 255.0

    # Find pixels (with optional brightness filter)
    if brightness_threshold >= 0:
        bright_mask = brightness > brightness_threshold
        y_coords, x_coords = np.where(bright_mask)
        if len(x_coords) == 0:
            # Lower threshold if no points found
            bright_mask = brightness > 0.5
            y_coords, x_coords = np.where(bright_mask)
    else:
        # No filter - get all pixels
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        y_coords = y_coords.flatten()
        x_coords = x_coords.flatten()

    filtered_count = len(x_coords)

    if num_points > 0 and len(x_coords) > num_points:
        indices = np.random.choice(len(x_coords), num_points, replace=False)
        x_coords = x_coords[indices]
        y_coords = y_coords[indices]

    final_count = len(x_coords)

    # Print info
    print(f"[Image] Size: {w}x{h}, Total pixels: {total_pixels}")
    print(f"[Filter] brightness_threshold: {brightness_threshold}, num_points: {num_points}")
    print(f"[Result] After filter: {filtered_count}, Final points: {final_count}")

    # Use actual pixel coordinates, origin at image (0,0)
    x_norm = x_coords.astype(float)           # [0, w]
    y_norm = (h - y_coords).astype(float)     # [0, h], Y flipped (image top = high Y)

    # Use original brightness value (0-255) scaled by z_scale
    z_coords = np.array([brightness[y, x] * 255 * z_scale for x, y in zip(x_coords, y_coords)])

    # Get colors from image
    colors = [img_array[y, x] / 255.0 for x, y in zip(x_coords, y_coords)]

    return list(zip(x_norm, y_norm, z_coords, colors)), (w, h)


def load_from_fits(fits_path, num_points=-1, brightness_threshold=-1, z_scale=1.0):
    """Extract points from a FITS file as 3D coordinates
    num_points: -1 means no limit, otherwise randomly sample this many points
    brightness_threshold: -1 means no filter, otherwise only include pixels above threshold
    """
    from astropy.io import fits

    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(float)

    h, w = data.shape
    total_pixels = w * h

    # Normalize data to [0, 1]
    data_min, data_max = np.nanmin(data), np.nanmax(data)
    if data_max > data_min:
        data_norm = (data - data_min) / (data_max - data_min)
    else:
        data_norm = np.zeros_like(data)

    # Find pixels (with optional brightness filter)
    if brightness_threshold >= 0:
        bright_mask = data_norm > brightness_threshold
        y_coords, x_coords = np.where(bright_mask)
        if len(x_coords) == 0:
            bright_mask = data_norm > 0.5
            y_coords, x_coords = np.where(bright_mask)
    else:
        # No filter - get all pixels
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        y_coords = y_coords.flatten()
        x_coords = x_coords.flatten()

    filtered_count = len(x_coords)

    if num_points > 0 and len(x_coords) > num_points:
        indices = np.random.choice(len(x_coords), num_points, replace=False)
        x_coords = x_coords[indices]
        y_coords = y_coords[indices]

    final_count = len(x_coords)

    # Print info
    print(f"[FITS] Size: {w}x{h}, Total pixels: {total_pixels}")
    print(f"[Filter] brightness_threshold: {brightness_threshold}, num_points: {num_points}")
    print(f"[Result] After filter: {filtered_count}, Final points: {final_count}")

    # Use actual pixel coordinates, origin at image (0,0)
    x_norm = x_coords.astype(float)           # [0, w]
    y_norm = (h - y_coords).astype(float)     # [0, h], Y flipped
    # Use original brightness value (0-255) scaled by z_scale
    z_coords = np.array([data_norm[y, x] * 255 * z_scale for x, y in zip(x_coords, y_coords)])

    # Color based on intensity (blue to white)
    colors = []
    for x, y in zip(x_coords, y_coords):
        intensity = data_norm[y, x]
        colors.append([intensity, intensity, 1.0])  # Blue to white

    return list(zip(x_norm, y_norm, z_coords, colors)), (w, h)


class AstroViewer3D(InteractiveScene):
    # Configuration
    data_source = "../test-img/sdss.jpg"  # Can be .jpg, .png, or .fits
    num_points = -1  # -1 means no limit
    brightness_threshold = 0.01  # -1 means no filter (show all pixels)
    z_scale = 1.0  # Z axis scale factor, 1.0 = original brightness value (0-255)

    def construct(self):
        # Load data based on file type
        if self.data_source.endswith('.fits'):
            points_data, (img_w, img_h) = load_from_fits(
                self.data_source, self.num_points, self.brightness_threshold, self.z_scale
            )
        else:
            points_data, (img_w, img_h) = load_from_image(
                self.data_source, self.num_points, self.brightness_threshold, self.z_scale
            )

        # Calculate max Z value for axis
        max_z = 255 * self.z_scale

        # Create axes based on actual image size
        x_axis = Arrow(start=np.array([-img_w*0.05, 0, 0]), end=np.array([img_w*1.05, 0, 0]),
                       color=RED, stroke_width=4)
        y_axis = Arrow(start=np.array([0, -img_h*0.05, 0]), end=np.array([0, img_h*1.05, 0]),
                       color=GREEN, stroke_width=4)
        z_axis = Arrow(start=np.array([0, 0, -max_z*0.05]), end=np.array([0, 0, max_z*1.05]),
                       color=BLUE, stroke_width=4)

        # Create tick marks and labels (every 1/4 of image size)
        ticks = Group()
        tick_labels = Group()
        tick_size = img_w * 0.01  # Tick size proportional to image
        label_size = max(img_w, img_h) * 0.03  # Label font size proportional to image

        for i in range(5):
            x_pos = i * img_w / 4
            # X axis ticks
            x_tick = Line3D(start=np.array([x_pos, -tick_size, 0]), end=np.array([x_pos, tick_size, 0]), color=RED)
            ticks.add(x_tick)
            # X axis labels
            x_label = Integer(int(x_pos), color=RED)
            x_label.scale(label_size)
            x_label.move_to([x_pos, -tick_size*5, 0])
            tick_labels.add(x_label)

        for i in range(5):
            y_pos = i * img_h / 4
            # Y axis ticks
            y_tick = Line3D(start=np.array([-tick_size, y_pos, 0]), end=np.array([tick_size, y_pos, 0]), color=GREEN)
            ticks.add(y_tick)
            # Y axis labels
            y_label = Integer(int(y_pos), color=GREEN)
            y_label.scale(label_size)
            y_label.move_to([-tick_size*5, y_pos, 0])
            tick_labels.add(y_label)

        # Z axis ticks (0-255 scaled)
        for i in range(5):
            z_pos = i * max_z / 4
            z_tick = Line3D(start=np.array([-tick_size, 0, z_pos]), end=np.array([tick_size, 0, z_pos]), color=BLUE)
            ticks.add(z_tick)
            z_label = Integer(int(i * 255 / 4), color=BLUE)  # Show original brightness value
            z_label.scale(label_size)
            z_label.move_to([-tick_size*5, 0, z_pos])
            tick_labels.add(z_label)

        # Create 3D points from loaded data
        dot_radius = max(img_w, img_h) * 0.005  # Dot size proportional to image
        dots = Group()
        for x, y, z, color in points_data:
            rgb_color = rgb_to_color(color[:3]) if len(color) >= 3 else WHITE
            dot = Sphere(radius=dot_radius, color=rgb_color)
            dot.move_to([x, y, z])
            # Store coordinates as custom attribute for click detection
            dot.point_coords = (x, y, z)
            dots.add(dot)

        # Store dots reference for click handler
        self.dots = dots
        self.points_data = points_data
        self.img_size = (img_w, img_h)

        # Set camera to view the entire image
        frame = self.camera.frame
        frame.set_euler_angles(theta=45 * DEGREES, phi=70 * DEGREES)
        frame.move_to([img_w/2, img_h/2, 0])  # Center camera on image
        frame.set_height(max(img_w, img_h) * 1.5)  # Zoom out to fit image

        # Add to scene
        self.add(x_axis, y_axis, z_axis)
        self.add(ticks)
        self.add(tick_labels)
        self.add(dots)

        # Print info to console instead of displaying on screen
        source_name = os.path.basename(self.data_source)
        print(f"[Info] Source: {source_name} | Size: {img_w}x{img_h} | Points: {len(points_data)}")
        
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
            print(f"[Point] X: {coords[0]:.1f}, Y: {coords[1]:.1f}, Z: {coords[2]:.3f} (dist: {min_dist:.2f})")

            # Remove previous highlight
            if self.highlight_ring is not None:
                self.remove(self.highlight_ring)

            # Create glow effect - multiple rings (size proportional to image)
            self.highlight_ring = Group()
            center = closest_dot.get_center()
            base_size = max(self.img_size) * 0.01
            for i, r in enumerate([base_size, base_size*1.5, base_size*2]):
                ring = Circle(radius=r, color=YELLOW, stroke_width=3-i)
                ring.set_stroke(opacity=0.8-i*0.2)
                ring.move_to(center)
                self.highlight_ring.add(ring)

            # Add outer glow sphere
            glow = Sphere(radius=base_size*1.2, color=YELLOW)
            glow.set_opacity(0.3)
            glow.move_to(center)
            self.highlight_ring.add(glow)

            self.add(self.highlight_ring)


class AstroViewerFITS(AstroViewer3D):
    """View FITS file data"""
    data_source = "../test-img/galaxies.fits"
    num_points = -1
    brightness_threshold = 0.7


class AstroViewerSDSS(AstroViewer3D):
    """View SDSS image"""
    data_source = "../test-img/sdss.jpg"
    num_points = -1
    brightness_threshold = 0.5


class AstroViewerM44(AstroViewer3D):
    """View M44 star cluster"""
    data_source = "../test-img/m44-1975-01-18.jpg"
    num_points = 100
    brightness_threshold = 0.7

