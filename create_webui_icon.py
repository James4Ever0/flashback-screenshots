#!/usr/bin/env python3
"""Generate favicon.ico with lightning symbol for Flashback web UI."""

from PIL import Image, ImageDraw

def create_lightning_favicon():
    """Create a favicon.ico with a lightning bolt symbol."""
    # Create a 64x64 image with transparent background
    img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Lightning bolt polygon points (centered in 64x64)
    # A stylized lightning bolt shape
    lightning_points = [
        (34, 4),   # Top point
        (20, 26),  # Upper left indent
        (30, 26),  # Upper right before dip
        (24, 40),  # Middle left
        (36, 40),  # Middle right before bottom
        (22, 60),  # Bottom point
        (28, 42),  # Return up left
        (18, 42),  # Return up right before top
    ]

    # Fill with a bright yellow/gold color
    draw.polygon(lightning_points, fill=(255, 215, 0, 255), outline=(255, 165, 0, 255), width=2)

    # Save as ICO with multiple sizes
    output_path = 'flashback/api/static/favicon.ico'
    img.save(output_path, format='ICO', sizes=[(16, 16), (32, 32), (64, 64)])
    print(f"Created: {output_path}")


if __name__ == "__main__":
    create_lightning_favicon()
