from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np

def create_glow_text_image(text, font_path, output_path, text_color=(255, 255, 255), glow_color=(255, 255, 255), font_size=24):
    """
    Create a text image with a glow effect.
    Minimal padding - image is tightly cropped to text with just enough space for glow.
    
    Args:
        text: Text to render
        font_path: Path to .ttf font file
        output_path: Where to save the PNG image
        text_color: RGB tuple for text color
        glow_color: RGB tuple for glow color
        font_size: Size of the font
    """
    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"[WARNING] Font not found: {font_path}, using default")
        font = ImageFont.load_default()
    
    # Calculate text size using textbbox (textsize is deprecated)
    dummy_img = Image.new("RGBA", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Minimal padding - just enough for glow blur (radius 3)
    glow_radius = 3
    padding = glow_radius + 2  # Just 5 pixels padding
    width = text_width + padding * 2
    height = text_height + padding * 2
    
    # Create an image with a transparent background
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Add text to the image (positioned with minimal padding)
    text_position = (padding - bbox[0], padding - bbox[1])  # Adjust for font baseline and left offset
    draw.text(text_position, text, font=font, fill=(*text_color, 255))

    # Create glow layer
    glow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    glow_draw.text(text_position, text, font=font, fill=(*glow_color, 180))
    
    # Apply Gaussian blur to create the glow effect (reduced radius)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=glow_radius))

    # Composite: glow behind text
    result = Image.alpha_composite(glow_layer, image)
    
    # Auto-crop to remove any extra transparent space
    result = autocrop_image(result)

    # Save the final image with transparency
    result.save(output_path, format="PNG")


def autocrop_image(image: Image.Image, padding: int = 2) -> Image.Image:
    """
    Automatically crop transparent borders from an image.
    
    Args:
        image: PIL Image with alpha channel
        padding: Pixels of padding to keep around the content
        
    Returns:
        Cropped image
    """
    # Get the alpha channel
    if image.mode != 'RGBA':
        return image
    
    # Get bounding box of non-transparent pixels
    bbox = image.getbbox()
    
    if bbox is None:
        return image
    
    # Add minimal padding
    left = max(0, bbox[0] - padding)
    top = max(0, bbox[1] - padding)
    right = min(image.width, bbox[2] + padding)
    bottom = min(image.height, bbox[3] + padding)
    
    return image.crop((left, top, right, bottom))

if __name__ == "__main__":
    text_to_display = "Your Text Here"
    font_file_path = "Montserrat-Black.ttf"
    output_image_path = "/home/subgenerator/readyimg/output_image.png"

    create_glow_text_image(text_to_display, font_file_path, output_image_path)
