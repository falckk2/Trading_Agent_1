#!/usr/bin/env python3
"""
Script to generate PNG images from PlantUML files using the PlantUML web service.
"""
import base64
import zlib
import os
import glob
import urllib.request
import urllib.parse

def plantuml_encode(plantuml_text):
    """Encode PlantUML text for URL."""
    zlibbed_str = zlib.compress(plantuml_text.encode('utf-8'))
    compressed_string = zlibbed_str[2:-4]
    return encode64(compressed_string)

def encode64(data):
    """Custom base64 encoding for PlantUML."""
    b64 = ""
    for i in range(0, len(data), 3):
        if i + 2 == len(data):
            b64 += _encode3bytes(data[i], data[i + 1], 0)
        elif i + 1 == len(data):
            b64 += _encode3bytes(data[i], 0, 0)
        else:
            b64 += _encode3bytes(data[i], data[i + 1], data[i + 2])
    return b64

def _encode3bytes(b1, b2, b3):
    """Encode 3 bytes into 4 characters."""
    c1 = b1 >> 2
    c2 = ((b1 & 0x3) << 4) | (b2 >> 4)
    c3 = ((b2 & 0xF) << 2) | (b3 >> 6)
    c4 = b3 & 0x3F
    return _encode6bit(c1 & 0x3F) + _encode6bit(c2 & 0x3F) + \
           _encode6bit(c3 & 0x3F) + _encode6bit(c4 & 0x3F)

def _encode6bit(b):
    """Encode 6-bit value to character."""
    if b < 10:
        return chr(48 + b)
    b -= 10
    if b < 26:
        return chr(65 + b)
    b -= 26
    if b < 26:
        return chr(97 + b)
    b -= 26
    if b == 0:
        return '-'
    if b == 1:
        return '_'
    return '?'

def generate_png_from_puml(puml_file, output_file):
    """Generate PNG from PlantUML file using web service."""
    print(f"Processing {puml_file}...")

    # Read the PlantUML file
    with open(puml_file, 'r', encoding='utf-8') as f:
        plantuml_text = f.read()

    # Encode the PlantUML text
    encoded = plantuml_encode(plantuml_text)

    # Try multiple servers in case one fails
    servers = [
        f"http://www.plantuml.com/plantuml/png/{encoded}",
        f"https://kroki.io/plantuml/png/{encoded}",
    ]

    last_error = None
    for url in servers:
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(request, timeout=30) as response:
                png_data = response.read()

            # Save the PNG
            with open(output_file, 'wb') as f:
                f.write(png_data)

            print(f"  ✓ Generated {output_file}")
            return True
        except Exception as e:
            last_error = e
            continue

    print(f"  ✗ Error generating {output_file}: {last_error}")
    # Try to get more details
    try:
        error_url = f"http://www.plantuml.com/plantuml/txt/{encoded}"
        with urllib.request.urlopen(error_url) as response:
            error_msg = response.read().decode('utf-8')
            if error_msg and 'error' in error_msg.lower():
                print(f"     PlantUML error: {error_msg[:200]}")
    except:
        pass
    return False

def main():
    """Main function to process all PlantUML files."""
    # Find all .puml files in the diagrams directory
    puml_files = glob.glob('diagrams/*.puml')

    if not puml_files:
        print("No .puml files found in diagrams/ directory")
        return

    print(f"Found {len(puml_files)} PlantUML files\n")

    success_count = 0
    for puml_file in sorted(puml_files):
        # Generate output filename
        output_file = puml_file.replace('.puml', '.png')

        # Generate PNG
        if generate_png_from_puml(puml_file, output_file):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"Successfully generated {success_count}/{len(puml_files)} PNG files")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
