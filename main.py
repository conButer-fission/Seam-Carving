from seam_carver import scale_image
import argparse
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("Path", type=str, help="Path of the image")    
    parser.add_argument("Width", type=int, help="Width of image")    
    parser.add_argument("Height", type=int, help="Height of image")    
    args = parser.parse_args()
    scale_image(args.Path, args.Width, args.Height)

if __name__ == '__main__':
    main()  