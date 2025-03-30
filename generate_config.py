# generate_config.py
from config import UnifiedConfig
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate default configuration file")
    parser.add_argument("--output", type=str, default="config.yaml", help="Output configuration file path")
    args = parser.parse_args()
    
    # Create default configuration
    config = UnifiedConfig()
    
    # Save to file
    config.to_file(args.output)
    print(f"Default configuration saved to {args.output}")

if __name__ == "__main__":
    main()