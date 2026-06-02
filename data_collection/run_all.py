import argparse
import sys
from pathlib import Path

# Add project root to sys.path to allow running directly from anywhere
project_root = str(Path(__file__).parent.parent.resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_collection.collect_twitter import TwitterCollector
from data_collection.collect_blog import BlogCollector
from data_collection.collect_papers import PaperCollector
from data_collection.collect_github import GitHubCollector

def main():
    parser = argparse.ArgumentParser(
        description="Collect data for Karpathy Digital Twin"
    )
    parser.add_argument(
        '--source',
        choices=['twitter', 'blog', 'papers', 'github', 'all'],
        default='all',
        help='Which source to collect from'
    )
    parser.add_argument(
        '--output',
        default='data/raw',
        help='Output directory for raw data'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Karpathy Digital Twin — Data Collection")
    print("=" * 60)
    
    collectors = {
        'twitter': TwitterCollector(f"{args.output}/twitter"),
        'blog': BlogCollector(f"{args.output}/blog"),
        'papers': PaperCollector(f"{args.output}/papers"),
        'github': GitHubCollector(f"{args.output}/github"),
    }
    
    if args.source == 'all':
        for name, collector in collectors.items():
            print(f"\n[{name.upper()}] Starting collection...")
            try:
                collector.collect()
                print(f"[{name.upper()}] [OK] Complete")
            except Exception as e:
                print(f"[{name.upper()}] [ERROR] Error: {e}")
    else:
        collector = collectors[args.source]
        print(f"\n[{args.source.upper()}] Starting collection...")
        try:
            collector.collect()
            print(f"[{args.source.upper()}] [OK] Complete")
        except Exception as e:
            print(f"[{args.source.upper()}] [ERROR] Error: {e}")
    
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
