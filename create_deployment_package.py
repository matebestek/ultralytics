"""
Package Parking Lot Monitor for deployment
Creates a clean distribution package with all necessary files
"""
import os
import shutil
from pathlib import Path

def create_deployment_package():
    # Define package name and paths
    package_name = "parking-lot-monitor-deploy-v2"
    source_dir = Path(".")
    deploy_dir = Path(package_name)
    
    # Create deployment directory
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir()
    
    print(f"📦 Creating deployment package: {package_name}")
    print("=" * 50)
    
    # Files to include
    files_to_copy = [
        "parking_lot_monitor.py",
        "requirements.txt",
        "DEPLOYMENT_README.md",
        "PATTERN_ANALYTICS.md",
        "start.bat",
        "start.sh",
        "yolo11n.pt"
    ]
    
    # Copy files
    for file in files_to_copy:
        src = source_dir / file
        if src.exists():
            dst = deploy_dir / file
            shutil.copy2(src, dst)
            print(f"✅ Copied: {file}")
        else:
            print(f"⚠️  Missing: {file}")
    
    # Rename README
    readme_src = deploy_dir / "DEPLOYMENT_README.md"
    readme_dst = deploy_dir / "README.md"
    if readme_src.exists():
        readme_src.rename(readme_dst)
        print(f"✅ Renamed: DEPLOYMENT_README.md -> README.md")
    
    # Make shell script executable (on Unix-like systems)
    start_sh = deploy_dir / "start.sh"
    if start_sh.exists():
        try:
            os.chmod(start_sh, 0o755)
            print(f"✅ Made executable: start.sh")
        except:
            pass
    
    print("=" * 50)
    print(f"✨ Deployment package created: {deploy_dir}/")
    print()
    print("📋 Next steps:")
    print("1. Copy the entire folder to target machine")
    print("2. Windows: Double-click start.bat")
    print("3. Linux/Mac: Run ./start.sh")
    print()
    print("📁 Package contents:")
    for item in sorted(deploy_dir.iterdir()):
        size = item.stat().st_size if item.is_file() else 0
        size_mb = size / (1024 * 1024)
        if size_mb > 1:
            print(f"   - {item.name} ({size_mb:.1f} MB)")
        else:
            print(f"   - {item.name}")
    
    # Create zip archive
    try:
        archive_name = shutil.make_archive(package_name, 'zip', '.', package_name)
        archive_size = Path(archive_name).stat().st_size / (1024 * 1024)
        print()
        print(f"📦 Created archive: {archive_name} ({archive_size:.1f} MB)")
        print(f"✅ Ready for deployment!")
    except Exception as e:
        print(f"⚠️  Could not create archive: {e}")

if __name__ == "__main__":
    create_deployment_package()
