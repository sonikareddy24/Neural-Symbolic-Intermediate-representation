import time
import subprocess
import os
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("watchdog module not found, installing...")
    install_package("watchdog")
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

MD_FILE = "/Users/apple/Desktop/compiler design /NSIR_Research_Paper.md"

class MdChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if os.path.abspath(event.src_path) == os.path.abspath(MD_FILE):
            print(f"Detected modification in {event.src_path}. Rebuilding docx...")
            try:
                subprocess.check_call([sys.executable, "/Users/apple/Desktop/compiler design /build_paper.py"])
            except Exception as e:
                print(f"Error rebuilding paper: {e}")

if __name__ == "__main__":
    if not os.path.exists(MD_FILE):
        print(f"Error: Could not find {MD_FILE}")
        sys.exit(1)
        
    event_handler = MdChangeHandler()
    observer = Observer()
    directory = os.path.dirname(MD_FILE)
    observer.schedule(event_handler, directory, recursive=False)
    
    print(f"Listening for changes in {MD_FILE}...")
    # Initial build to ensure it's up to date
    try:
        subprocess.check_call([sys.executable, "/Users/apple/Desktop/compiler design /build_paper.py"])
    except Exception as e:
        print(f"Initial build failed: {e}")

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
