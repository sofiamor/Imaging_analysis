import sys
import os
import subprocess

def check_python_version():
    print("Python version:", sys.version)

def check_pip_version():
    try:
        import pip
        print("Pip version:", pip.__version__)
    except ImportError:
        print("Pip is not installed.")

def check_installed_packages():
    try:
        import pip
        installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'list'])
        print("Installed packages:")
        print(installed_packages.decode())
    except Exception as e:
        print("Error checking installed packages:", e)

def check_skimage_installation():
    try:
        import skimage
        print("scikit-image version:", skimage.__version__)
    except ImportError:
        print("scikit-image is not installed.")

def main():
    check_python_version()
    check_pip_version()
    check_installed_packages()
    check_skimage_installation()

if __name__ == "__main__":
    main()
