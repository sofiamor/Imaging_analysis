import os
import sys

def set_project_root():
    """
    Ensures the project root (Voltage_imaging/) is in sys.path
    so all submodules can be imported easily.
    """
    possible_roots = [
        r"C:\Users\sofik\.vscode\Voltage_imaging",
        r"C:\Users\sofik\Documents\GitHub\Imaging_analysis\Voltage_imaging",
        os.path.join(os.path.expanduser("~"), "Imaging_analysis", "Voltage_imaging")
    ]

    for root in possible_roots:
        if os.path.exists(root):
            if root not in sys.path:
                sys.path.insert(0, root)
            os.chdir(root)
            print(f"📁 Project root set to: {root}")
            return root

    raise FileNotFoundError("Could not locate your Voltage_imaging project folder.")
