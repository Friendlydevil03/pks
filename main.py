from tkinter import Tk
from ui.app import ParkingManagementSystem
from utils.style_config import apply_styling
from utils.window_manager import WindowManager

if __name__ == "__main__":
    root = Tk()
    # Apply consistent styling
    style = apply_styling(root)

    # Setup window manager
    window_manager = WindowManager(root, "config", "1280x720")
    window_manager.restore_window_position()

    # Create the application
    app = ParkingManagementSystem(root)


    # Save window position on exit
    def on_closing():
        window_manager.save_window_position()
        root.destroy()


    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()