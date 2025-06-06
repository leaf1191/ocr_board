import tkinter as tk
from image_gui import ImageGUI


if __name__ == '__main__':
    root = tk.Tk()
    app = ImageGUI(root)
    print("🟢 Alt+S 눌러서 스크린샷 찍고 자동 분석하세요")
    root.mainloop()