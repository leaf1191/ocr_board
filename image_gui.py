import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from pix2tex.cli import LatexOCR
from image_convert import *

class ImageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image GUI")
        
        self.setup_ui()
        
        self.model = LatexOCR()
        
    def setup_ui(self):
        self.root.geometry("800x600")

        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=10, pady=10)
        
        self.image_label = tk.Label(self.frame)
        self.image_label.pack(pady=5)
        
        self.button_frame = tk.Frame(self.frame)
        self.button_frame.pack(pady=5)
        
        self.load_button = tk.Button(self.button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        self.process_button = tk.Button(self.button_frame, text="Process Image", command=self.process_image)
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        self.result_label = tk.Label(self.frame, text="Result:")
        self.result_label.pack(pady=5)
        
        self.result_text = tk.Text(self.frame, height=5, width=50)
        self.result_text.pack(pady=5)
        
        self.copy_button = tk.Button(self.frame, text="Copy Result", command=self.copy_result)
        self.copy_button.pack(pady=5)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if file_path:
            try:
                pil_img = Image.open(file_path)
                self.cv_img = pil_to_cv(pil_img)
                self.cv_img = resize_image(self.cv_img)
                self.image = cv_to_pil(self.cv_img)
                self.image.thumbnail((400, 300))
                self.photo = ImageTk.PhotoImage(self.image)
                self.image_label.config(image=self.photo)
                self.image_label.image = self.photo
                self.current_image_path = file_path
            except Exception as e:
                messagebox.showerror("Error", f"Image load failed: {e}")
    
    def process_image(self):
        if hasattr(self, 'current_image_path'):
            try:
                # 여기서 이미지 전처리 및 모델 학습 작업 수행
                self.cv_img = process_image(self.cv_img)

                # gui 업데이트 로직
                self.image = cv_to_pil(self.cv_img)
                self.image.thumbnail((400, 300))
                self.photo = ImageTk.PhotoImage(self.image)
                self.image_label.config(image=self.photo)
                self.image_label.image = self.photo

                """
                모델 학습 후 gui에 str 띄우는 것은 이 로직 참고하면 될 듯
                latex_code = self.model(self.image) << 여기서 latex_code는 str
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, latex_code)
                """
            except Exception as e:
                messagebox.showerror("Error", f"Image processing failed: {e}")
        else:
            messagebox.showwarning("Warning", "Please load an image first.")
    
    def copy_result(self):
        result = self.result_text.get(1.0, tk.END).strip()
        if result:
            self.root.clipboard_clear()
            self.root.clipboard_append(result)
            messagebox.showinfo("Success", "Result copied to clipboard.")
        else:
            messagebox.showwarning("Warning", "No result to copy.")