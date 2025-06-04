import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms, models
from pix2tex.cli import LatexOCR
from image_convert import *
import json
import requests
import pyautogui
import keyboard
from PIL import ImageGrab

# Notion 정보
NOTION_TOKEN = "시크릿 노션 토큰 기입"
NOTION_PAGE_ID = "노션 사이트 id 기입"

def append_to_notion_page(content, token, block_id):
    url = f"https://api.notion.com/v1/blocks/{block_id}/children"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }

    blocks = []
    lines = content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        elif line.startswith('\\') or any(cmd in line for cmd in ['\\frac', '\\ln', '\\left', '\\sum', '\\int', '\\sqrt', '\\log', '\\tan']):
            blocks.append({
                "object": "block",
                "type": "equation",
                "equation": {"expression": line}
            })
        else:
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": line}
                    }]
                }
            })

    payload = {"children": blocks}
    res = requests.patch(url, headers=headers, data=json.dumps(payload))

    if res.status_code == 200:
        print("✅ Notion 전송 완료")
    else:
        print(f"❌ 전송 실패: {res.status_code}, {res.text}")

def load_label_map(label_map_path):
    id2label, id2category = {}, {}
    with open(label_map_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                idx, formula, category = parts
                id2label[int(idx)] = formula
                id2category[int(idx)] = category
    return id2label, id2category

def load_model(model_path, label_map_path, device):
    id2label, id2category = load_label_map(label_map_path)
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, len(id2label))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, id2label, id2category

class ImageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Formula Classifier")
        self.setup_ui()

        self.ocr_model = LatexOCR()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.id2label, self.id2category = load_model(
            "classify/your_dataset/saved_model5.pth",
            "classify/your_dataset/label_map.txt",
            self.device
        )
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.latex_code = ""
        self.label = ""
        self.category = ""

        # Alt+S 단축키 등록
        self.root.bind_all('<Alt-s>', lambda event: self.capture_with_selection())

    def setup_ui(self):
        self.root.geometry("800x600")
        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        self.image_label = tk.Label(self.frame)
        self.image_label.pack(pady=5)

        button_frame = tk.Frame(self.frame)
        button_frame.pack(pady=5)
        tk.Button(button_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Process Image", command=self.process_image).pack(side=tk.LEFT, padx=5)

        tk.Label(self.frame, text="Result:").pack(pady=5)
        self.result_text = tk.Text(self.frame, height=7, width=70)
        self.result_text.pack(pady=5)

        tk.Button(self.frame, text="Copy Result", command=self.copy_result).pack(pady=5)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if path:
            try:
                self.image = Image.open(path)
                self.image.thumbnail((400, 300))
                self.photo = ImageTk.PhotoImage(self.image)
                self.image_label.config(image=self.photo)
                self.image_label.image = self.photo
                self.current_image_path = path
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def process_image(self):
        if hasattr(self, 'image'):
            try:
                self.latex_code = self.ocr_model(self.image)
                tensor = self.transform(self.image.convert("L")).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.model(tensor)
                    idx = output.argmax(1).item()
                    self.label = self.id2label[idx]
                    self.category = self.id2category[idx]

                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"[정제코드 전체 수식 예측]\n{self.latex_code}\n\n[ocr_board 분류 수식]\n{self.label}\n[분류 카테고리]\n{self.category}")

                # 자동 Notion 전송
                content = f"[정제코드 전체 수식 예측]\n{self.latex_code}\n\n[ocr_board 분류 수식]\n{self.label}\n[분류 카테고리]\n{self.category}"
                append_to_notion_page(content, NOTION_TOKEN, NOTION_PAGE_ID)

            except Exception as e:
                messagebox.showerror("Error", str(e))

    def copy_result(self):
        result = self.result_text.get(1.0, tk.END).strip()
        if result:
            self.root.clipboard_clear()
            self.root.clipboard_append(result)
            messagebox.showinfo("Success", "Copied to clipboard")

    def capture_with_selection(self):
        # 투명 캡처창
        overlay = tk.Toplevel(self.root)
        overlay.attributes("-fullscreen", True)
        overlay.attributes("-alpha", 0.3)
        overlay.configure(bg="black")
        overlay.lift()
        overlay.attributes("-topmost", True)
        overlay.config(cursor="crosshair")

        start = [0, 0]
        rect = None

        canvas = tk.Canvas(overlay, cursor="crosshair", bg="gray")
        canvas.pack(fill=tk.BOTH, expand=True)

        def on_mouse_down(event):
            start[0], start[1] = event.x_root, event.y_root

        def on_mouse_drag(event):
            nonlocal rect
            canvas.delete("rect")
            rect = canvas.create_rectangle(
                start[0], start[1], event.x_root, event.y_root,
                outline="red", width=2, tag="rect"
            )

        def on_mouse_up(event):
            x1, y1 = start
            x2, y2 = event.x_root, event.y_root
            overlay.destroy()
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            # 스크린샷 캡처

            screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))

            self.image = screenshot
            self.image.thumbnail((400, 300))
            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo
            self.current_image_path = None


            self.process_image()

        canvas.bind("<ButtonPress-1>", on_mouse_down)
        canvas.bind("<B1-Motion>", on_mouse_drag)
        canvas.bind("<ButtonRelease-1>", on_mouse_up)


if __name__ == '__main__':
    root = tk.Tk()
    app = ImageGUI(root)
    print("🟢 Alt+S 눌러서 스크린샷 찍고 자동 분석하세요")
    root.mainloop()
