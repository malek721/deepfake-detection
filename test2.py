import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from mtcnn import MTCNN

class MyCNN(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, downsample=None, drop_rate=0.3):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.drop = nn.Dropout(drop_rate)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.drop(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = self.relu(out)
            return out

    def __init__(self, num_classes=2, drop_rate=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1, drop_rate=drop_rate)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2, drop_rate=drop_rate)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2, drop_rate=drop_rate)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride, drop_rate):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [MyCNN.ResidualBlock(in_channels, out_channels, stride, downsample, drop_rate)]
        for _ in range(1, blocks):
            layers.append(MyCNN.ResidualBlock(out_channels, out_channels, drop_rate=drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model yükleme
model_path = r"C:\Users\admin\PycharmProjects\deepface_detecoration\models\deepfake_cnn_new_model.pth"
model = MyCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Yüz algılayıcılar: önce Haar Cascade, sonra MTCNN filtreleme
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mtcnn = MTCNN()

# Görüntü ön işlemleri (128x128)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Kaydedilecek dosyalar için
processed_folder = r"C:\Users\admin\Desktop\yapa zeka\processed_model_inputs"


def analyze_video(video_path, sample_rate=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video"

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(processed_folder, video_name)

    if os.path.exists(save_dir):
        for file in os.listdir(save_dir):
            file_path = os.path.join(save_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    os.makedirs(save_dir, exist_ok=True)

    frame_num = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = frame[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (128, 128))
                fname = f"face_{saved:04d}.jpg"
                cv2.imwrite(os.path.join(save_dir, fname), face_resized)
                saved += 1
        frame_num += 1
    cap.release()

    for img_name in os.listdir(save_dir):
        img_path = os.path.join(save_dir, img_name)
        img = cv2.imread(img_path)
        if img is None or len(mtcnn.detect_faces(img)) == 0:
            os.remove(img_path)
    remaining = len(os.listdir(save_dir))
    if remaining == 0:
        return "No valid face inputs"

    preds = []
    for img_name in os.listdir(save_dir):
        img_path = os.path.join(save_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            p = torch.softmax(out, dim=1)[0, 1].item()
            preds.append(p)

    if preds:
        mean_p = sum(preds) / len(preds)
        deepfake_rate = round(mean_p * 100, 2)
        final_result = "Yüklenen video fake’dir" if mean_p > 0.5 else "Yüklenen video gerçektir"
        return deepfake_rate, final_result
    else:
        return 0, "Yüz algılanamadı."


# if __name__ == "__main__":
#     video_file = r"C:\Users\admin\Desktop\yapa zeka\data\original\146.mp4"
#     print(f"Processing video: {video_file}")
#     analyze_video(video_file)
