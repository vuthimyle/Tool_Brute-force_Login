import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split


def preprocess_image(image_path, img_width=80, img_height=30):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype('float32') / 255.0
    return img

def create_model(img_width, img_height, num_characters, char_list):
    class CaptchaModel(nn.Module):
        def __init__(self):
            super(CaptchaModel, self).__init__()
            self.conv_layers = nn.Sequential(
                # First Conv Block - increased filters, added batch norm
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.2),
                
                # Second Conv Block
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.2),
                
                # Third Conv Block
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.2)
            )
            
            conv_output_size = (img_width // 8) * (img_height // 8) * 256
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(conv_output_size, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, len(char_list) * num_characters)
            )
            
        def forward(self, x):
            x = self.conv_layers(x)
            x = self.classifier(x)
            return x.view(-1, num_characters, len(char_list))
    
    return CaptchaModel()

def train_model():
    img_width = 160
    img_height = 60
    num_characters = 4
    char_list = '0123456789'
    epochs = 70  # Increased epochs
    batch_size = 64  # Increased batch size
    
    X = []  # Images
    y = []  # Labels
    data_dir = 'capcha'
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):
            print(f"Processing {filename}")
            label = os.path.splitext(filename)[0]  # Get label from filename (1234.png -> 1234)
            if len(label) == num_characters and label.isdigit():
                img_path = os.path.join(data_dir, filename)
                img = preprocess_image(img_path, img_width, img_height)
                X.append(img)
                y.append([int(c) for c in label])

    X = np.array(X)
    y = np.array(y)
    
    # Reshape X for PyTorch (batch_size, channels, height, width)
    X = X.reshape(-1, 1, img_height, img_width)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    model = create_model(img_width, img_height, num_characters, char_list)
    criterion = nn.CrossEntropyLoss()
    
    # Added learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        
        # Mini-batch training
        indices = torch.randperm(len(X_train))
        total_loss = 0
        num_batches = len(X_train) // batch_size
        
        for i in range(num_batches):
            batch_indices = indices[i*batch_size:(i+1)*batch_size]
            batch_x = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = sum(criterion(outputs[:, j], batch_y[:, j]) for j in range(num_characters))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = sum(criterion(val_outputs[:, i], y_test[:, i]) for i in range(num_characters))
            
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Load best model before saving
    model.load_state_dict(torch.load('best.pth'))
    torch.save(model.state_dict(), 'best_model_32.pth')

def predict_captcha(model, image_bytes, char_list):
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Preprocess
        img = cv2.resize(img, (160, 60))  # Match training dimensions
        img = img.astype('float32') / 255.0
        img = img.reshape(1, 1, 60, 160)  # [batch_size, channels, height, width]
        
        # Predict
        with torch.no_grad():
            outputs = model(torch.FloatTensor(img))
            _, predicted = torch.max(outputs.data, 2)
            predicted = predicted.numpy()[0]
            
        # Convert indices to characters
        result = ''.join([char_list[i] for i in predicted])
        return result
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

if __name__ == "__main__":
    train_model()