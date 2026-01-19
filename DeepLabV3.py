import os
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.tensorboard import SummaryWriter
import time

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.png', '_mask.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
class DeepLabV3Wrapper(nn.Module):
    def __init__(self, model):
        super(DeepLabV3Wrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['out']



def train_model(model, train_loader, test_loader, criterion, optimizer, model_save_dir,num_epochs=25):


    # -----------

    # 创建保存模型和TensorBoard日志的目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(model_save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
    # 将模型结构传到TensorBoard
    # 从train_loader中获取一个数据批次
    images, labels = next(iter(train_loader))
    # 包装模型
    model_wrapper = DeepLabV3Wrapper(model)
    writer.add_graph(model_wrapper, images.to(device))

    best_loss = float('inf')
    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)['out']
            #print(outputs.shape)
            #print(labels.shape)
            loss = criterion(outputs, labels.float())

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)


        # 验证阶段
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)['out']
                loss = criterion(outputs, labels.float())
                #loss = criterion(outputs, labels.squeeze(1).float())

                #loss = criterion(outputs, labels.unsqueeze(1).float())

                running_loss += loss.item() * inputs.size(0)

        epoch_test_loss = running_loss / len(test_loader.dataset)
        writer.add_scalar('Loss/test', epoch_test_loss, epoch)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f'Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}')
        print(f'Epoch {epoch} Duration: {epoch_duration:.2f} seconds')

        # 保存最佳模型
        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

    print('训练完成')
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f'Total Training Duration: {total_duration:.2f} seconds')
    writer.close()

if __name__ == '__main__':
    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((640, 1024)),
        transforms.ToTensor(),
    ])
    image_dir = 'dataset/images'
    mask_dir = 'dataset/masks'
    model_save_dir = 'model_save'
    # 创建数据集
    dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)

    # 自动划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -----------

    # 加载预训练的DeepLabV3模型
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))  # 修改最后的分类层，适应单通道掩模
    model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()  # 使用二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 将模型移至GPU（如果有）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # 训练模型
    train_model(model, train_loader, test_loader, criterion, optimizer,model_save_dir, num_epochs=200)
