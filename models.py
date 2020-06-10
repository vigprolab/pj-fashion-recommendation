import os
import numpy
import cv2
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch import optim
import utils.transforms
import dataset
import nets
import pickle


class FRModel(object):
    def __init__(self, param):
        super(FRModel, self).__init__()
        self.device = param['device']
        self.model_path = param['model_path']
        self.param_path = param['param_path']

        if param['mode'] == 'train':
            self.img_size = param['img_size']

            self.train_data_transform = transforms.Compose([
                utils.transforms.Resize(param['img_size']),
                utils.transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

            self.dataset_path = param['dataset_path']
            self.batch_size = param['batch_size']

            self.train_dataset = dataset.FRDataset(dataset_path=param['dataset_path'], transform=self.train_data_transform)
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

            self.n_classes = len(self.train_dataset)
            self.num_fout1 = param['num1']
            self.num_fout2 = param['num2']
            self.num_fout3 = param['num3']

            self.net = nets.FRNet(param['img_size'], len(self.train_dataset), param['num1'], param['num2'], param['num3']).to(param['device'])

            self.learning_rate = param['lr']
            self.optimizer = optim.Adam(self.net.parameters(), param['lr'])
            self.criterion = nn.CrossEntropyLoss()

            self.save_train_parameters()

        elif param['mode'] == 'test':
            t_param = self.load_train_parameters()
            self.dataset_path = t_param['dataset_path']

            self.test_data_transform = transforms.Compose([
                utils.transforms.GetFace(),
                utils.transforms.Resize(t_param['img_size']),
                utils.transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

            self.net = nets.FRNet(t_param['img_size'], t_param['n_classes'], t_param['num1'], t_param['num2'], t_param['num3']).to(param['device'])

    def train(self,epoch):
        self.net.train()

        for times in range(epoch):
            running_loss = 0.0

            for idx, data in enumerate(self.train_loader):
                #入力データ・ラベル
                batch_faces = data["face"].to(self.device)
                batch_labels = data["lbl"].to(self.device)

                # optimizerの初期化 -> 順伝播 -> Lossの計算 -> 逆伝播 -> パラメータの更新
                self.optimizer.zero_grad()
                outputs = self.net(batch_faces)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()

                # lossの表示
                running_loss += loss.item()
                if idx % 1000 == 999:    # 1000回毎にprint
                    print('[%d, %5d] loss: %.3f' %
                        (times + 1, idx + 1, running_loss / 1000))
                    running_loss = 0.0

        # モデルの保存
        self.save_model()
        print('finished training')

    def test(self, test_img_type, test_img_path, num_recom):
        # test画像の読み込み
        if test_img_type == 'take':
            self.save_frame_camera_key(0, test_img_path)
            test_img = cv2.imread(test_img_path)
        elif test_img_type == 'choose':
            test_img = cv2.imread(test_img_path)

        # test画像の前処理
        test_img = self.test_data_transform(test_img)

        # VGG16にtest_imgを入力する際に生じるデータサイズの差異の解消
        test_img = test_img.numpy().copy()  # tensor -> numpy
        test_img = torch.from_numpy(test_img[None, :, :, :]).float()  # 次元追加, numpy -> tensor

        # モデルの読み込み
        self.load_model(self.net)
        self.net.eval()
        with torch.no_grad():
            # 順伝搬
            output = self.net(test_img.to(self.device))

        # レコメンド画像の保存
        self.save_recom_img(num_recom, output)

    def save_model(self):
        """Save model to self.model_path"""
        torch.save(self.net.state_dict(), self.model_path)

    def load_model(self, model):
        """Load model from self.model_path"""
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))

    def save_frame_camera_key(self, device_num, img_path, delay=1, window_name='frame'):
        """Take and save a test image"""
        cap = cv2.VideoCapture(device_num)

        if not cap.isOpened():
            return

        while True:
            ret, frame = cap.read()
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('c'):
                cv2.imwrite(img_path, frame)
            elif key == ord('q'):
                break

        cv2.destroyWindow(window_name)

    def save_recom_img(self, num_recom, output):
        """Save recommended fashion images"""
        output = F.softmax(output, dim=1)  # 以降，レコメンドクラスの取得を行いやすくするための処理

        # レコメンドクラス(レコメンド画像のディレクトリ名)の取得
        recom_list = []
        output = output.to(torch.device('cpu'))
        output_numpy = output.numpy().copy()
        for i in range(num_recom):
            max_idx = output_numpy.argmax()
            recom_list.append(max_idx + 1)
            output_numpy[0][max_idx] = 0

        # 各レコメンドクラスごとファッション画像を取得
        for i, l in enumerate(recom_list):
            file_list = []
            for file in os.listdir(self.dataset_path + '/' + str(l)):  # レコメンドディレクトリ内の画像ファイルの取得
                if not file.startswith('.'):  # 隠しファイルの除去
                    file_list.append(file)

            for f in file_list:  # レコメンドディレクトリのファッション画像の保存
                fname_without_ext = os.path.splitext(f)[0]
                if fname_without_ext == '1':  # ファッション画像のみ
                    file_path = self.dataset_path + '/' + str(l) + '/' + f  # レコメンドディレクトリのファッション画像のパス
                    shutil.copyfile(file_path, './results/recom_img{}.png'.format(i+1))  # レコメンドファッション画像を./resultにコピー
        print('saved recommended fashion images')

    def save_train_parameters(self):
        """Save train parameters """
        with open(self.param_path, 'wb') as parameters:
            p = {
                'img_size': self.img_size,
                'dataset_path': self.dataset_path,
                'n_classes': self.n_classes,
                'num1': self.num_fout1,
                'num2': self.num_fout2,
                'num3': self.num_fout3}
            pickle.dump(p, parameters)

    def load_train_parameters(self):
        """Load train parameters"""
        with open(self.param_path, 'br') as parameters:
            param = pickle.load(parameters)
        return param
