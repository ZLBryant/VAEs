import torch
from torch import nn
from torch.optim import Adam
from torch.nn import MSELoss
from utils import img_save
from torch.optim.lr_scheduler import StepLR, LambdaLR, ExponentialLR, MultiStepLR, ReduceLROnPlateau

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        hidden_dims = [args.in_channels, 32, 64, 128, 256]
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU(inplace=True)
                )
            )
        self.encoder = nn.Sequential(*modules)
        self.linear_mu = nn.Linear(hidden_dims[-1] * 4, args.z_dim)#卷积后最后为2*2
        self.linear_log_sigma_2 = nn.Linear(hidden_dims[-1] * 4, args.z_dim)
        self.decode_input = nn.Linear(args.z_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU(inplace=True)
                )
            )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-2], hidden_dims[-1], kernel_size=3, stride=2, padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.Sigmoid()
            )
        )
        self.decoder = nn.Sequential(*modules)
        self.device = args.device
        self.z_dim = args.z_dim
        self.model_init()

    def model_init(self):
        for p in self.parameters():
            if p.data.ndim < 2:
                torch.nn.init.uniform_(p.data, -0.1, 0.1)
            else:
                torch.nn.init.kaiming_uniform_(p.data)

    def reparametrisation(self, mu, log_sigma_2):
        sigma = torch.exp(log_sigma_2 / 2)
        noise = torch.rand_like(mu)
        return mu + sigma * noise

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        mu = self.linear_mu(x)
        log_sigma_2 = self.linear_log_sigma_2(x)
        return mu, log_sigma_2

    def forward(self, x):
        mu, log_sigma_2 = self.encode(x)
        sample = self.reparametrisation(mu, log_sigma_2)
        sample = self.decode_input(sample).view(sample.shape[0], -1, 2, 2)
        x_predict = self.decoder(sample)
        x_predict = x_predict[:, :, 2:-2, 2:-2]
        return mu, log_sigma_2, x_predict

    def generate_random_samples(self, sample_num):
        #在标准正态分布上进行采样，生成样本
        z = torch.randn(sample_num, self.z_dim).to(self.device)
        z = self.decode_input(z).view(z.shape[0], -1, 2, 2)
        samples = self.decoder(z)
        return samples[:, :, 2:-2, 2:-2]

    def generate_similar_samples(self, sample, sample_num):
        #生成与sample比较相似的样本
        sample = sample.to(self.device)
        mu, log_sigma_2 = self.encode(sample)
        noise = torch.randn(sample_num, self.z_dim).to(self.device)
        sigma = torch.exp(log_sigma_2 / 2)
        samples_z = mu + sigma * noise
        samples_z = self.decode_input(samples_z).view(samples_z.shape[0], -1, 2, 2)
        similar_samples = self.decoder(samples_z)
        return similar_samples[:, :, 2:-2, 2:-2]

    def generate_interpolation_samples(self, sample1, sample2, sample_num):
        #在两个样本之间进行差值
        sample1 = sample1.to(self.device)
        sample2 = sample2.to(self.device)
        mu1, _ = self.encode(sample1)
        mu2, _ = self.encode(sample2)
        diff = (mu2 - mu1) / (sample_num - 1)
        interpolation_samples = torch.zeros(sample_num, self.z_dim).to(self.device)
        for i in range(0, sample_num):
            interpolation_samples[i, :] = mu1 + i * diff
        interpolation_samples = self.decode_input(interpolation_samples).view(interpolation_samples.shape[0], -1, 2, 2)
        interpolation_samples = self.decoder(interpolation_samples.to(self.device))
        return interpolation_samples[:, :, 2:-2, 2:-2]

    def generate_samples(self, img1, img2, args):
        #随机采样
        random_samples = self.generate_random_samples(args.sample_num)
        img_save(random_samples, args.random_samples_save_path)
        # 生成相似样本
        similar_samples = self.generate_similar_samples(img1.unsqueeze(0), args.sample_num)
        img_save(similar_samples, args.similar_samples_save_path)
        # 生成插值样本
        interpolation_samples = self.generate_interpolation_samples(img1.unsqueeze(0), img2.unsqueeze(0), args.sample_num)
        img_save(interpolation_samples, args.interpolation_samples_save_path)

def model_train(model, train_iter, val_iter, args):
    model.train()
    lr = args.lr
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = MSELoss(size_average=False)
    scheduler = MultiStepLR(optimizer, milestones=[10, 25, 40], gamma=args.lr_decay)
    min_loss = float("inf")
    patience = 0
    exit_count = 0
    for cur_epoch in range(args.epoch):
        loss = 0
        for batch_idx, batch_data in enumerate(train_iter):#batch_data:[64*1*28*28, 64]
            batch_size = batch_data[0].shape[0]
            batch_data = batch_data[0].to(args.device)
            target = batch_data.clone()
            mu, log_sigma_2, recon_result = model(batch_data)
            tmp = log_sigma_2.sum()
            recon_loss = criterion(target, recon_result)
            KL_loss = torch.sum(0.5 * (-1 - log_sigma_2 + mu ** 2 + torch.exp(log_sigma_2)))
            cur_loss = (recon_loss + KL_loss) / batch_size
            loss += cur_loss
            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()
        loss = 0
        if cur_epoch % args.eval_iter_num == 0:
            model.eval()
            for batch_idx, batch_data in enumerate(val_iter):
                batch_size = batch_data[0].shape[0]
                batch_data = batch_data[0].to(args.device)
                target = batch_data.clone()
                mu, log_sigma_2, recon_result = model(batch_data)
                recon_loss = criterion(target, recon_result)
                KL_loss = torch.sum(0.5 * (-1 - log_sigma_2 + mu ** 2 + torch.exp(log_sigma_2)))
                cur_loss = (recon_loss + KL_loss) / batch_size
                loss += cur_loss.item()
            print("cur_epoch:{0:.1f}, loss:{1:.3f}".format(cur_epoch, loss / len(val_iter)))
            if loss < min_loss:
                patience = 0
                torch.save(model, args.model_save_path)
                min_loss = loss
            else:
                patience += 1
                if patience == args.patience:
                    if exit_count == args.exit_threshold:
                        exit(0)
                    else:
                        exit_count += 1
                        patience = 0
                        model = torch.load(args.model_save_path)
                        lr = lr * args.lr_decay
                        optimizer = Adam(model.parameters(), lr=lr)
            model.train()
        '''if cur_epoch % 10 == 9:
            torch.save(model.state_dict(), model_path + 'model' + str(cur_epoch) + '.pth')
            #生成随机样本
            random_samples = model.generate_random_samples(args.sample_num)
            image_name = 'random_samples' + str(cur_epoch)
            img_save(random_samples, image_name)
            #生成相似样本
            similar_samples = model.generate_similar_samples(number_dict[8], args.sample_num)
            image_name = 'similar_samples' + str(cur_epoch)
            img_save(similar_samples, image_name)
            #生成插值样本
            interpolation_samples = model.generate_interpolation_samples(number_dict[0], number_dict[8], args.sample_num)
            image_name = 'interpolation_sample' + str(cur_epoch)
            img_save(interpolation_samples, image_name)'''

def model_test(model, test_iter):
    model.eval()
    for batch_idx, batch_data in enumerate(test_iter):
        batch_size = batch_data[0].shape[0]
        batch_data = batch_data[0].to(model.device)
        batch_data = batch_data.view(batch_size, -1)
        target = batch_data.clone()
        _, _, recon_result = model(batch_data)
        img_name = 'target' + str(batch_idx)
        img_save(target, img_name)
        img_name = 'recon_result' + str(batch_idx)
        img_save(recon_result, img_name)
