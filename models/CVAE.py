import torch
from torch import nn
from torch.optim import Adam
from torch.nn import MSELoss
from utils import img_save, to_onehot
from torch.optim.lr_scheduler import StepLR, LambdaLR, ExponentialLR, MultiStepLR, ReduceLROnPlateau

class CVAE(nn.Module):
    def __init__(self, args):
        super(CVAE, self).__init__()
        hidden_dims = [args.in_channels + 1, 32, 64, 128, 256]
        modules = []
        self.class_num = args.class_num
        self.class_emb = nn.Linear(args.class_num, args.img_size)
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
        self.decode_input = nn.Linear(args.z_dim + args.class_num, hidden_dims[-1] * 4)
        hidden_dims[0] -= 1
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

    def encode(self, x, c):
        c = self.class_emb(c).view(x.shape[0], -1, x.shape[2], x.shape[3])
        x = torch.cat((x, c), dim = 1)
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        mu = self.linear_mu(x)
        log_sigma_2 = self.linear_log_sigma_2(x)
        return mu, log_sigma_2

    def forward(self, x, c):
        c = to_onehot(c, self.device)
        mu, log_sigma_2 = self.encode(x, c)
        sample = self.reparametrisation(mu, log_sigma_2)
        sample = torch.cat((sample, c), dim=-1)
        sample = self.decode_input(sample).view(sample.shape[0], -1, 2, 2)
        x_predict = self.decoder(sample)
        x_predict = x_predict[:, :, 2:-2, 2:-2]
        return mu, log_sigma_2, x_predict

    def generate_class_samples(self, c, sample_num):
        #在标准正态分布上进行采样，生成样本
        c = c.repeat(sample_num, 1)
        z = torch.randn(sample_num, self.z_dim).to(self.device)
        z = torch.cat((z, c), dim=1)
        z = self.decode_input(z).view(z.shape[0], -1, 2, 2)
        samples = self.decoder(z)
        return samples[:, :, 2:-2, 2:-2]

    def generate_interpolation_samples(self, c1, c2, sample_num):
        #在两个样本之间进行差值
        diff = (c1 - c2) / (sample_num - 1)
        z = torch.zeros(sample_num, self.z_dim).to(self.device)
        interpolation_samples = torch.zeros(sample_num, self.z_dim + self.class_num).to(self.device)
        for i in range(0, sample_num):
            cur_c = c2 + i * diff
            interpolation_samples[i, :] = torch.cat((z[i, :], cur_c[0, :]), dim=-1)
        interpolation_samples = self.decode_input(interpolation_samples).view(interpolation_samples.shape[0], -1, 2, 2)
        interpolation_samples = self.decoder(interpolation_samples.to(self.device))
        return interpolation_samples[:, :, 2:-2, 2:-2]

    def generate_samples(self, c1, c2, args):
        c1 = torch.LongTensor([c1]).to(self.device)
        c2 = torch.LongTensor([c2]).to(self.device)
        c1 = to_onehot(c1, self.device)
        c2 = to_onehot(c2, self.device)
        #对当前类别随机采样
        random_samples = self.generate_class_samples(c1, args.sample_num)
        img_save(random_samples, args.random_samples_save_path)
        # 生成插值样本
        interpolation_samples = self.generate_interpolation_samples(c1, c2, args.sample_num)
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
            label = batch_data[1].to(args.device)
            batch_data = batch_data[0].to(args.device)
            target = batch_data.clone()
            mu, log_sigma_2, recon_result = model(batch_data, label)
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
                label = batch_data[1].to(args.device)
                batch_data = batch_data[0].to(args.device)
                target = batch_data.clone()
                mu, log_sigma_2, recon_result = model(batch_data, label)
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
