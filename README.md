实验 1：重建图片 
做了什么拿一批训练集里真实的 MNIST 手写数字图片，完整跑一遍整个 VAE 网络：
真实图片送入 Encoder，得到mu,log_var
重参数采样得到隐向量 z
z 送入 Decoder，输出重建图片recon
把「原图」和「重建出来的图」上下并排画出来。
这对应 VAE 的训练时的行为训练的时候就是干这件事：输入真实图片，尽量把自己重建回来。
GT（真值）就是输入的原图。
Loss 里面的重建损失，就是在缩小recon和原图的差距。
现象重建出来的数字大体和原图一致，但是会略微模糊。
CS231n 考点：VAE 生成 / 重建图片模糊，来自 KL 正则带来的近似误差。
⚠️注意：这不是生成新图片！这只是把已经见过的图片，先编码再解码复原，叫重建，不是生成。实验 2：随机生成新图片 Generate（VAE 作为生成模型的核心实验）python运行sample_z = torch.randn(16, latent_dim).to(device)
gen_imgs = model.decoder(sample_z)   # 只调用decoder！encoder完全不用
做了什么完全不输入任何真实图片，Encoder 直接不用！
直接从标准高斯分布 \(\mathcal N(0,I)\) 随机采样 16 个隐向量sample_z
只把随机 z 喂给训练好的 Decoder
Decoder 输出 16 张之前训练集里不一定存在的手写数字。

为什么可以这么干？训练时 KL 散度强迫所有图片对应的后验分布靠近标准高斯。
于是整个高斯球内任意 z，Decoder 都能输出合理图像。
这就是 VAE 叫生成模型的证据
实验 1 是 “复原见过的图”；实验 2 是 “创造没见过的图”。
AE 普通自编码器做不到这一步：随机 z 丢进 AE 的 Decoder 只会输出噪声乱图。
现象产出 16 张手写数字，有的清晰有的模糊，个别会像 “半成数字”。实验 3：隐空间插值 Latent Space Interpolationpython运行# 取两张图，得到它们的mu（确定性隐向量，不用采样z）
mu1,_ = torch.chunk(model.encoder(h1),2,dim=1)
mu2,_ = torch.chunk(model.encoder(h2),2,dim=1)
# 在mu1、mu2之间线性插值
z_interp = (1‑alpha)*mu1 + alpha*mu2
img_interp = model.decoder(z_interp)
做了什么
拿两张完全不同的 MNIST 图片，比如一张数字 0、一张数字 1。
通过 Encoder 拿到两张图的均值 mu（不用带噪声的采样 z，mu 是图片确定性表征）。
在 mu1 和 mu2 之间做线性插值，生成一系列中间隐向量。
每一个中间 z 都送入 Decoder，把图片画出来。
现象图片会平滑渐变，从第一张数字慢慢过渡变成第二张数字，不会突然跳变。
