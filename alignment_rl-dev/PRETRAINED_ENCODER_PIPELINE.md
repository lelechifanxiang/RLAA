# 预训练PSF/波前编码器的Pipeline设计

## 🎯 核心思路

**将高维PSF/波前压缩为低维特征向量，再用于RL训练**

```
Pipeline概览：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
阶段1：预训练编码器（离线）
  PSF (81,920维) → CNN编码器 → 特征向量 (32维)
  
阶段2：冻结编码器 + RL训练（在线）
  PSF → 预训练编码器(冻结) → 特征向量 → RL策略

或者：端到端微调
  PSF → 预训练编码器(可微调) → 特征向量 → RL策略
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📐 方案1：PSF自监督预训练（推荐）

### 1.1 编码器架构

```python
class PSFEncoder(nn.Module):
    """PSF图像编码器：将5个视场的PSF压缩为紧凑特征向量"""
    
    def __init__(self, latent_dim=32):
        super().__init__()
        
        # 输入：5个视场 × 128×128 PSF
        self.encoder = nn.Sequential(
            # Conv层：逐渐降维
            nn.Conv2d(5, 32, kernel_size=7, stride=2, padding=3),   # 128→64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # 64→32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 32→16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 16→8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 全局平均池化：8×8→1×1
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # 全连接层：压缩到latent_dim
            nn.Linear(256, latent_dim),
        )
        
    def forward(self, psf_batch):
        """
        输入：(batch, 5_fields, 128, 128)
        输出：(batch, latent_dim)
        """
        return self.encoder(psf_batch)


class PSFDecoder(nn.Module):
    """PSF图像解码器：重建PSF（用于预训练）"""
    
    def __init__(self, latent_dim=32):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            
            # 反卷积：逐渐升维
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # 8→16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 16→32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1),   # 32→64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 5, 7, stride=2, padding=3, output_padding=1),    # 64→128
            nn.Sigmoid(),  # PSF范围[0,1]
        )
        
    def forward(self, latent):
        """
        输入：(batch, latent_dim)
        输出：(batch, 5_fields, 128, 128)
        """
        return self.decoder(latent)


class PSFAutoEncoder(nn.Module):
    """完整的PSF自编码器"""
    
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = PSFEncoder(latent_dim)
        self.decoder = PSFDecoder(latent_dim)
        
    def forward(self, psf_batch):
        latent = self.encoder(psf_batch)
        reconstruction = self.decoder(latent)
        return reconstruction, latent
```

---

### 1.2 预训练任务1：自编码重建

**目标**：学习压缩PSF的紧凑表示

```python
def pretrain_psf_autoencoder(
    env_config: LensEnvConfig,
    latent_dim: int = 32,
    num_samples: int = 50_000,
    batch_size: int = 256,
    epochs: int = 50,
):
    """预训练PSF自编码器"""
    
    # 1. 收集训练数据：采样大量PSF
    print("收集PSF训练数据...")
    env = LensAlignmentEnv(env_config)
    psf_dataset = []
    
    for _ in tqdm(range(num_samples)):
        # 随机reset + 随机动作
        env.reset()
        for _ in range(np.random.randint(1, 10)):
            action = env.action_space.sample()
            env.step(action)
        
        # 获取当前PSF（需要修改环境暴露PSF）
        psf = env._mgr.get_current_psf()  # shape: (5, 128, 128)
        psf_dataset.append(psf)
    
    psf_dataset = torch.stack(psf_dataset)  # (50000, 5, 128, 128)
    
    # 2. 创建DataLoader
    dataset = TensorDataset(psf_dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. 训练自编码器
    model = PSFAutoEncoder(latent_dim).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for (psf_batch,) in dataloader:
            psf_batch = psf_batch.to('cuda')
            
            # 前向传播
            reconstruction, latent = model(psf_batch)
            loss = criterion(reconstruction, psf_batch)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # 4. 保存编码器
    torch.save(model.encoder.state_dict(), 'psf_encoder.pth')
    print("✓ PSF编码器已保存")
    
    return model.encoder
```

---

### 1.3 预训练任务2：对比学习（更好）

**思路**：相似对准状态 → 相似PSF特征

```python
class PSFContrastiveEncoder(nn.Module):
    """对比学习PSF编码器（SimCLR风格）"""
    
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = PSFEncoder(latent_dim)
        
        # 投影头（对比学习专用）
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim // 2),
        )
        
    def forward(self, psf_batch):
        features = self.encoder(psf_batch)
        projections = self.projection_head(features)
        return features, projections


def pretrain_psf_contrastive(
    env_config: LensEnvConfig,
    latent_dim: int = 32,
    num_samples: int = 50_000,
    batch_size: int = 256,
    temperature: float = 0.5,
):
    """对比学习预训练"""
    
    # 1. 收集训练数据：(PSF, 对准状态, MTF质量)
    print("收集对比学习数据...")
    env = LensAlignmentEnv(env_config)
    dataset = []
    
    for _ in tqdm(range(num_samples)):
        env.reset()
        for _ in range(np.random.randint(1, 20)):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            psf = env._mgr.get_current_psf()
            state = info['state']  # (dx, dy)
            quality = info['quality_metric']
            
            dataset.append({
                'psf': psf,
                'state': state,
                'quality': quality,
            })
    
    # 2. 构建正负样本对
    def create_contrastive_pairs(dataset, batch_size):
        """构建正负样本对"""
        batch_indices = np.random.choice(len(dataset), batch_size)
        
        anchors = []
        positives = []
        
        for idx in batch_indices:
            anchor_data = dataset[idx]
            
            # 正样本：质量相似且状态接近的样本
            anchor_state = anchor_data['state']
            anchor_quality = anchor_data['quality']
            
            # 找到距离最近的样本
            distances = [
                np.linalg.norm(anchor_state - dataset[i]['state'])
                + abs(anchor_quality - dataset[i]['quality'])
                for i in range(len(dataset))
            ]
            positive_idx = np.argpartition(distances, 10)[1:11]  # 排除自己，取前10
            positive_idx = np.random.choice(positive_idx)
            
            anchors.append(anchor_data['psf'])
            positives.append(dataset[positive_idx]['psf'])
        
        return torch.stack(anchors), torch.stack(positives)
    
    # 3. 对比学习训练
    model = PSFContrastiveEncoder(latent_dim).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    def contrastive_loss(anchor_proj, positive_proj, temperature):
        """NT-Xent损失（归一化温度交叉熵）"""
        # 计算相似度矩阵
        batch_size = anchor_proj.shape[0]
        
        # 拼接anchor和positive
        features = torch.cat([anchor_proj, positive_proj], dim=0)  # (2B, D)
        
        # 计算余弦相似度
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)  # (2B, 2B)
        similarity_matrix = similarity_matrix / temperature
        
        # 构建标签：anchor_i的正样本是positive_i
        labels = torch.arange(batch_size).to('cuda')
        labels = torch.cat([labels + batch_size, labels])  # [B, B+1, ..., 2B-1, 0, 1, ..., B-1]
        
        # 计算交叉熵损失
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to('cuda')
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
    
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        num_batches = num_samples // batch_size
        
        for _ in range(num_batches):
            anchors, positives = create_contrastive_pairs(dataset, batch_size)
            anchors, positives = anchors.to('cuda'), positives.to('cuda')
            
            # 前向传播
            anchor_features, anchor_proj = model(anchors)
            positive_features, positive_proj = model(positives)
            
            # 计算对比损失
            loss = contrastive_loss(anchor_proj, positive_proj, temperature)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Contrastive Loss: {avg_loss:.4f}")
    
    # 4. 保存编码器（不包含投影头）
    torch.save(model.encoder.state_dict(), 'psf_encoder_contrastive.pth')
    print("✓ 对比学习PSF编码器已保存")
    
    return model.encoder
```

---

### 1.4 预训练任务3：监督预训练（最直接）

**思路**：PSF → 预测对准状态/质量

```python
def pretrain_psf_supervised(
    env_config: LensEnvConfig,
    latent_dim: int = 32,
    num_samples: int = 50_000,
):
    """监督预训练：预测对准状态和质量"""
    
    # 1. 收集标注数据
    print("收集监督学习数据...")
    env = LensAlignmentEnv(env_config)
    X_psf = []
    y_state = []
    y_quality = []
    
    for _ in tqdm(range(num_samples)):
        env.reset()
        for _ in range(np.random.randint(1, 20)):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            psf = env._mgr.get_current_psf()
            state = info['state']  # (dx, dy)
            quality = info['quality_metric']
            
            X_psf.append(psf)
            y_state.append(state)
            y_quality.append(quality)
    
    X_psf = torch.stack(X_psf)
    y_state = torch.tensor(y_state, dtype=torch.float32)
    y_quality = torch.tensor(y_quality, dtype=torch.float32).unsqueeze(1)
    
    # 2. 创建监督模型
    class PSFSupervisedEncoder(nn.Module):
        def __init__(self, latent_dim=32):
            super().__init__()
            self.encoder = PSFEncoder(latent_dim)
            
            # 预测头
            self.state_predictor = nn.Linear(latent_dim, 2)     # 预测(dx, dy)
            self.quality_predictor = nn.Linear(latent_dim, 1)   # 预测q
            
        def forward(self, psf):
            features = self.encoder(psf)
            state_pred = self.state_predictor(features)
            quality_pred = self.quality_predictor(features)
            return features, state_pred, quality_pred
    
    # 3. 训练
    model = PSFSupervisedEncoder(latent_dim).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    dataset = TensorDataset(X_psf, y_state, y_quality)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    for epoch in range(50):
        total_state_loss = 0
        total_quality_loss = 0
        
        for psf_batch, state_batch, quality_batch in dataloader:
            psf_batch = psf_batch.to('cuda')
            state_batch = state_batch.to('cuda')
            quality_batch = quality_batch.to('cuda')
            
            # 前向传播
            features, state_pred, quality_pred = model(psf_batch)
            
            # 多任务损失
            state_loss = F.mse_loss(state_pred, state_batch)
            quality_loss = F.mse_loss(quality_pred, quality_batch)
            loss = state_loss + quality_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_state_loss += state_loss.item()
            total_quality_loss += quality_loss.item()
        
        print(f"Epoch {epoch+1}, State Loss: {total_state_loss/len(dataloader):.4f}, "
              f"Quality Loss: {total_quality_loss/len(dataloader):.4f}")
    
    # 4. 保存编码器（不包含预测头）
    torch.save(model.encoder.state_dict(), 'psf_encoder_supervised.pth')
    print("✓ 监督学习PSF编码器已保存")
    
    return model.encoder
```

---

## 🔗 集成到RL训练

### 2.1 修改环境以暴露PSF编码特征

```python
class LensAlignmentEnvWithPSFEncoder(LensAlignmentEnv):
    """使用预训练PSF编码器的环境"""
    
    def __init__(self, cfg: LensEnvConfig, encoder_path: str):
        super().__init__(cfg)
        
        # 加载预训练编码器
        self.psf_encoder = PSFEncoder(latent_dim=32)
        self.psf_encoder.load_state_dict(torch.load(encoder_path))
        self.psf_encoder.eval()  # 冻结模式
        self.psf_encoder.to('cuda')
        
        # 修改观测空间
        # 原始：300 (MTF历史) + 20 (动作历史) = 320
        # 新增：32 (PSF特征) × 10步 = 320
        # 总计：320 + 320 = 640维
        self._psf_feature_dim = 32
        self._psf_feature_buffer = np.zeros(
            (self._mtf_history_len, self._psf_feature_dim),
            dtype=np.float32
        )
        
        # 更新观测空间维度
        self._obs_dim = (
            self._mtf_history_len * self._n_mtf +
            self._action_history_len * self._n_action +
            self._mtf_history_len * self._psf_feature_dim  # 新增
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32
        )
    
    def _get_obs(self) -> np.ndarray:
        """返回观测：MTF历史 + 动作历史 + PSF特征历史"""
        obs_parts = [
            self._mtf_obs_buffer.flatten(),
            self._action_buffer.flatten(),
            self._psf_feature_buffer.flatten(),  # 新增
        ]
        return np.concatenate(obs_parts).astype(np.float32)
    
    def _update_psf_features(self):
        """计算当前PSF的编码特征"""
        # 获取当前PSF
        psf = self._mgr.get_current_psf()  # (5, 128, 128)
        psf_tensor = torch.from_numpy(psf).unsqueeze(0).to('cuda')  # (1, 5, 128, 128)
        
        # 编码
        with torch.no_grad():
            features = self.psf_encoder(psf_tensor)  # (1, 32)
        
        features_np = features.cpu().numpy().squeeze()  # (32,)
        
        # 更新历史buffer（滚动）
        self._psf_feature_buffer[:-1] = self._psf_feature_buffer[1:]
        self._psf_feature_buffer[-1] = features_np
    
    def step(self, action):
        # 原始step逻辑
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 更新PSF特征
        self._update_psf_features()
        
        # 返回新观测
        new_obs = self._get_obs()
        return new_obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        
        # 初始化PSF特征buffer
        self._update_psf_features()
        # 填充历史（重复第一帧）
        first_feature = self._psf_feature_buffer[-1]
        self._psf_feature_buffer[:] = first_feature
        
        new_obs = self._get_obs()
        return new_obs, info
```

---

### 2.2 训练脚本

```python
def train_with_pretrained_psf_encoder():
    """使用预训练PSF编码器进行RL训练"""
    
    # 阶段1：预训练编码器（如果还没有）
    if not os.path.exists('psf_encoder_supervised.pth'):
        print("=" * 60)
        print("阶段1：预训练PSF编码器")
        print("=" * 60)
        
        env_config = make_lens_rl_config()
        encoder = pretrain_psf_supervised(
            env_config,
            latent_dim=32,
            num_samples=50_000,
        )
    
    # 阶段2：RL训练
    print("\n" + "=" * 60)
    print("阶段2：RL训练（使用预训练编码器）")
    print("=" * 60)
    
    # 创建环境（使用预训练编码器）
    env_config = make_lens_rl_config()
    train_env = make_vec_env(
        lambda: LensAlignmentEnvWithPSFEncoder(
            env_config,
            encoder_path='psf_encoder_supervised.pth'
        ),
        n_envs=12,
    )
    
    # SAC训练（观测维度变为640）
    model = SAC(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(net_arch=[256, 256]),  # 可能需要更大网络
        verbose=1,
    )
    
    model.learn(total_timesteps=300_000)
    model.save("sac_with_psf_encoder")
    
    print("✓ 训练完成")
```

---

### 2.3 可选：端到端微调

```python
class LensAlignmentEnvWithFinetunablePSFEncoder(LensAlignmentEnv):
    """允许微调PSF编码器的环境"""
    
    def __init__(self, cfg: LensEnvConfig, encoder_path: str, finetune: bool = True):
        super().__init__(cfg)
        
        # 加载预训练编码器
        self.psf_encoder = PSFEncoder(latent_dim=32)
        self.psf_encoder.load_state_dict(torch.load(encoder_path))
        
        if finetune:
            self.psf_encoder.train()  # 允许梯度更新
            # 但使用小学习率
            self.encoder_optimizer = torch.optim.Adam(
                self.psf_encoder.parameters(),
                lr=1e-5  # 比RL策略学习率小1-2个数量级
            )
        else:
            self.psf_encoder.eval()
            self.encoder_optimizer = None
        
        self.psf_encoder.to('cuda')
        # ... 其余同上
    
    def update_encoder(self, td_error):
        """根据TD误差微调编码器"""
        if self.encoder_optimizer is not None:
            # 使用TD误差作为监督信号
            encoder_loss = td_error.pow(2).mean()
            
            self.encoder_optimizer.zero_grad()
            encoder_loss.backward()
            self.encoder_optimizer.step()


# 需要自定义RL算法以支持编码器微调
# 这比较复杂，建议先尝试冻结版本
```

---

## 🎯 方案2：波前Zernike系数预训练（更简单）

### 波前方案的优势

```python
class WavefrontEncoder(nn.Module):
    """波前Zernike系数编码器（更简单）"""
    
    def __init__(self, n_fields=5, n_zernike=15, latent_dim=32):
        super().__init__()
        
        # 输入：5场 × 15项Zernike = 75维（已经很紧凑）
        self.encoder = nn.Sequential(
            nn.Linear(n_fields * n_zernike, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
    
    def forward(self, zernike_coeffs):
        """
        输入：(batch, 5_fields, 15_zernike)
        输出：(batch, latent_dim)
        """
        batch_size = zernike_coeffs.shape[0]
        flat = zernike_coeffs.reshape(batch_size, -1)  # (batch, 75)
        return self.encoder(flat)


def pretrain_wavefront_encoder(env_config, num_samples=50_000):
    """预训练波前编码器（监督学习）"""
    
    # 收集数据
    env = LensAlignmentEnv(env_config)
    X_zernike = []
    y_quality = []
    
    for _ in tqdm(range(num_samples)):
        env.reset()
        for _ in range(np.random.randint(1, 20)):
            action = env.action_space.sample()
            _, _, _, _, info = env.step(action)
            
            # 计算Zernike系数（需要实现）
            zernike = env._mgr.compute_zernike_coefficients()  # (5, 15)
            quality = info['quality_metric']
            
            X_zernike.append(zernike)
            y_quality.append(quality)
    
    # 训练encoder预测质量
    model = WavefrontEncoder(latent_dim=32)
    quality_head = nn.Linear(32, 1)
    
    # ... 训练逻辑同上
    
    torch.save(model.state_dict(), 'wavefront_encoder.pth')
    return model
```

**波前方案更简单**：
- 输入已经是75维（vs PSF的82k维）
- 不需要复杂CNN
- 编码器更轻量（3层MLP）

---

## 📊 三种方案对比

| 方案 | 预训练数据 | 预训练时间 | 编码器复杂度 | RL训练速度 | 推荐度 |
|-----|-----------|-----------|-------------|-----------|--------|
| **PSF自编码** | 5万PSF | ~2小时 | CNN(重) | 慢(-30%) | ⭐⭐⭐ |
| **PSF对比学习** | 5万PSF对 | ~3小时 | CNN(重) | 慢(-30%) | ⭐⭐⭐⭐ |
| **PSF监督** | 5万标注 | ~2小时 | CNN(重) | 慢(-30%) | ⭐⭐⭐⭐⭐ |
| **波前监督** | 5万标注 | ~0.5小时 | MLP(轻) | 快(-5%) | ⭐⭐⭐⭐⭐ |

---

## 💡 最终推荐

### 优先级排序

```
优先级1（立即）：修复环境配置 ⭐⭐⭐⭐⭐
  └─ 不需要任何预训练

优先级2（条件性）：波前监督预训练 ⭐⭐⭐⭐
  ├─ 前提：环境配置修复后仍不够
  ├─ 优势：简单、快速、轻量
  └─ 实现：1-2天

优先级3（实验性）：PSF监督预训练 ⭐⭐⭐
  ├─ 前提：波前方案仍不够
  ├─ 优势：信息最丰富
  └─ 实现：3-5天

不推荐：PSF自编码/对比学习 ⭐⭐
  └─ 监督学习更直接有效
```

---

## 🚀 实施建议

### 如果要尝试预训练编码器

**推荐路线**：
1. **先实现波前方案**（更简单）
   - 在optics_core中添加Zernike拟合
   - 预训练轻量编码器（75→32维）
   - 集成到环境（obs: 320+320=640维）

2. **如果波前不够，再尝试PSF**
   - 使用监督预训练（最直接）
   - 先冻结编码器训练RL
   - 如果需要再尝试端到端微调

### 关键注意事项

✅ **预训练数据要多样化**
   - 覆盖整个对准空间（±0.8mm）
   - 包含各种公差实现
   - 包含不同质量水平

✅ **监督信号要准确**
   - 使用info['quality_metric']而非reward
   - 使用info['state']而非估计值

✅ **编码器要适度**
   - latent_dim=32-64（不要太大）
   - 避免过拟合（dropout/L2正则）

✅ **RL训练要调整**
   - 观测维度变大，可能需要更大网络
   - 学习率可能需要调整
   - Buffer size可能需要增加

---

**总结**：这个pipeline在理论上完全可行，但实施成本较高。建议先尝试简单方法（环境配置+状态观测），确认需要后再投入预训练编码器。
