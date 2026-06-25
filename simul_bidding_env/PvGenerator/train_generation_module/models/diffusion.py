


import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import  Dict, Optional,List,  TypeVar, Tuple


from .model_utils import (
    LastLayer, TransformerBlock, exists, extract, cosine_beta_schedule,
    Residual, SinusoidalPositionEmbeddings, ResnetBlock1d, PreNorm1d,
    LinearAttention1d, Downsample1d, Upsample1d, Attention1d
)
from einops import rearrange


Tensor = TypeVar('torch.tensor')

class DiffusionModel(nn.Module):
    """Diffusion Model for generating data."""
    def __init__(self, input_dim: int, latent_dim: int, info_dict: Dict, args, device: str = 'cpu', hidden_dims: Optional[List[int]] = None, **kwargs) -> None:
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.info_dict = info_dict
        self.args = args
        self.device = device
        self.timesteps = args.diffusion_timesteps
        self.loss_type = args.diffusion_loss_type
        self.denoise_depth = args.diffusion_denoise_depth
        self.diffusion_multi = eval(args.diffusion_multi)
        self.diffusion_init_dim = args.diffusion_init_dim
        self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.max_diff_loss = None
        vae_hidden_dims = eval(self.args.diffusion_vae_hidden)
        if hidden_dims is None:
            diffusion_multi = self.diffusion_multi if self.diffusion_multi else [1, 2, 4, 8]
            diffusion_init_dim = args.latent_dim
            num_resolution = len(diffusion_multi)
            self.num_resolution = num_resolution
            self.unet_scale = 2 ** (num_resolution - 1)
            diffusion_init_dim = diffusion_init_dim // self.unet_scale
            hidden_dims = [diffusion_init_dim * m for m in diffusion_multi]
        base_dim = hidden_dims[0]
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        ).to(device)
        self.build_vae_encoder(vae_hidden_dims, input_dim)
        self.build_vae_decoder(vae_hidden_dims)
        self.build_diffusion_model(hidden_dims, base_dim, time_dim)

    def build_vae_encoder(self, vae_hidden_dims: List[int], input_dim: int) -> None:
        """Build the VAE encoder."""
        vae_modules = []
        vae_input_dim = input_dim
        vae_h_dims = vae_hidden_dims[0]
        vae_modules.append(
            nn.Sequential(
                nn.Linear(in_features=vae_input_dim, out_features=vae_h_dims),
                nn.BatchNorm1d(vae_h_dims),
                nn.LeakyReLU()
            ).to(self.device)
        )
        for i in range(len(vae_hidden_dims) - 1):
            bn_layer_i_1 = nn.BatchNorm1d(vae_hidden_dims[i])
            bn_layer_i_2 = nn.BatchNorm1d(vae_hidden_dims[i + 1])
            trans_block = TransformerBlock(input_dim=vae_hidden_dims[i], hidden_dim=vae_hidden_dims[i], num_heads=self.args.attn_heads)
            layer_test = nn.Sequential(
                trans_block,
                bn_layer_i_1,
                nn.LeakyReLU(),
                nn.Linear(in_features=vae_hidden_dims[i], out_features=vae_hidden_dims[i + 1]),
                bn_layer_i_2,
                nn.LeakyReLU()
            ).to(self.device)
            vae_modules.append(layer_test)
        self.vae_encoder = nn.Sequential(*vae_modules).to(self.device)
        flat_multi = 1
        self.vae_fc_mu = nn.Linear(vae_hidden_dims[-1] * flat_multi, self.latent_dim).to(self.device)
        self.vae_fc_var = nn.Linear(vae_hidden_dims[-1] * flat_multi, self.latent_dim).to(self.device)

    def build_vae_decoder(self, vae_hidden_dims: List[int]) -> None:
        """Build the VAE decoder."""
        vae_modules = []
        self.vae_decoder_input = nn.Linear(self.latent_dim, vae_hidden_dims[-1]).to(self.device)
        for i in reversed(range(len(vae_hidden_dims) - 1)):
            bn_layer_i_1 = nn.BatchNorm1d(vae_hidden_dims[i + 1])
            bn_layer_i_2 = nn.BatchNorm1d(vae_hidden_dims[i])
            vae_modules.append(
                nn.Sequential(
                    TransformerBlock(input_dim=vae_hidden_dims[i + 1], hidden_dim=vae_hidden_dims[i + 1], num_heads=self.args.attn_heads),
                    bn_layer_i_1,
                    nn.LeakyReLU(),
                    nn.Linear(in_features=vae_hidden_dims[i + 1], out_features=vae_hidden_dims[i]),
                    bn_layer_i_2,
                    nn.LeakyReLU()
                ).to(self.device)
            )
        self.vae_decoder = nn.Sequential(*vae_modules).to(self.device)
        self.vae_final_layer = LastLayer(vae_hidden_dims[0], self.input_dim, self.info_dict, mode=self.args.loss_mode, args=self.args).to(self.device)

    def build_diffusion_model(self, hidden_dims: List[int], base_dim: int, time_dim: int) -> None:
        """Build the diffusion model."""
        resnet_block_groups = 8
        attn_heads = self.args.attn_heads
        attn_dim_head = hidden_dims[-1] // attn_heads
        input_channels = self.latent_dim
        self.first_layer = nn.Sequential(
            nn.Conv1d(input_channels, base_dim * self.unet_scale, 7, padding=3),
            nn.GroupNorm(resnet_block_groups, base_dim * self.unet_scale),
            nn.SiLU()
        ).to(self.device)
        dims = [base_dim, *hidden_dims]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolution = len(in_out)
        self.expand_dim_layer = nn.Conv1d(base_dim, base_dim * self.unet_scale, 3, padding=1)
        self.recover_dim_layer = nn.ModuleList()
        for i in range(num_resolution):
            self.recover_dim_layer.append(Downsample1d(base_dim))
        self.recover_dim_layer = nn.Sequential(*self.recover_dim_layer)
        block_klass = partial(ResnetBlock1d, groups=resnet_block_groups)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolution - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm1d(dim_in, LinearAttention1d(dim_in))),
                Downsample1d(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
            ]))
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm1d(mid_dim, Attention1d(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm1d(dim_out, LinearAttention1d(dim_out))),
                Upsample1d(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1)
            ]))
        self.final_res_block = block_klass(base_dim * 2, base_dim, time_emb_dim=time_dim)
        self.final_conv = nn.Sequential(
            nn.Conv1d(base_dim * self.unet_scale, base_dim, 7, padding=3),
            nn.GroupNorm(resnet_block_groups, base_dim),
            nn.SiLU()
        ).to(self.device)
        self.final_mlp = nn.Linear(base_dim * self.unet_scale, input_channels)

    def vae_dist(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Compute the VAE distribution parameters."""
        result = self.vae_encoder(x)
        mu = self.vae_fc_mu(result)
        log_var = self.vae_fc_var(result)
        return [mu, log_var]

    def vae_repara(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Perform reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the embedding."""
        mu, log_var = self.vae_dist(x)
        z = self.vae_repara(mu, log_var)
        return z

    def get_noise(self, x_emb: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Compute the noise."""
        x = x_emb
        float_time = time.to(dtype=torch.float64)
        t = self.time_mlp(float_time) if exists(self.time_mlp) else None
        x = x.unsqueeze(-1)
        x = rearrange(x, pattern='b (c h) 1 -> b c h', h=self.unet_scale)
        r = x.clone()
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = rearrange(x, 'b c h -> b (c h) 1')
        x = x.squeeze(-1)
        ret = self.final_mlp(x)
        return ret

    def embed_and_get_noise(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Embed the input and compute the noise."""
        z = self.embedding(x)
        return self.get_noise(z, time)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform the forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        ret = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return ret

    def vae_decode(self, z: torch.Tensor, end_to_embedding: bool = True) -> torch.Tensor:
        """Decode the latent vector."""
        x_recons = self.vae_decoder_input(z)
        x_recons = self.vae_decoder(x_recons)
        if not end_to_embedding:
            x_recons = self.vae_final_layer(x_recons)
        return x_recons

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass of the model."""
        x_start = args[0]
        t = torch.randint(0, self.timesteps, (x_start.shape[0],), device=self.device).long()
        noise = kwargs['noise']
        kld_weight = kwargs['M_N']
        mode = kwargs['mode']
        embed_input = x_start
        recons_target = x_start
        vae_mu, vae_log_var = self.vae_dist(embed_input)
        z_start = self.vae_repara(vae_mu, vae_log_var)
        diff_z_input = z_start.clone().detach()
        if noise is None:
            noise = torch.randn_like(diff_z_input)
        z_noisy = self.q_sample(x_start=diff_z_input, t=t, noise=noise)
        predicted_noise = self.get_noise(z_noisy, t)
        if self.loss_type == 'l1':
            diff_loss = F.l1_loss(noise, predicted_noise)
        elif self.loss_type == 'l2':
            diff_loss = F.mse_loss(noise, predicted_noise)
        elif self.loss_type == "huber":
            diff_loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        x_recons = self.vae_decode(z_start, end_to_embedding=False)
        if mode == 'direct':
            vae_recons_loss = F.mse_loss(x_recons, recons_target)
        elif mode == 'multi_head_ce':
            ce_loss = torch.nn.CrossEntropyLoss()
            vae_recons_loss = 0
            onehot_dim = 0
            for key in self.info_dict:
                start, end = self.info_dict[key]['pos']
                if self.info_dict[key]['dtype'] == 'onehot':
                    if key == 'zip_code':
                        sub_step = (end - start) // 6
                        sub_start = start
                        for i in range(6):
                            sub_end = sub_start + sub_step
                            y_label = recons_target[:, sub_start:sub_end]
                            y_label = torch.argmax(y_label, dim=-1)
                            key_loss = ce_loss(x_recons[:, sub_start:sub_end], y_label)
                            vae_recons_loss += key_loss
                    else:
                        y_label = recons_target[:, start:end]
                        y_label = torch.argmax(y_label, dim=-1)
                        key_loss = ce_loss(x_recons[:, start:end], y_label)
                        vae_recons_loss += key_loss
                    onehot_dim += (end - start)
            mse_loss = F.mse_loss(x_recons[:, onehot_dim:], x_start[:, onehot_dim:])
            vae_recons_loss += mse_loss
        vae_KL_loss = torch.mean(-0.5 * torch.sum(1 + vae_log_var - vae_mu ** 2 - vae_log_var.exp(), dim=1), dim=0)
        vae_loss = kld_weight * vae_KL_loss + vae_recons_loss
        loss = diff_loss + vae_loss
        return {'loss': loss, 'Reconstruction_Loss': vae_loss.detach(), 'KLD': diff_loss.detach()}

    @torch.no_grad()
    def p_sample(self, z: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
        """Sample from the model."""
        betas_t = extract(self.betas, t, z.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, z.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, z.shape)
        pred_noise = self.get_noise(z, t)
        model_mean = sqrt_recip_alphas_t * (z - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, z.shape)
            noise = torch.randn_like(z)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape: List[int], device: str, noise: Optional[torch.Tensor] = None, denoise_depth: Optional[int] = None, return_type: str = 'all') -> List[torch.Tensor]:
        """Perform the sampling loop."""
        batch_size, feature_dim = shape[0], shape[1]
        if noise is None:
            data = torch.randn(shape, device=device, dtype=torch.float64)
        else:
            data = noise
        if denoise_depth is None:
            denoise_depth = self.denoise_depth
        data_list = []
        for i in reversed(range(denoise_depth)):
            input_t_i = i
            input_t = torch.full((batch_size,), input_t_i, device=device, dtype=torch.long)
            data = self.p_sample(data, input_t, i)
            if return_type == 'all':
                data_list.append(data)
            del input_t
        if return_type == 'all':
            ret = data_list
        elif return_type == 'last':
            ret = data
        return ret

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Predict the start from noise."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    @torch.no_grad()
    def sample(self, sample_num: int, device: str, return_type: str = 'all') -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Sample from the model."""
        sample_fn = self.p_sample_loop
        shape = [sample_num, self.latent_dim]
        generated_z_list = sample_fn(shape, device, return_type=return_type)
        if return_type == 'all':
            generated_x_list = []
            for generated_z in generated_z_list:
                generated_x = self.vae_decode(generated_z, end_to_embedding=False)
                generated_x_list.append(generated_x)
        else:
            generated_x_list = self.vae_decode(generated_z_list, end_to_embedding=False)
        return generated_z_list, generated_x_list

    @torch.no_grad()
    def generate(self, ori_data: torch.Tensor) -> torch.Tensor:
        """Generate data from the model."""
        z = self.embedding(ori_data)
        ret = self.vae_decode(z, end_to_embedding=False)
        return ret