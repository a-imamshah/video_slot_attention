from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from utils import Tensor
from utils import assert_shape
from utils import build_grid
from utils import conv_transpose_out_shape


class SlotAttention(nn.Module):
    def __init__(self, in_features, num_iterations, num_slots, slot_size, mlp_hidden_size, epsilon=1e-8):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(self.in_features)
        self.norm_mem_slots = nn.LayerNorm(self.in_features)
        # I guess this is layer norm across each slot? should look into this
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k1 = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_v1 = nn.Linear(self.slot_size, self.slot_size, bias=False)
        # Linear maps for the memory attention module.
        self.project_k2 = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_v2 = nn.Linear(self.slot_size, self.slot_size, bias=False)

        self.mlp_i = nn.Sequential(
            nn.Linear(self.slot_size*2, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        self.mlp_m = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        self.register_buffer(
            "slots_mu",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        self.register_buffer(
            "slots_log_sigma",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )

        self.register_buffer(
            "mem_slots_mu",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        self.register_buffer(
            "mem_slots_log_sigma",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )

        # Linear maps for the attention module.
        self.project_qm = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_km = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_vm = nn.Linear(self.slot_size, self.slot_size, bias=False)

    def forward(self, inputs: Tensor, memory_slots, instant_slots):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        batch_size, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.

        batch_size, num_mem_slots, mem_slot_size = memory_slots.shape
        memory_slots = self.norm_mem_slots(memory_slots)  # Apply layer norm to the input.
        
        k1 = self.project_k1(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(k1.size(), (batch_size, num_inputs, self.slot_size))
        v1 = self.project_v1(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(v1.size(), (batch_size, num_inputs, self.slot_size))

        k2 = self.project_k2(memory_slots)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(k2.size(), (batch_size, num_mem_slots, self.slot_size))
        v2 = self.project_v2(memory_slots)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(v2.size(), (batch_size, num_mem_slots, self.slot_size))

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = instant_slots
            instant_slots = self.norm_slots(instant_slots)

            # Attention.
            q = self.project_q(instant_slots)  # Shape: [batch_size, num_slots, slot_size].
            assert_shape(q.size(), (batch_size, self.num_slots, self.slot_size))

            attn_norm_factor = self.slot_size ** -0.5

            ############### Attention on Input Features ######################################
            attn_logits1 = attn_norm_factor * torch.matmul(k1, q.transpose(2, 1))
            attn1 = F.softmax(attn_logits1, dim=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].
            assert_shape(attn1.size(), (batch_size, num_inputs, self.num_slots))

            # Weighted mean.
            attn1 = attn1 + self.epsilon
            attn1 = attn1 / torch.sum(attn1, dim=1, keepdim=True)
            updates1 = torch.matmul(attn1.transpose(1, 2), v1)
            # `updates` has shape: [batch_size, num_slots, slot_size].
            assert_shape(updates1.size(), (batch_size, self.num_slots, self.slot_size))
            #################################################################################

            ############### Attention on Memory Slots ########################################
            attn_logits2 = attn_norm_factor * torch.matmul(k2, q.transpose(2, 1))
            attn2 = F.softmax(attn_logits2, dim=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].
            assert_shape(attn1.size(), (batch_size, num_inputs, self.num_slots))

            # Weighted mean.
            attn2 = attn2 + self.epsilon
            attn2 = attn2 / torch.sum(attn2, dim=1, keepdim=True)
            updates2 = torch.matmul(attn2.transpose(1, 2), v2)
            # `updates` has shape: [batch_size, num_slots, slot_size].
            assert_shape(updates2.size(), (batch_size, self.num_slots, self.slot_size))
            #################################################################################

            updates = self.mlp_i(torch.cat((updates1, updates2), dim=-1))

            # Slot update.
            # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
            instant_slots = self.gru(
                updates.view(batch_size * self.num_slots, self.slot_size),
                slots_prev.view(batch_size * self.num_slots, self.slot_size),
            )
            instant_slots = instant_slots.view(batch_size, self.num_slots, self.slot_size)
            assert_shape(instant_slots.size(), (batch_size, self.num_slots, self.slot_size))
            instant_slots = instant_slots + self.mlp(self.norm_mlp(instant_slots))
            assert_shape(instant_slots.size(), (batch_size, self.num_slots, self.slot_size))

        ############### Updating on Memory Slots with Attention ##########################

        km = self.project_km(instant_slots)  # Shape: [batch_size, num_slots, slot_size].
        assert_shape(km.size(), (batch_size, self.num_slots, self.slot_size))
        vm = self.project_vm(instant_slots)  # Shape: [batch_size, num_slots, slot_size].
        assert_shape(vm.size(), (batch_size, self.num_slots, self.slot_size))
        qm = self.project_qm(memory_slots)  # Shape: [batch_size, num_mem_slots, slot_size].
        assert_shape(qm.size(), (batch_size, num_mem_slots, self.slot_size))

        attn_logitsm = attn_norm_factor * torch.matmul(qm, km.transpose(2, 1))
        attnm = F.softmax(attn_logitsm, dim=-1)
        # `attn` has shape: [batch_size, num_inputs, num_slots].
        assert_shape(attnm.size(), (batch_size, num_mem_slots, self.num_slots))

        # Weighted mean.
        attnm = attnm + self.epsilon
        attnm = attnm / torch.sum(attnm, dim=1, keepdim=True)
        updatesm = torch.matmul(attnm, vm)
        # `updates` has shape: [batch_size, num_slots, slot_size].
        assert_shape(updatesm.size(), (batch_size, num_mem_slots, self.slot_size))

        memory_slots = memory_slots + self.mlp_m(updatesm)
        #################################################################################

        return memory_slots, instant_slots


class SlotAttentionModel(nn.Module):
    def __init__(
        self,
        resolution: Tuple[int, int],
        num_slots: int,
        num_iterations,
        in_channels: int = 3,
        kernel_size: int = 5,
        slot_size: int = 64,
        hidden_dims: Tuple[int, ...] = (64, 64, 64, 64),
        decoder_resolution: Tuple[int, int] = (4, 4),
        empty_cache=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_mem_slots = 15
        self.num_iterations = num_iterations
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.slot_size = slot_size
        self.empty_cache = empty_cache
        self.hidden_dims = hidden_dims
        self.decoder_resolution = decoder_resolution
        self.out_features = self.hidden_dims[-1]

        modules = []
        channels = self.in_channels
        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        out_channels=h_dim,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=self.kernel_size // 2,
                    ),
                    nn.LeakyReLU(),
                )
            )
            channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, resolution)
        self.encoder_out_layer = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.LeakyReLU(),
            nn.Linear(self.out_features, self.out_features),
        )

        # Build Decoder
        modules = []

        in_size = decoder_resolution[0]
        out_size = in_size

        for i in range(len(self.hidden_dims) - 1, -1, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hidden_dims[i],
                        self.hidden_dims[i - 1],
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            out_size = conv_transpose_out_shape(out_size, 2, 2, 5, 1)

        assert_shape(
            resolution,
            (out_size, out_size),
            message="Output shape of decoder did not match input resolution. Try changing `decoder_resolution`.",
        )

        # same convolutions
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.out_features, self.out_features, kernel_size=5, stride=1, padding=2, output_padding=0,
                ),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(self.out_features, 4, kernel_size=3, stride=1, padding=1, output_padding=0,),
            )
        )

        assert_shape(resolution, (out_size, out_size), message="")

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, self.decoder_resolution)

        self.slot_attention = SlotAttention(
            in_features=self.out_features,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=128,
        )

    def forward(self, batch):
        if self.empty_cache:
            torch.cuda.empty_cache()

        batch_size, nFrames, num_channels, height, width = batch.shape
        outs = []


        ####################################################################################################
        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots = torch.randn((batch_size, self.num_slots, self.slot_size)).cuda()
        slots = self.slot_attention.slots_mu + self.slot_attention.slots_log_sigma.exp() * slots

        memory_slots = torch.randn((batch_size, self.num_mem_slots, self.slot_size)).cuda()
        memory_slots = self.slot_attention.mem_slots_mu + self.slot_attention.mem_slots_log_sigma.exp() * memory_slots

        prev_slot = slots
        ###################################################################################################

        for i in range(nFrames):
        
            x = batch[:,i,:,:,:]
            encoder_out = self.encoder(x)
            encoder_out = self.encoder_pos_embedding(encoder_out)
            # `encoder_out` has shape: [batch_size, filter_size, height, width]
            encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
            # `encoder_out` has shape: [batch_size, filter_size, height*width]
            encoder_out = encoder_out.permute(0, 2, 1)
            encoder_out = self.encoder_out_layer(encoder_out)
            # `encoder_out` has shape: [batch_size, height*width, filter_size]

            memory_slots, slots = self.slot_attention(encoder_out, memory_slots, prev_slot)
            assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))


            ##################
            prev_slot = slots
            #####################



            # `slots` has shape: [batch_size, num_slots, slot_size].
            batch_size, num_slots, slot_size = slots.shape

            slots = slots.view(batch_size * num_slots, slot_size, 1, 1)
            decoder_in = slots.repeat(1, 1, self.decoder_resolution[0], self.decoder_resolution[1])

            out = self.decoder_pos_embedding(decoder_in)
            out = self.decoder(out)
            outs.append(out)
        out = torch.stack(outs, dim = 1)

        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].
        assert_shape(out.size(), (batch_size * num_slots, nFrames, num_channels + 1, height, width))

        out = out.view(batch_size, num_slots, nFrames, num_channels + 1, height, width)
        recons = out[:, :, :, :num_channels, :, :]
        masks = out[:, :, :, -1:, :, :]
        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)
        return recon_combined, recons, masks, slots

    def loss_function(self, input):
        recon_combined, recons, masks, slots = self.forward(input)
        loss = F.mse_loss(recon_combined, input)
        return {
            "loss": loss,
        }


class SoftPositionEmbed(nn.Module):
    def __init__(self, num_channels: int, hidden_size: int, resolution: Tuple[int, int]):
        super().__init__()
        self.dense = nn.Linear(in_features=num_channels + 1, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs: Tensor):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        assert_shape(inputs.shape[1:], emb_proj.shape[1:])
        return inputs + emb_proj
