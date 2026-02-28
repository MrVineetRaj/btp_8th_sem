import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.denoisingModule import EncodingBlock, EncodingBlockEnd, DecodingBlock, DecodingBlockEnd
from models.residualprojectionModule import UCNet
from models.textureReconstructionModule import ConvDown, ConvUp
from models.edgemap import EdgeMap
from models.edgefeatureextractionModule import EAFM
from models.intermediatevariableupdateModule import EGIM
from models.variableguidereconstructionModule import IGRM
from models.degradationEstimation import DegradationEstimator, DegradationAwareConv
import torch.nn.functional as F

def make_model(args, parent=False):
    return EDDUN(args)

class EDDUN(nn.Module):
    def __init__(self, args):
        super(EDDUN, self).__init__()

        # -------Denoising block-------------------------------
        self.channel0 = args.n_colors  # Number of channels
        self.up_factor = args.scale[0]  # Upscale factor
        self.down_factor = args.scale[0]
        self.patch_size = args.patch_size
        self.batch_size = int(args.batch_size / args.n_GPUs)
        
        # Degradation estimation settings
        self.use_degradation = getattr(args, 'use_degradation', False)
        self.kernel_size = getattr(args, 'kernel_size', 21)
        
        # Degradation estimator module
        if self.use_degradation:
            self.deg_estimator = DegradationEstimator(
                in_channels=args.n_colors,
                kernel_size=self.kernel_size,
                num_features=64
            )
            self.deg_aware_conv = DegradationAwareConv(kernel_size=self.kernel_size)
        
        # Denoising block
        self.Encoding_block1 = EncodingBlock(64)
        self.Encoding_block2 = EncodingBlock(64)
        self.Encoding_block3 = EncodingBlock(64)
        self.Encoding_block4 = EncodingBlock(64)

        self.Encoding_block_end = EncodingBlockEnd(64)

        self.Decoding_block1 = DecodingBlock(256)
        self.Decoding_block2 = DecodingBlock(256)
        self.Decoding_block3 = DecodingBlock(256)
        self.Decoding_block4 = DecodingBlock(256)

        self.feature_decoding_end = DecodingBlockEnd(256)
        # ReLU activation layer instance
        self.act = nn.ReLU()
        # 2D convolution instance: 64 input channels, 3 output channels, 3x3 kernel
        self.construction = nn.Conv2d(64, 3, 3, padding=1)

        G0 = 64
        kSize = 3
        T = 4
        self.Fe_e = nn.ModuleList(  # Feature extraction module via convolution
            [nn.Sequential(
                *[
                    nn.Conv2d(3, G0, kSize, padding=(kSize - 1) // 2, stride=1),
                    nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
                ]
            ) for _ in range(T)]
        )

        self.RNNF = nn.ModuleList(
            [nn.Sequential(
                *[
                    nn.Conv2d((i + 2) * G0, G0, 1, padding=0, stride=1),
                    nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1),
                    self.act,
                    nn.Conv2d(64, 3, 3, padding=1)
                ]
            ) for i in range(T)]
        )

        self.Fe_f = nn.ModuleList(
            [nn.Sequential(
                *[
                    nn.Conv2d((2 * i + 3) * G0, G0, 1, padding=0, stride=1)
                ]
            ) for i in range(T - 1)]
        )

        # ----------------------Texture reconstruction module-----------------------------------------------
        self.eta = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.mu = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(T)])
        self.delta = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(T)])
        self.delta_1 = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.delta_2 = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.delta_3 = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.gama = nn.Parameter(torch.tensor(0.01))
        self.conv_up = ConvUp(3, self.up_factor)
        self.conv_down = ConvDown(3, self.up_factor)
        # ----------------------Residual projection block---------------------------------------------------
        # Blur kernel (used as fallback when degradation estimation is disabled)
        self.blur = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1,
                              padding=1, bias=False)
        
        # Noise-aware regularization parameter (learnable)
        self.noise_reg_weight = nn.Parameter(torch.tensor(1.0))
        # Downsampling
        # self.down_sample = nn.MaxPool2d(kernel_size=self.up_factor + 1, stride=1)
        self.UCNet = UCNet(3, 64)
        self.delta_down = nn.Conv2d(3, 3, kernel_size=3, stride=16, padding=1)  # Downsample stride set to 8
        self.linear_layer = nn.Linear(3, 3)
        # -------------------Edge guidance block----------------------------------------------------
        self.input_down = nn.Conv2d(3, 3, kernel_size=3, stride=8, padding=1)
        self.f_down = nn.Conv2d(8, 8, kernel_size=3, stride=8, padding=1)
        self.y_down = nn.Conv2d(3, 3, kernel_size=3, stride=4, padding=1)
        self.edgemap = EdgeMap()
        self.EAFM = EAFM()
        self.EGIM = EGIM()
        self.IGRM = IGRM()
        # self.test_only = args.test_only

    def forward(self, y, return_degradation=False):  # [batch_size ,3 ,7 ,270 ,480] ;
        """Forward pass with optional degradation estimation.
        
        Args:
            y: Low-resolution input image (B, C, H, W)
            return_degradation: If True, also return estimated degradation parameters
            
        Returns:
            x: Super-resolved output image
            (optional) deg_params: Dict with 'blur_kernel' and 'noise_level' if return_degradation=True
        """
        fea_list = []  # Store features before denoising module at each stage
        V_list = []  # Store input feature maps for each RNNF module layer
        outs = []  # Store processed images x at each stage
        x_texture = []  # Store texture module output at each layer
        delta_list = []  # Store compensation at each stage

        f_init = []
        x_init = []
        v_init = []
        
        # Estimate degradation parameters if enabled
        estimated_kernel = None
        estimated_noise = None
        if self.use_degradation:
            estimated_kernel, estimated_noise = self.deg_estimator(y)
            # Add small epsilon to noise level for numerical stability
            estimated_noise = estimated_noise + 1e-6
        
        if not self.training:  # Testing mode
            y = F.interpolate(y, size=(256, 256), mode='bilinear', align_corners=False)

        x_texture.append(torch.nn.functional.interpolate(
            # Interpolate original LR image y to get initial HR image x
            y, scale_factor=self.up_factor, mode='bilinear', align_corners=False))
        target_size = y.shape[2:]
        # x_texture[0] is bicubic upsampled x, f(0) initialized by edgemap, v(0)=x(0)
        # #--------------------Initial module--------------------------------------------------
        x = x_texture[0]  # Upsampled initial SR image
        # print("x shape:", x.shape)  # [batch_size,3,400,400]
        # 1. Network input initialization
        f_init.append(self.edgemap(x))  # Initialize f(0) channel=8
        x_init.append(x)  # Initialize x(0) channel=3
        v_init.append(x)  # Initialize v(0) channel=3
        z_ls = []
        t_ls = []
        r_ls = []
        # #-------------------------------------------------------------------------------------
        # #--------------------------------------Move denoising outside-----------------------------
        fea = self.Fe_e[0](x_init[0])
        encode0, down0 = self.Encoding_block1(fea)
        encode1, down1 = self.Encoding_block2(down0)
        encode2, down2 = self.Encoding_block3(down1)
        encode3, down3 = self.Encoding_block4(down2)
        media_end = self.Encoding_block_end(down3)

        decode3 = self.Decoding_block1(media_end, encode3)
        decode2 = self.Decoding_block2(decode3, encode2)
        decode1 = self.Decoding_block3(decode2, encode1)
        decode0 = self.feature_decoding_end(decode1, encode0)

        decode0 = self.construction(self.act(decode0))
        # #-------------------------------------------------------------------------------------
        for i in range(len(self.Fe_e)):
            # --------------------denoising module---------------------------------------------
            # fea = self.Fe_e[i](x_init[i])
            # fea_list.append(fea)
            # if i != 0:  # If not first feature map, concatenate with previously processed features
            #     fea = self.Fe_f[i - 1](torch.cat(fea_list, 1))
            # # print("fea shape:", fea.shape)  # [batch_size,64,400,400]
            # # 1.Encoding block-----------------------------------------------------------------------
            # encode0, down0 = self.Encoding_block1(fea)
            # # print("down0:", down0.shape)  # [batch_size,64,200,200]
            # # print("encode0:", encode0.shape)  # [batch_size,128,400,400]
            # encode1, down1 = self.Encoding_block2(down0)
            # # print("down1", down1.shape)  # [batch_size,64,100,100]
            # # print("encode1", encode1.shape)  # [batch_size,128,200,200]
            # encode2, down2 = self.Encoding_block3(down1)
            # # print("down2", down2.shape)  # [batch_size,64,50,50]
            # # print("encode2", encode2.shape)  # [batch_size,128,100,100]
            # encode3, down3 = self.Encoding_block4(down2)
            # # print("down3", down3.shape)  # [batch_size,64,25,25]
            # # print("encode3", encode3.shape)  # [batch_size,128,50,50]
            #
            # media_end = self.Encoding_block_end(down3)
            # # print("media_end shape", media_end.shape)  # [batch_size,128,25,25]
            #
            # # Use high-level features from original image in encoder to denoise coarse features,
            # # then fuse extracted features with decoded coarse features in channel dimension for media_end,
            # # to further enhance image features. Then decode and fuse with extracted features
            # # from each dimension to enhance image features [via decoding]
            # # 2.Decoding block---------------------------------------------------------------------
            # decode3 = self.Decoding_block1(media_end, encode3)
            # decode2 = self.Decoding_block2(decode3, encode2)
            # decode1 = self.Decoding_block3(decode2, encode1)
            # decode0 = self.feature_decoding_end(decode1, encode0)
            #
            # fea_list.append(decode0)  # Denoised feature map
            # V_list.append(decode0)
            # if i == 0:  # For initial denoised feature map, add ReLU and conv for further extraction
            #     decode0 = self.construction(self.act(decode0))
            # else:
            #     decode0 = self.RNNF[i - 1](torch.cat(V_list, 1))
            x_init[i] = x_init[i] + decode0
            input_target = x_init[i].shape[2:]
            # print("input_target shape", input_target)   # [400,400]
            y_target = y.shape[2:]  # [200,200]
            # Add extracted denoised features to original image [originally used to update variable v,
            # but v is no longer needed later, so used as temp variable, while x is needed, so save in list]
            # v = x_texture[i] + decode0
            # print("v:"+str(v.max()))
            # # --------------------texture module [to be deleted]--------------------------------------
            # x_texture.append(x_texture[i] - self.delta[i] * (
            #         self.conv_up(self.conv_down(x) - y) + self.eta[i] * (x - v)))
            # # -----------------------edge guided module---------------------------------------------
            # Downsample input
            x_init[i] = self.input_down(x_init[i])
            v_init[i] = self.input_down(v_init[i])
            f_init[i] = self.f_down(f_init[i])
            y = self.y_down(y)
            # 1.EAFM module
            z_ls.append(f_init[i] - self.delta_1[i] * (f_init[i] - self.edgemap(x_init[i])))
            f_init.append(self.EAFM(z_ls[i]))
            # 2.EGIM module
            t_ls.append(v_init[i] - self.delta_2[i] * (v_init[i] - x_init[i]))
            v_init.append(self.EGIM(t_ls[i], f_init[i + 1]))
            # 3.IGRM module with noise-aware data fidelity
            temp_size = self.conv_down(x_init[i]).shape[2:]
            temp_y = F.interpolate(y, size=temp_size, mode='bilinear', align_corners=False)
            
            # Compute data fidelity term
            data_fidelity = self.conv_up(self.conv_down(x_init[i]) - temp_y)
            
            # Apply noise-aware weighting if degradation estimation is enabled
            if self.use_degradation and estimated_noise is not None:
                # Scale data fidelity by inverse of noise level (higher noise = lower weight)
                noise_weight = self.noise_reg_weight / (estimated_noise.view(-1, 1, 1, 1) + 1e-6)
                noise_weight = torch.clamp(noise_weight, 0.1, 10.0)  # Prevent extreme values
                data_fidelity = data_fidelity * noise_weight
            
            r_ls.append(x_init[i] - self.delta_3[i] * (
                    data_fidelity + self.mu[i] * (v_init[i + 1] - x_init[i])))
            x_init.append(self.IGRM(r_ls[i], v_init[i + 1]))
            # Upsample output
            x_init[i + 1] = F.interpolate(x_init[i + 1], size=input_target, mode='bilinear',
                                          align_corners=False)
            v_init[i + 1] = F.interpolate(v_init[i + 1], size=input_target, mode='bilinear',
                                          align_corners=False)
            f_init[i + 1] = F.interpolate(f_init[i + 1], size=input_target, mode='bilinear',
                                          align_corners=False)
            y = F.interpolate(y, size=y_target, mode='bilinear', align_corners=False)
            # #-----------------------RPM module with degradation-aware blur--------------------------
            # Use estimated blur kernel if available, otherwise use fixed blur
            if self.use_degradation and estimated_kernel is not None:
                blurred_x = self.deg_aware_conv(x_init[i + 1], estimated_kernel)
            else:
                blurred_x = self.blur(x_init[i + 1])  # Apply blur kernel to x
            # Downsample blurred image to match y size
            down_out = F.interpolate(blurred_x, size=target_size, mode='bilinear',
                                     align_corners=False)
            difference = y - down_out
            delta_uc = F.interpolate(difference, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
            # print(delta_uc.shape)  # [batch_size,3,400,400]
            delta_down = self.delta_down(delta_uc)  # Adjust downsample stride to 8
            # print("delta_down", delta_down.shape)
            delta = self.UCNet(delta_down)
            # print("delta:", delta.shape)  # [batch_size,3,400,400]
            texture_size = x_init[i + 1].shape[2:]
            delta_up = F.interpolate(delta, size=texture_size, mode='bilinear',
                                     align_corners=False)  # Upsample for adding to original x
            # print("delta_up:", delta_up.shape)
            # print("x_texture:", x_texture[i + 1].shape)
            # delta_up = self.linear_layer(delta_up.view(1, 3, -1))
            delta_list.append(delta_up)
            x = x_init[i + 1] + delta_up
            # #--------------------next stage update--------------------------------------------------
            outs.append(x)
            
            # x[i + 1] = x

        if return_degradation and self.use_degradation:
            deg_params = {
                'blur_kernel': estimated_kernel,
                'noise_level': estimated_noise
            }
            return x, deg_params
        
        return x


