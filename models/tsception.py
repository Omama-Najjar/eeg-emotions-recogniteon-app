import torch
import torch.nn as nn
import torch.nn.functional as F

class TSception(nn.Module):
    def __init__(self, num_classes=2, input_size=(1, 8, 256), fs=128, 
                 num_T=15, num_S=15, hidden_dim=128, dropout_rate=0.5):
        """
        TSception adapted for an 8-channel Unicorn input.
        
        Args:
            num_classes (int): Number of target classes (default 2 for binary)
            input_size (tuple): (channels_in, num_EEG_channels, sequence_length)
                For our dataset it's (1, 8, 256) where 256 = 2s at 128Hz
            fs (int): Sampling frequency (Hz)
            num_T (int): Number of temporal filters per kernel length
            num_S (int): Number of spatial filters
            hidden_dim (int): Hidden dimension for the fully connected layer
            dropout_rate (float): Dropout probability
        """
        super(TSception, self).__init__()
        
        # Unpack input size
        in_channels, eeg_channels, seq_len = input_size
        assert eeg_channels == 8, "Expected 8 EEG channels for Unicorn compatibility!"
        
        # Temporal block: 3 multi-scale temporal convolutions
        # To capture different frequency dynamics as suggested (e.g. 0.5s, 0.25s, 0.125s)
        self.kernel_length1 = int(fs * 0.5)
        self.kernel_length2 = int(fs * 0.25)
        self.kernel_length3 = int(fs * 0.125)
        
        # Ensure kernel shapes are odd for padding 'same' convenience if needed, 
        # though TSception normally pads so that output seq_len is consistent.
        # We will use padding=0 and handle the out lengths.
        
        self.Tception1 = nn.Sequential(
            nn.Conv2d(in_channels, num_T, kernel_size=(1, self.kernel_length1), stride=1, padding=(0, self.kernel_length1 // 2)),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        )
        self.Tception2 = nn.Sequential(
            nn.Conv2d(in_channels, num_T, kernel_size=(1, self.kernel_length2), stride=1, padding=(0, self.kernel_length2 // 2)),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        )
        self.Tception3 = nn.Sequential(
            nn.Conv2d(in_channels, num_T, kernel_size=(1, self.kernel_length3), stride=1, padding=(0, self.kernel_length3 // 2)),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        )
        
        # We need to compute the output sizes after AvgPool2d
        # The padded convs keep seq_len mostly the same. 
        # Actually, let's strictly use formula for output dimension:
        # pool yields approx (seq_len // 8)
        
        # Spatial Block
        # After concatenating the 3 temporal branches along the channel dimension (dim=1)
        # Total channels = num_T * 3
        
        self.Sception1 = nn.Sequential(
            # Spatial convolution acts across the EEG channels
            # Input dim after temp: (batch, num_T*3, 8, seq_len')
            nn.Conv2d(num_T * 3, num_S, kernel_size=(eeg_channels, 1), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        )
        
        self.Sception2 = nn.Sequential(
            nn.Conv2d(num_T * 3, num_S, kernel_size=(int(eeg_channels // 2), 1), stride=(int(eeg_channels // 2), 1), padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        )
        
        # Fusion block
        self.BN_t = nn.BatchNorm2d(num_T * 3)
        self.BN_s = nn.BatchNorm2d(num_S * 2) 

        # We need to dynamically calculate the flattened size before the FC layer
        # Let's create a dummy input to pass through the network up to the flatten point
        dummy_in = torch.randn(1, *input_size)
        flattened_size = self._get_flatten_size(dummy_in)
        
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def _get_flatten_size(self, x):
        """ Runs a dummy pass to infer the size of the tensor entering the FC layers. """
        y1 = self.Tception1(x)
        y2 = self.Tception2(x)
        y3 = self.Tception3(x)
        
        # Trim time dimensions if they don't match due to padding logic.
        min_len = min(y1.size(-1), y2.size(-1), y3.size(-1))
        y1 = y1[:, :, :, :min_len]
        y2 = y2[:, :, :, :min_len]
        y3 = y3[:, :, :, :min_len]
        
        out = torch.cat((y1, y2, y3), dim=1)
        out = self.BN_t(out)
        
        z1 = self.Sception1(out)
        z2 = self.Sception2(out)
        
        # z2 will have 2 channels left if eeg_channels=8 and kernel=4, stride=4
        # We need them to be the same size so we can concat them. 
        # Easiest way in 8-channel mode: Since Sception1 reduces eeg_channels to 1 (kernel size 8),
        # and Sception2 reduces eeg_channels to 2, let's flatten the channel dimensions
        
        z1 = torch.flatten(z1, 1)
        z2 = torch.flatten(z2, 1)
        
        out = torch.cat((z1, z2), dim=1)
        return out.shape[1]

    def forward(self, x):
        y1 = self.Tception1(x)
        y2 = self.Tception2(x)
        y3 = self.Tception3(x)
        
        min_len = min(y1.size(-1), y2.size(-1), y3.size(-1))
        y1 = y1[:, :, :, :min_len]
        y2 = y2[:, :, :, :min_len]
        y3 = y3[:, :, :, :min_len]
        
        out = torch.cat((y1, y2, y3), dim=1)
        out = self.BN_t(out)
        
        z1 = self.Sception1(out)
        z2 = self.Sception2(out)
        
        z1 = torch.flatten(z1, 1)
        z2 = torch.flatten(z2, 1)
        
        out = torch.cat((z1, z2), dim=1)
        out = self.classifier(out)
        
        return out

class OptimizedTSception(nn.Module):
    def __init__(self, num_classes=2, input_size=(1, 8, 256), fs=128, 
                 num_T=15, num_S=15, hidden_dim=128, dropout_rate=0.5, pool_size=4,
                 inception_windows=[0.5, 0.25, 0.125]):
        """
        Optimized TSception with Low-Density Spatial Redesign (Hypothesis B)
        and parameterizable Hyperparameters (Hypothesis C).
        """
        super(OptimizedTSception, self).__init__()
        
        in_channels, eeg_channels, seq_len = input_size
        assert eeg_channels == 8, "Expected 8 EEG channels for Unicorn compatibility!"
        
        # Temporal Block (Dynamically generated based on inception_windows)
        self.t_blocks = nn.ModuleList()
        for win_ratio in inception_windows:
            k_len = int(fs * win_ratio)
            # Ensure odd kernel for symmetric padding if desired, but we will just use padding=(0, k_len//2)
            self.t_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, num_T, kernel_size=(1, k_len), stride=1, padding=(0, k_len // 2)),
                nn.LeakyReLU(),
                nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_size))
            ))
            
        num_temp_branches = len(inception_windows)
        self.BN_t = nn.BatchNorm2d(num_T * num_temp_branches)
        
        # Spatial Block (Hypothesis B: Low-Density Spatial Redesign)
        # Pathway 1: Global Spatial Kernel
        self.Sception_Global = nn.Sequential(
            nn.Conv2d(num_T * num_temp_branches, num_S, kernel_size=(eeg_channels, 1), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 16)) # Stabilize spatial length mapping
        )
        
        # Pathway 2: Targeted Lateral Kernel (C3, C4, PO7, PO8) -> Indices [1, 3, 5, 7]
        self.lateral_indices = [1, 3, 5, 7]
        self.Sception_Lateral = nn.Sequential(
            nn.Conv2d(num_T * num_temp_branches, num_S, kernel_size=(4, 1), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 16)) # Stabilize spatial length mapping
        )
        
        self.BN_s = nn.BatchNorm2d(num_S * 2) 

        # Classifier
        flattened_size = num_S * 2 * 16
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Temporal forward
        y_list = [block(x) for block in self.t_blocks]
        
        # Trim time dimensions if mismatch
        min_len = min([y.size(-1) for y in y_list])
        y_list = [y[:, :, :, :min_len] for y in y_list]
        
        out = torch.cat(y_list, dim=1)
        out = self.BN_t(out)
        
        # Spatial forward
        z_global = self.Sception_Global(out)
        
        # Safe slicing for Lateral Pathway
        out_lateral = out[:, :, self.lateral_indices, :].clone()
        z_lateral = self.Sception_Lateral(out_lateral)
        
        # Concat along channel dim
        z_out = torch.cat((z_global, z_lateral), dim=1)
        z_out = self.BN_s(z_out)
        
        z_out = torch.flatten(z_out, 1)
        out = self.classifier(z_out)
        
        return out



class DualStreamTSception(nn.Module):
    def __init__(self, num_classes=2, input_size=(1, 8, 256), fs=128, 
                 num_T_ea=15, num_S_ea=15, num_T_raw=15, num_S_raw=15, 
                 hidden_dim=128, dropout_rate=0.5, pool_size=4,
                 inception_windows=[0.5, 0.25, 0.125], num_streams=3):
        super(DualStreamTSception, self).__init__()
        
        self.num_streams = num_streams
        in_channels, eeg_channels, seq_len = input_size
        assert eeg_channels == 8, "Expected 8 EEG channels"
        
        # Learnable Attention Weights
        self.stream_weights = nn.Parameter(torch.ones(self.num_streams))
        
        # Temporal Blocks
        self.t_blocks_ea = nn.ModuleList()
        self.t_blocks_raw = nn.ModuleList()
        for win_ratio in inception_windows:
            k_len = int(fs * win_ratio)
            self.t_blocks_ea.append(nn.Sequential(
                nn.Conv2d(in_channels, num_T_ea, kernel_size=(1, k_len), stride=1, padding=(0, k_len // 2)),
                nn.LeakyReLU(),
                nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_size))
            ))
            self.t_blocks_raw.append(nn.Sequential(
                nn.Conv2d(in_channels, num_T_raw, kernel_size=(1, k_len), stride=1, padding=(0, k_len // 2)),
                nn.LeakyReLU(),
                nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_size))
            ))
            
        num_temp_branches = len(inception_windows)
        self.BN_t_ea = nn.BatchNorm2d(num_T_ea * num_temp_branches)
        self.BN_t_raw = nn.BatchNorm2d(num_T_raw * num_temp_branches)
        
        # Spatial Blocks
        self.Sception_EA_Global = nn.Sequential(
            nn.Conv2d(num_T_ea * num_temp_branches, num_S_ea, kernel_size=(eeg_channels, 1), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 16))
        )
        
        if self.num_streams == 3:
            self.Sception_Raw_Global = nn.Sequential(
                nn.Conv2d(num_T_raw * num_temp_branches, num_S_raw, kernel_size=(eeg_channels, 1), stride=1, padding=0),
                nn.LeakyReLU(),
                nn.AdaptiveAvgPool2d((1, 16))
            )
            
        self.lateral_indices = [1, 3, 5, 7]
        self.Sception_Raw_Lateral = nn.Sequential(
            nn.Conv2d(num_T_raw * num_temp_branches, num_S_raw, kernel_size=(4, 1), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 16))
        )
        
        total_S = num_S_ea + (num_S_raw * 2 if self.num_streams == 3 else num_S_raw)
        self.BN_s = nn.BatchNorm2d(total_S) 

        flattened_size = total_S * 16
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x_raw, x_ea):
        weights = torch.softmax(self.stream_weights, dim=0)
        
        y_ea_list = [block(x_ea) for block in self.t_blocks_ea]
        min_len_ea = min([y.size(-1) for y in y_ea_list])
        y_ea_list = [y[:, :, :, :min_len_ea] for y in y_ea_list]
        out_t_ea = torch.cat(y_ea_list, dim=1)
        out_t_ea = self.BN_t_ea(out_t_ea)
        
        y_raw_list = [block(x_raw) for block in self.t_blocks_raw]
        min_len_raw = min([y.size(-1) for y in y_raw_list])
        y_raw_list = [y[:, :, :, :min_len_raw] for y in y_raw_list]
        out_t_raw = torch.cat(y_raw_list, dim=1)
        out_t_raw = self.BN_t_raw(out_t_raw)
        
        z_ea_global = self.Sception_EA_Global(out_t_ea)
        z_ea_global = z_ea_global * weights[0]
        
        out_lateral = out_t_raw[:, :, self.lateral_indices, :].clone()
        z_raw_lateral = self.Sception_Raw_Lateral(out_lateral)
        
        if self.num_streams == 3:
            z_raw_global = self.Sception_Raw_Global(out_t_raw)
            z_raw_global = z_raw_global * weights[1]
            z_raw_lateral = z_raw_lateral * weights[2]
            z_out = torch.cat((z_ea_global, z_raw_global, z_raw_lateral), dim=1)
        else:
            z_raw_lateral = z_raw_lateral * weights[1]
            z_out = torch.cat((z_ea_global, z_raw_lateral), dim=1)
            
        z_out = self.BN_s(z_out)
        z_out = torch.flatten(z_out, 1)
        out = self.classifier(z_out)
        
        return out

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

class EEGNet(nn.Module):
    def __init__(self, num_classes=2, eeg_channels=8, samples=256, dropout_rate=0.5, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2
        
        # Block 1: Temporal Conv
        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        
        # Block 2: Depthwise Spatial Conv
        self.block2 = nn.Sequential(
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (eeg_channels, 1), groups=self.F1, bias=False, max_norm=1),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # Block 3: Separable Conv
        self.block3 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16), padding=(0, 8), groups=self.F1 * self.D, bias=False),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # Classifier
        # Calculate flattened size
        dummy = torch.randn(1, 1, eeg_channels, samples)
        x = self.block1(dummy)
        x = self.block2(x)
        x = self.block3(x)
        flattened_size = x.view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, num_classes)
        )

    def forward(self, x_raw, x_ea=None):
        # EEGNet uses only X_raw, x_ea is dummy if passed
        x = self.block1(x_raw)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        # Max norm constraint on linear weights
        self.classifier[0].weight.data = torch.renorm(self.classifier[0].weight.data, p=2, dim=0, maxnorm=0.25)
        out = self.classifier(x)
        return out


if __name__ == "__main__":
    model = TSception(num_classes=2, input_size=(1, 8, 256))
    dummy_input = torch.randn(4, 1, 8, 256)
    out = model(dummy_input)
    print(f"TSception Output shape for batch size 4: {out.shape} (Expected: 4, 2)")
    
    opt_model = OptimizedTSception(num_classes=2, input_size=(1, 8, 256))
    out_opt = opt_model(dummy_input)
    print(f"OptimizedTSception Output shape for batch size 4: {out_opt.shape} (Expected: 4, 2)")

    dummy_input = torch.randn(4, 1, 8, 256)
    out = model(dummy_input)
    print(f"Output shape for batch size 4: {out.shape} (Expected: 4, 2)")

# Added Optimized TSception
