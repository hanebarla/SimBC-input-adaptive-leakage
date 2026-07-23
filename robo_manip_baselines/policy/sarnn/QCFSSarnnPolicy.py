"""QCFS/spiking variant of the multi-camera SARNN policy."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from eipl.layer import SNNSpatialSoftmax

from . import qcfs_layer
from .SarnnPolicy import SarnnPolicy


class QCFSSarnnPolicy(SarnnPolicy):
    """SARNN with QCFS/IF activations in both image encoders."""

    def __init__(
        self,
        state_dim,
        num_images,
        image_size_list,
        num_attentions,
        lstm_hidden_dim,
        qcfs_L=8,
        qcfs_T=0,
        thresh=8.0,
        dec_T=0,
        reset=True,
        lif_alpha=0.0,
        spike_dec_type="default",
    ):
        self._validate_qcfs_args(qcfs_T, dec_T, spike_dec_type)
        if isinstance(image_size_list[0], Iterable):
            image_sizes = list(image_size_list)
        else:
            image_sizes = [image_size_list] * num_images
        if len(image_sizes) != num_images:
            raise ValueError(
                "num_images must match the number of image sizes: "
                f"{num_images} != {len(image_sizes)}."
            )

        super().__init__(
            state_dim,
            num_images,
            image_sizes,
            num_attentions,
            lstm_hidden_dim,
        )
        self.qcfs_T = qcfs_T
        self.spike_dec_type = spike_dec_type
        self.merge_temporal_dim = (
            qcfs_layer.MergeTemporalDim(qcfs_T) if qcfs_T > 0 else None
        )

        encoded_sizes = [(size[0] - 6, size[1] - 6) for size in image_sizes]
        if any(min(size) <= 0 for size in encoded_sizes):
            raise ValueError("Each input image dimension must be larger than 6.")

        def make_if():
            return qcfs_layer.IF(
                T=qcfs_T,
                L=qcfs_L,
                thresh=thresh,
                reset=reset,
                lif_alpha=1.0 if reset else lif_alpha,
            )

        def make_spike_decoder():
            if spike_dec_type == "default":
                return qcfs_layer.ST2FR(qcfs_T, dec_time=dec_T)
            alpha = 1.0 if reset else lif_alpha
            return qcfs_layer.ST2DECAY_FR(qcfs_T, alpha=alpha, dec_time=dec_T)

        if qcfs_T > 0:
            self.image_spike_decoder_list = nn.ModuleList(
                make_spike_decoder() for _ in range(num_images)
            )
            self.attention_spike_decoder_list = nn.ModuleList(
                make_spike_decoder() for _ in range(num_images)
            )
        else:
            self.image_spike_decoder_list = nn.ModuleList()
            self.attention_spike_decoder_list = nn.ModuleList()

        def make_encoder(output_channels, spatial_softmax=None):
            layers = [
                nn.Conv2d(3, 16, 3, 1, 0),
                make_if(),
                nn.Conv2d(16, 32, 3, 1, 0),
                make_if(),
                nn.Conv2d(32, output_channels, 3, 1, 0),
                make_if(),
            ]
            if spatial_softmax is not None:
                layers.append(spatial_softmax)
            return nn.Sequential(*layers)

        self.attention_encoder_list = nn.ModuleList(
            make_encoder(
                self.num_attentions,
                SNNSpatialSoftmax(
                    width=size[0],
                    height=size[1],
                    temperature=1e-4,
                    normalized=True,
                    spike_dec=(
                        self.attention_spike_decoder_list[index] if qcfs_T > 0 else None
                    ),
                ),
            )
            for index, size in enumerate(encoded_sizes)
        )
        self.image_encoder_list = nn.ModuleList(
            make_encoder(self.num_attentions) for _ in range(num_images)
        )
        self.attention_encoder_list.apply(self._initialize_weights)
        self.image_encoder_list.apply(self._initialize_weights)

    @staticmethod
    def _validate_qcfs_args(qcfs_T, dec_T, spike_dec_type):
        if isinstance(qcfs_T, bool) or not isinstance(qcfs_T, int) or qcfs_T < 0:
            raise ValueError(f"qcfs_T must be a non-negative integer, got {qcfs_T!r}.")
        if isinstance(dec_T, bool) or not isinstance(dec_T, int) or dec_T < 0:
            raise ValueError(f"dec_T must be a non-negative integer, got {dec_T!r}.")
        if dec_T > 0 and qcfs_T == 0:
            raise ValueError("dec_T requires qcfs_T greater than zero.")
        if spike_dec_type not in {"default", "decay"}:
            raise ValueError(
                "spike_dec_type must be either 'default' or 'decay', "
                f"got {spike_dec_type!r}."
            )

    def _repeat_over_time(self, image):
        temporal = image.unsqueeze(0).repeat(self.qcfs_T, 1, 1, 1, 1)
        return self.merge_temporal_dim(temporal)

    def forward(self, state, image_list, lstm_state=None):
        if len(image_list) != len(self.image_encoder_list):
            raise ValueError(
                f"Expected {len(self.image_encoder_list)} images, "
                f"got {len(image_list)}."
            )
        encoder_inputs = (
            [self._repeat_over_time(image) for image in image_list]
            if self.qcfs_T > 0
            else image_list
        )

        encoded_images = []
        for index, (image, encoder) in enumerate(
            zip(encoder_inputs, self.image_encoder_list)
        ):
            encoded_image = encoder(image)
            if self.qcfs_T > 0:
                encoded_image = self.image_spike_decoder_list[index](encoded_image)
            encoded_images.append(encoded_image)

        attentions = [
            encoder(image)[0].reshape(-1, self.num_attentions * 2)
            for image, encoder in zip(encoder_inputs, self.attention_encoder_list)
        ]
        lstm_input = torch.cat([state, *attentions], dim=-1)
        lstm_output = self.lstm(lstm_input, lstm_state)
        predicted_state = self.state_decoder(lstm_output[0])
        predicted_attentions = [
            decoder(lstm_output[0]).reshape(-1, self.num_attentions, 2)
            for decoder in self.attention_decoder_list
        ]
        predicted_images = []
        for encoded_image, attention, inv_softmax, decoder in zip(
            encoded_images,
            predicted_attentions,
            self.inv_softmax_list,
            self.image_decoder_list,
        ):
            predicted_images.append(
                decoder(torch.mul(inv_softmax(attention), encoded_image))
            )

        attentions = [
            attention.reshape(-1, self.num_attentions, 2) for attention in attentions
        ]
        return (
            predicted_state,
            predicted_images,
            attentions,
            predicted_attentions,
            lstm_output,
        )

    def update_LIF_param(self, alpha):
        """Update centered membrane leakage and all decay decoders."""

        for module in self.modules():
            if isinstance(module, (qcfs_layer.IF, qcfs_layer.ST2DECAY_FR)):
                module.update_LIF_param(alpha)

    def reset_qcfs_state(self):
        """Clear membrane and firing-rate history at an episode boundary."""

        for module in self.modules():
            if isinstance(module, (qcfs_layer.IF, qcfs_layer.ST2FR)):
                module.reset_state()

    def reset_state(self):
        self.reset_qcfs_state()
