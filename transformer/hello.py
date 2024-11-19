import numpy as np
import numpy.typing as npt


class MultiHeadAttention:
    """Multi-Head Attention layer implementation.

    Attention(Q, K, V) = softmax(QKᵀ/√dₖ)V

    MultiHead(Q, K, V) = concat(head₁, ..., headₕ)Wₒ
    where headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)

    Args:
        d_model: Model dimension
        h: Number of attention heads
        dropout_rate: Dropout rate
    """

    def __init__(self, d_model: int, h: int, dropout_rate: float = 0.1) -> None:
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.dropout_rate = dropout_rate

        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_q = np.random.normal(0, scale, (d_model, d_model))
        self.W_k = np.random.normal(0, scale, (d_model, d_model))
        self.W_v = np.random.normal(0, scale, (d_model, d_model))
        self.W_o = np.random.normal(0, scale, (d_model, d_model))

    def split_heads(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Split the last dimension into (h, d_k)."""
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.h, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def scaled_dot_product_attention(
        self,
        Q: npt.NDArray[np.float64],
        K: npt.NDArray[np.float64],
        V: npt.NDArray[np.float64],
        mask: npt.NDArray[np.float64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Scaled Dot-Product Attention mechanism.

        Attention(Q, K, V) = softmax(QKᵀ/√dₖ)V

        Args:
            Q: Query matrix of shape (batch_size, h, seq_len_q, d_k)
            K: Key matrix of shape (batch_size, h, seq_len_k, d_k)
            V: Value matrix of shape (batch_size, h, seq_len_v, d_k)
            mask: Optional mask matrix of shape (1, 1, seq_len_q, d_k)

        Returns:
            - output: Attention output of shape (batch_size, h, seq_len_q, d_k)
            - attention_weights: Attention weights of shape (batch_size, h, seq_len_q, seq_len_k)
        """
        d_k = Q.shape[-1]

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

        if mask is not None:
            scores += mask * -1e9

        attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True) + 1e-9

        # Apply dropout
        attention_weights = np.where(
            np.random.uniform(0, 1, attention_weights.shape) > self.dropout_rate,
            attention_weights,
            0,
        )

        output = np.matmul(attention_weights, V)
        return output, attention_weights

    def __call__(
        self,
        Q: npt.NDArray[np.float64],
        K: npt.NDArray[np.float64],
        V: npt.NDArray[np.float64],
        mask: npt.NDArray[np.float64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Apply Multi-Head Attention.

        Args:
            Q: Query matrix of shape (batch_size, seq_len_q, d_model)
            K: Key matrix of shape (batch_size, seq_len_k, d_model)
            V: Value matrix of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask matrix

        Returns:
            output: Multi-Head Attention output
            attention_weights: Attention weights
        """
        batch_size = Q.shape[0]

        Q = np.matmul(Q, self.W_q)
        K = np.matmul(K, self.W_k)
        V = np.matmul(V, self.W_v)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(batch_size, -1, self.d_model)
        output = np.matmul(output, self.W_o)

        return output, attention_weights


class PositionWiseFeedForward:
    """Position-wise Feed-Forward Network.

    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        dropout_rate: Dropout rate
    """

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1) -> None:
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        # Initialize weights
        scale = np.sqrt(2.0 / (d_model + d_ff))
        self.W_1 = np.random.normal(0, scale, (d_model, d_ff))
        self.b_1 = np.zeros(d_ff)
        self.W_2 = np.random.normal(0, scale, (d_ff, d_model))
        self.b_2 = np.zeros(d_model)

    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply Position-wise Feed-Forward Network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        output = np.matmul(x, self.W_1) + self.b_1
        output = np.maximum(0, output)

        output = np.where(
            np.random.uniform(0, 1, output.shape) > self.dropout_rate, output, 0
        )

        output = np.matmul(output, self.W_2) + self.b_2
        return output


class LayerNormalization:
    """Layer Normalization.

    LN(x) = γ * (x - μ) / (σ + ε) + β
    where μ = mean(x), σ = std(x)

    Args:
        d_model: Model dimension
        eps: Small constant to numerical stability
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply Layer Normalization.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Normalized tensor of shape (batch_size, seq_len, d_model)
        """
        mu = np.mean(x, axis=-1, keepdims=True)
        sigma = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mu) / (sigma + self.eps) + self.beta


class EncoderLayer:
    """Transformer Encoder layer.

    Args:
        d_model: Model dimension
        h: Number of attention heads
        d_ff: Feed-forward dimension
        dropout_rate: Dropout rate
    """

    def __init__(
        self, d_model: int, h: int, d_ff: int, dropout_rate: float = 0.1
    ) -> None:
        self.mha = MultiHeadAttention(d_model, h, dropout_rate)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout_rate)
        self.ln1 = LayerNormalization(d_model)
        self.ln2 = LayerNormalization(d_model)
        self.dropout_rate = dropout_rate

    def __call__(
        self, x: npt.NDArray[np.float64], mask: npt.NDArray[np.float64] | None = None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Process input through the Encoder layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor

        Returns:
            output: Processed tensor
            attention_weights: Attention weights
        """
        attn_output, attention_weights = self.mha(x, x, x, mask)
        attn_output = np.where(
            np.random.uniform(0, 1, attn_output.shape) > self.dropout_rate,
            attn_output,
            0,
        )
        out1 = self.ln1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = np.where(
            np.random.uniform(0, 1, ffn_output.shape) > self.dropout_rate, ffn_output, 0
        )
        out2 = self.ln2(out1 + ffn_output)

        return out2, attention_weights


class DecoderLayer:
    """Transformer Decoder layer.

    Args:
        d_model: Model dimension
        h: Number of attention heads
        d_ff: Feed-forward dimension
        dropout_rate: Dropout rate
    """

    def __init__(
        self, d_model: int, h: int, d_ff: int, dropout_rate: float = 0.1
    ) -> None:
        self.mha1 = MultiHeadAttention(d_model, h, dropout_rate)
        self.mha2 = MultiHeadAttention(d_model, h, dropout_rate)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout_rate)
        self.ln1 = LayerNormalization(d_model)
        self.ln2 = LayerNormalization(d_model)
        self.ln3 = LayerNormalization(d_model)
        self.dropout_rate = dropout_rate

    def __call__(
        self,
        x: npt.NDArray[np.float64],
        enc_output: npt.NDArray[np.float64],
        look_ahead_mask: npt.NDArray[np.float64] | None = None,
        padding_mask: npt.NDArray[np.float64] | None = None,
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """Process input through the Decoder layer.

        Args:
            x: Input tensor of shape (batch_size, target_seq_len, d_model)
            enc_output: Encoder output tensor of shape (batch_size, input_seq_len, d_model)
            look_ahead_mask: Mask for self-attention of shape (1, 1, target_seq_len, target_seq_len)
            padding_mask: Mask for encoder-decoder attention of shape (1, 1, 1, input_seq_len)

        Returns:
            - output: Processed tensor of shape (batch_size, target_seq_len, d_model)
            - attn_weights1: Self-attention weights of shape (batch_size, num_heads, target_seq_len, target_seq_len)
            - attn_weights2: Encoder-decoder attention weights of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """
        attn1_output, attn1_weights = self.mha1(x, x, x, look_ahead_mask)
        attn1_output = np.where(
            np.random.uniform(0, 1, attn1_output.shape) > self.dropout_rate,
            attn1_output,
            0,
        )
        out1 = self.ln1(x + attn1_output)

        attn2_output, attn2_weights = self.mha2(
            out1, enc_output, enc_output, padding_mask
        )
        attn2_output = np.where(
            np.random.uniform(0, 1, attn2_output.shape) > self.dropout_rate,
            attn2_output,
            0,
        )
        out2 = self.ln2(out1 + attn2_output)

        ffn_output = self.ffn(out2)
        ffn_output = np.where(
            np.random.uniform(0, 1, ffn_output.shape) > self.dropout_rate, ffn_output, 0
        )
        out3 = self.ln3(out2 + ffn_output)

        return out3, attn1_weights, attn2_weights


class PositionalEncoding:
    """Positional Encoding for Transformer.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000) -> None:
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        pe = np.zeroes((max_seq_len, d_model))
        position = np.arange(0, max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe[np.newaxis, :, :]

    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, : x.shape[1], :]


class Transformer:
    """Complete Transformer model.

    Args:
        d_model: Model dimension
        h: Number of attention heads
        d_ff: Feed-forward dimension
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        max_seq_len: Maximum sequence length
        dropout_rate: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        h: int,
        d_ff: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        max_seq_len: int = 5000,
        dropout_rate: float = 0.1,
    ) -> None:
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        self.encoder_layers = [
            EncoderLayer(d_model, h, d_ff, dropout_rate)
            for _ in range(num_encoder_layers)
        ]
        self.decoder_layers = [
            DecoderLayer(d_model, h, d_ff, dropout_rate)
            for _ in range(num_decoder_layers)
        ]

        self.dropout_rate = dropout_rate

    def encode(
        self,
        x: npt.NDArray[np.float64],
        mask: npt.NDArray[np.float64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]:
        """Encode input sequence.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional padding mask of shape (batch_size, 1, 1, input_seq_len)

        Returns:
            - output: Encoded tensor of shape (batch_size, seq_len, d_model)
            - attention_weights: Attention weights
        """
        # Add Positional Encoding
        x = self.pos_encoding(x)

        x = np.where(np.random.uniform(0, 1, x.shape) > self.dropout_rate, x, 0)

        attention_weights = {}

        for i in range(self.num_encoder_layers):
            x, attn_weights = self.encoder_layers[i](x, mask)
            attention_weights[f"encoder_layer{i+1}"] = attn_weights

        return x, attention_weights

    def decode(
        self,
        x: npt.NDArray[np.float64],
        enc_output: npt.NDArray[np.float64],
        look_ahead_mask: npt.NDArray[np.float64] | None = None,
        padding_mask: npt.NDArray[np.float64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]:
        """Decode input sequence.

        Args:
            x: Input tensor of shape (batch_size, target_seq_len, d_model)
            enc_output: Encoder output tensor of shape (batch_size, input_seq_len, d_model)
            look_ahead_mask: Mask for self-attention of shape (1, 1, target_seq_len, target_seq_len)
            padding_mask: Mask for encoder-decoder attention of shape (batch_size, 1, 1, input_seq_len)

        Returns:
            - output: Decoded tensor of shape (batch_size, target_seq_len, d_model)
            - attention_weights: Attention weights
        """
        # Add Positional Encoding
        x = self.pos_encoding(x)

        x = np.where(np.arandom.uniform(0, 1, x.shape) > self.dropout_rate, x, 0)

        attention_weights = {}

        for i in range(self.num_decoder_layers):
            x, attn1_weights, attn2_weights = self.decoder_layers[i](
                x, enc_output, look_ahead_mask, padding_mask
            )
            attention_weights[f"decoder_layer{i+1}_self_attn"] = attn1_weights
            attention_weights[f"decoder_layer{i+1}_enc_dec_attn"] = attn2_weights

        return x, attention_weights

    def __call__(
        self,
        enc_input: npt.NDArray[np.float64],
        dec_input: npt.NDArray[np.float64],
        enc_padding_mask: npt.NDArray[np.float64] | None = None,
        look_ahead_mask: npt.NDArray[np.float64] | None = None,
        dec_padding_mask: npt.NDArray[np.float64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]:
        """Process input through the entire Transformer.

        Args:
            enc_input: Encoder input tensor of shape (batch_size, input_seq_len, d_model)
            dec_input: Decoder input tensor of shape (batch_size, target_seq_len, d_model)
            enc_padding_mask: Encoder padding mask of shape (batch_size, 1, 1, input_seq_len)
            look_ahead_mask: Look-ahead mask of shape (1, 1, target_seq_len, target_seq_len)
            dec_padding_mask: Decoder padding mask of shape (batch_size, 1, 1, target_seq_len)

        Returns:
            - output: Final output of shape (batch_size, target_seq_len, d_model)
            - attention_weights: Dict containing attention weights
        """
        enc_output, enc_attn_weights = self.encode(enc_input, enc_padding_mask)

        dec_output, dec_attn_weights = self.decode(
            dec_input, enc_output, look_ahead_mask, dec_padding_mask
        )

        attention_weights = {**enc_attn_weights, **dec_attn_weights}

        return dec_output, attention_weights
