from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


TEXT = (
    "agents plan actions observe results and update context. "
    "language models predict tokens from previous tokens. "
    "attention lets each token read useful earlier tokens. "
) * 40


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, block_size: int) -> None:
        super().__init__()
        # 多头 attention 会把 d_model 平均分给每个 head。
        # d_model 是每个 token 的向量维度；block_size 才是最大上下文长度。
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # qkv 一次性算出 query/key/value，减少三次线性层的样板代码。
        # 输入和输出 shape 都围绕 (batch_size, seq_len, d_model) 展开。
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        # register_buffer 注册的是“不是参数、但要跟着模型移动/保存”的 tensor。
        # mask 不需要训练，所以不用 nn.Parameter。
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # 多头 attention 标准 shape: (batch_size, num_heads, seq_len, head_dim)。
        # 多头不是多跑几个模型，而是把 d_model 切成几份，让不同 head 学不同关系。
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # q @ k^T 得到 token 之间的相似度分数。
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        # causal mask 保证当前位置不能看未来 token。
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v

        # 把 heads 维度拼回 d_model，恢复 (batch_size, seq_len, d_model)。
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, block_size: int) -> None:
        super().__init__()
        # 一个 GPT block 通常由两块组成：
        # attention 负责 token 之间交流，MLP 负责每个 token 自己的非线性加工。
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            # 常见设计会先把 hidden 扩到 4 倍，再压回原维度。
            # 这不是改变序列长度，而是改变每个 token 向量内部的维度。
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Transformer block：先 LayerNorm，再 attention/MLP。
        # 残差连接 x + f(x) 保持 shape 不变，也让梯度更容易流动。
        # 第一行：每个 token 先从上下文里读信息。
        x = x + self.attn(self.ln1(x))

        # 第二行：每个 token 再独立过一段 MLP 做加工。
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int) -> None:
        super().__init__()
        d_model = 64
        self.block_size = block_size

        # token_embedding 负责“这个 token 是什么”。
        # 输入 idx 是整数 id；embedding 输出连续向量。神经网络只能处理数值张量。
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # position_embedding 负责“这个 token 在第几个位置”。
        # 如果没有位置信息，attention 本身不天然知道顺序。
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(
            Block(d_model, num_heads=4, block_size=block_size),
            Block(d_model, num_heads=4, block_size=block_size),
        )
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx: (batch_size, seq_len)，里面是 token ids。
        batch_size, seq_len = idx.shape

        # positions 是 0..seq_len-1，对应序列里的每个位置。
        # shape 是 (seq_len,)，加到 token_embedding 时会自动 broadcast 到 batch 维度。
        positions = torch.arange(seq_len, device=idx.device)

        # token 信息和位置信息相加，得到每个位置的初始表示。
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(x)
        x = self.ln(x)

        # logits: (batch_size, seq_len, vocab_size)，每个位置预测下一个 token。
        # 对某个位置来说，vocab_size 个数字就是“下一个 token 是词表中每个 id 的分数”。
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # cross_entropy 要求 (N, classes) 和 (N,)，所以合并 batch_size/seq_len。
            loss = F.cross_entropy(
                logits.view(batch_size * seq_len, -1),
                targets.view(batch_size * seq_len),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, steps: int) -> torch.Tensor:
        # generate 是推理阶段：不再给 targets，也不算 loss，只反复预测下一个 token。
        for _ in range(steps):
            # 如果序列超过 block_size，只保留最后 block_size 个 token 作为上下文。
            context = idx[:, -self.block_size :]
            logits, _ = self(context)

            # 只取最后一个位置的 logits，因为生成时只需要预测“下一个 token”。
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def make_data() -> tuple[torch.Tensor, dict[str, int], dict[int, str]]:
    # 字符级 tokenizer：简单但足够演示语言模型训练。
    # 后面如果换成 BPE tokenizer，这里 data 的构造方式会变，但模型仍然吃 token ids。
    chars = sorted(set(TEXT))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in TEXT], dtype=torch.long)
    return data, stoi, itos


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 每个样本是一段连续 token ids。x 是输入上下文，y 是每个位置的下一个 token。
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])

    # y 是 x 的下一个字符序列，用来做 next-token prediction。
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


def decode(ids: torch.Tensor, itos: dict[int, str]) -> str:
    return "".join(itos[i] for i in ids.tolist())


def main() -> None:
    torch.manual_seed(123)

    # block_size 是模型一次最多能看的上下文长度。
    block_size = 32
    data, stoi, itos = make_data()
    model = TinyGPT(vocab_size=len(stoi), block_size=block_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    x, y = get_batch(data, batch_size=4, block_size=block_size)
    logits, loss = model(x, y)
    assert loss is not None

    print("shape check")
    print(f"x: {tuple(x.shape)}")
    print(f"logits: {tuple(logits.shape)}")
    print(f"loss: {loss.item():.4f}")
    print()

    for step in range(1, 301):
        x, y = get_batch(data, batch_size=32, block_size=block_size)
        _, loss = model(x, y)
        assert loss is not None

        # 标准训练三步：清梯度 -> 反向传播 -> 更新参数。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 50 == 0:
            print(f"step {step:04d} | loss={loss.item():.4f}")

    start = torch.tensor([[stoi["a"]]], dtype=torch.long)
    sample = model.generate(start, steps=180)[0]
    print()
    print(decode(sample, itos))


if __name__ == "__main__":
    main()
