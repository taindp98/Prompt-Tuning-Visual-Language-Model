import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict
import scipy.io as sio


from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from .clip_text import clip
from .clip_text.simple_tokenizer import SimpleTokenizer as _Tokenizer
import tqdm

_tokenizer = _Tokenizer()
import numpy as np
import copy
import clip.clip as clip_ori

from torch_sinkhorn.problem import Epsilon
from torch_sinkhorn.sinkhorn import Sinkhorn, LinearProblem


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


CUSTOM_TEMPLATES_ori = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of an aircraft {}.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of a {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

CUSTOM_TEMPLATES = {
    "OxfordPets": "X X X X {}, a type of pet.",
    "OxfordFlowers": "X X X X {}, a type of flower.",
    "FGVCAircraft": "X X X X {}, a type of aircraft.",
    "DescribableTextures": "X X X X {} texture.",
    "EuroSAT": "X X X X {}.",
    "StanfordCars": "X X X X {}, a type of car",
    "Food101": "X X X X {}, a type of food.",
    "SUN397": "X X X X {}.",
    "Caltech101": "X X X X {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, class_feature, weight, tokenized_prompts, flag=False):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if flag:
            x = self.transformer(x)
        else:
            counter = 0
            outputs = self.transformer.resblocks([x, class_feature, weight, counter])
            x = outputs[0]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )
        return x

class TextEncoder_PLOTPP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding # torch tensor size [77, 512]
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection # torch tensor size [512, 1024]
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        #prompt: torch tensor size [400, 77, 512]
        #tokenized_prompts: torch tensor size [400, 77]
        x = prompts + self.positional_embedding.type(self.dtype)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD, [400, 77, 512]
        x = self.ln_final(x).type(self.dtype)
        
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x # torch tensor size [400, 1024]

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class PromptLearner(nn.Module):

    N: int = 4 ## number of text prompts
    M: int = 4 ## number of vision prompts
    
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            print("use given words to initialize context vectors")
            temp = "a photo of a"
            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)

            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

            ctx_vectors_src = embedding[0, 1 : 1 + n_ctx, :]

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

            ## using 4 prompts
            ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

            ## number of context tokens in vision prompts
            n_ctx_vision=4
            ctx_vision_vectors = torch.empty(self.M, n_ctx_vision ,768, dtype=dtype)
            nn.init.normal_(ctx_vision_vectors, std=0.02)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.ctx_vision = nn.Parameter(ctx_vision_vectors) # parameters of vision prompt to be learned

        clip_model_ = load_clip_to_cpu(cfg)
        clip_model_.cuda()

        temp = CUSTOM_TEMPLATES_ori[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            self.text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

        vis_dim = clip_model.visual.output_dim
        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim // 4, bias=True)),
                    ("relu", QuickGELU()),
                    ("linear2", nn.Linear(vis_dim // 4, 4 * ctx_dim, bias=True)),
                ]
            )
        )
        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()
        classnames = [name.replace("_", " ") for name in classnames]
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        # print(prompts)

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.N,1) #[400,77], 1st prompt for 100 classes, 2nd prompt for 100 classes, ...
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.prev_ctx = None
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def forward(self):
        class_feature = self.meta_net(self.text_features)
        class_feature = class_feature.reshape(class_feature.shape[0], -1, 512)
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx
        
        ctx_vision = self.ctx_vision

        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1,-1)
        ctx = ctx.permute(1, 0, 2, 3) #  N 100 16 512
        ctx = ctx.contiguous().view(self.N*self.n_cls,self.n_ctx,ctx.shape[3])
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts, class_feature, ctx_vision


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


from scipy.optimize import linear_sum_assignment


class CustomCLIP(nn.Module):
    use_tcp: bool = False
    N: int = 4 ## number of prompts

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embedding = self.prompt_learner.text_features
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_PLOTPP(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.domain_sim = -1
        self.domain_sim_src = -1
        self.weight = cfg.TRAINER.COOP.W
        self.tau_a = 1.0
        self.tau_b = 0.5

    def _forward_ot(self, image: torch.Tensor, label):

        b = image.shape[0]
        prompts, class_prompt, vision_prompts = self.prompt_learner()
        # print(f"vision_prompts: {vision_prompts.shape}")
        tokenized_prompts = self.tokenized_prompts
        image_features = self.image_encoder(image.type(self.dtype), vision_prompts) ## [4, 8, 512]
        # print(f"image_features: {image_features.shape}")
        image_feature_pool = image_features.mean(dim=0)
        M = image_features.shape[0]
        self.d = image_features.shape[-1]
        
        # if self.dataset == 'ImageNet':
        #     text_features = self.text_encoder(prompts.to(self.device), tokenized_prompts.to(self.device)) 
        #     text_features = text_features.to(self.device)
        #     text_features =  text_features.contiguous().view(self.N, self.n_cls, self.d)  
        #     text_feature_pool = text_features.mean(dim=0)
        # else:
        #     text_features = self.text_encoder(prompts, tokenized_prompts).contiguous().view(self.N, self.n_cls, self.d)
        #     text_feature_pool = text_features.mean(dim=0)

        # text_features = self.text_encoder(
        #     prompts, class_prompt, self.weight, tokenized_prompts.detach()
        # ).contiguous().view(self.N, self.n_cls, self.d)
        text_features = self.text_encoder(prompts, tokenized_prompts).contiguous().view(self.N, self.n_cls, self.d)
        # print(f"text_features: {text_features.shape}")

        text_feature_pool = text_features.mean(dim=0)
        image_features =  F.normalize(image_features, dim=2)  # N c d 
        image_feature_pool = F.normalize(image_feature_pool, dim=1)
        text_features = F.normalize(text_features, dim=2)
        text_feature_pool = F.normalize(text_feature_pool, dim=1)
        sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous()
        sim = sim.view(M,self.N,b*self.n_cls)
        sim = sim.permute(2,0,1)
        wdist = 1.0 - sim

        with torch.no_grad():
            # KK = torch.exp(-wdist / self.eps)
            # T = self.Sinkhorn(KK,xx,yy)
            epsilon = Epsilon(0.01, init=1.0, decay=0.5)
            ot_prob = LinearProblem(
                wdist, epsilon=epsilon, tau_a=self.tau_a, tau_b=self.tau_b
            )
            sinkhorn = Sinkhorn(lse_mode=True, use_danskin=True)
            sol = sinkhorn(ot_prob)
            T = sol[0].matrix  # [12800, 49,4]
        if torch.isnan(T).any():
            return None

        sim_op = torch.sum(T * sim, dim=(1, 2))
        sim_op = sim_op.contiguous().view(b, self.n_cls)  # [128,100]
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_feature_pool @ text_feature_pool.t()
        logits = logit_scale * sim_op

        if self.prompt_learner.training:
            loss = F.cross_entropy(logits, label)
            return logits, loss
        else:
            return logits

    def forward(self, image, label=None):
        if self.use_tcp:
            return self._forward_tcp(image, label)
        else:
            return self._forward_ot(image, label)

    def _forward_tcp(self, image, label):

        image_features = self.image_encoder(
            image.type(self.dtype)
        )  ## vit: [32, 512] | rn50: [128, 1024]
        text_features_old = self.ori_embedding
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        text_features_old = text_features_old / text_features_old.norm(
            dim=-1, keepdim=True
        )
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, class_prompt = self.prompt_learner()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.text_encoder(
            prompts, class_prompt, self.weight, tokenized_prompts.detach()
        )
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale.detach() * image_features.detach() @ text_features_norm.t()

        if self.prompt_learner.training:
            score = cos(text_features_norm, text_features_old)
            score = 1.0 - torch.mean(score)
            loss = F.cross_entropy(logits, label) + 8.0 * score
            return logits, loss
        else:
            return logits


@TRAINER_REGISTRY.register()
class TCP(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(classnames)
        self.n_cls = len(classnames)
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.w = cfg.TRAINER.COOP.W

        print("Turning off gradients in both the image and the text encoder")

        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
            else:
                print(name)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(
            "prompt_learner", self.model.prompt_learner, self.optim, self.sched
        )

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        self.proto = None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, loss = self.model(image, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    # def model_inference(self, input):
    #    return self.model(input)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        print(names)

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "token_midfix" in state_dict:
                del state_dict["token_midfix"]
            print(
                "Loading weights to {} "
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
