_base_ = ['./mask2former_r50_8xb2-160k_ade20k-512x512.py']
# pretrained = r"C:\Users\Amansour\Downloads\upernet_160k_nextvit_base_1n1k6m_pretrained.pth"
pretrained = "https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_distilled_models/fastvit_ma36.pth.tar"
# pretrained ='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
depths = [2, 2, 6, 2]



# model = dict(
#     backbone=dict(
#         _delete_=True,
#         type='nextvit_small_timm',#'SwinTransformer'
#         # embed_dims=96,
#         # depths=depths,
#         # num_heads=[3, 6, 12, 24],
#         # window_size=7,
#         # mlp_ratio=4,
#         # qkv_bias=True,
#         # qk_scale=None,
#         # drop_rate=0.,
#         # attn_drop_rate=0.,
#         # drop_path_rate=0.3,
#         # patch_norm=True,
#         # out_indices=(0, 1, 2, 3),
#         # with_cp=False,
#         # frozen_stages=-1,
#         # resume=pretrained,
#     ),
#         # init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
#     decode_head=dict(in_channels=[96, 256, 512, 1024]))
###############################
model = dict(
    backbone=dict(
        _delete_=True,
        type='fastvit_small',#'SwinTransformer'
        # embed_dims=96,
        # depths=depths,
        # num_heads=[3, 6, 12, 24],
        # window_size=7,
        # mlp_ratio=4,
        # qkv_bias=True,
        # qk_scale=None,
        # drop_rate=0.,
        # attn_drop_rate=0.,
        # drop_path_rate=0.3,
        # patch_norm=True,
        # out_indices=(0, 1, 2, 3),
        # with_cp=False,
        # frozen_stages=-1,
        resume=pretrained,
    ),
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(in_channels=[76, 152, 304, 608]))#[96, 192, 384, 768] #[76, 152, 304, 608]

    # run_beit_pretraining.py --data_path "C:\ghaf\self_supervised"  --output_dir "C:\ghaf\self_supervised\out"  --num_mask_patches 75 --model "beit_base_patch16_224_8k_vocab" --discrete_vae_weight_path "C:\Users\Amansour\Downloads\" --batch_size 128 --lr 1.5e-3 --warmup_steps 10000 --epochs 150 --clip_grad 3.0 --drop_path 0.1 --layer_scale_init_value 0.1
#######################
# model = dict(
#     backbone=dict(
#         _delete_=True,
#         type='SwinTransformer',#'SwinTransformer'
#         embed_dims=96,
#         depths=depths,
#         num_heads=[3, 6, 12, 24],
#         window_size=7,
#         mlp_ratio=4,
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         drop_path_rate=0.3,
#         patch_norm=True,
#         out_indices=(0, 1, 2, 3),
#         with_cp=False,
#         frozen_stages=-1,
#         # resume=pretrained,
#     ),
#         # init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
#     decode_head=dict(in_channels=[96, 192, 384, 768]))#[96, 192, 384, 768] #[76, 152, 304, 608]
#######################    
# norm_cfg = dict(type='SyncBN', requires_grad=True)

# model = dict(
#     backbone=dict(
#         _delete_=True,
#         type='nextvit_base',
#         frozen_stages=-1,
#         norm_eval=False,
#         with_extra_norm=False,
#         norm_cfg=norm_cfg,
#         resume='C:/Users/Amansour/Downloads/upernet_160k_nextvit_base_1n1k6m_pretrained.pth'
#         # resume=pretrained,
#     ),
#         # init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
#     decode_head=dict(in_channels=[96, 256, 512, 1024]))#[96, 192, 384, 768] #[76, 152, 304, 608]
######################## 
# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))
