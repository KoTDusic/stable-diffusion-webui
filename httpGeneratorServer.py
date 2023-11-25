import io
import base64
import importlib
from PIL import Image
from modules import shared
from modules import sd_hijack
from modules.processing import process_images_inner
from modules.sd_models import checkpoints_list, checkpoint_alisases
from fastapi import Response, status, FastAPI
from pydantic import BaseModel

from modules.sd_vae import reload_vae_weights
from modules.call_queue import wrap_queued_call
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from modules.sd_models import reload_model_weights
from modules.sd_samplers import all_samplers_map
loras = importlib.import_module("extensions-builtin.Lora.lora").available_loras


class Txt2imgParams(BaseModel):
    text: str | None = "test"
    negative_text: str | None = ""
    model: str | None = "mdjrny-v4"
    hypernetwork: str | None = None
    sampler_name: str | None = "DDIM"
    batch_size: int | None = 1
    count: int | None = 1
    scale: float | None = 7.5
    steps: int | None = 50
    seed: int | None = 42
    hr_fix: bool | None = False
    tiling: bool | None = False
    C: int | None = 4
    f: int | None = 8
    w: int | None = 512
    h: int | None = 512


class Img2imgParams(Txt2imgParams):
    strength: float | None = 0.7
    image_data: str | None = ""


errors_count = 0


def txt2imgPatched(opt: Txt2imgParams):
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        prompt=opt.text,
        negative_prompt=opt.negative_text,
        seed=opt.seed,
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        seed_enable_extras=False,
        sampler_name=opt.sampler_name,
        batch_size=opt.batch_size,
        n_iter=opt.count,
        steps=opt.steps,
        cfg_scale=opt.scale,
        width=opt.w,
        height=opt.h,
        restore_faces=False,
        tiling=False,
        enable_hr=opt.hr_fix,
        denoising_strength=0.7 if opt.hr_fix else None
    )

    processed = process_images_inner(p)

    p.close()

    shared.total_tqdm.clear()
    bytes_list = [from_img_to_base64(image) for image in processed.images]
    final_dict = {"images": bytes_list}
    return final_dict


def img2imgPatched(opt: Img2imgParams):
    buffer: bytes = base64.b64decode(opt.image_data)
    img: Image = Image.open(io.BytesIO(buffer))

    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        prompt=opt.text,
        negative_prompt=opt.negative_text,
        seed=opt.seed,
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        seed_enable_extras=False,
        sampler_name=opt.sampler_name,
        batch_size=opt.batch_size,
        n_iter=opt.count,
        steps=opt.steps,
        cfg_scale=opt.scale,
        width=opt.w,
        height=opt.h,
        restore_faces=False,
        tiling=False,
        init_images=[img],
        mask=None,
        mask_blur=4,
        inpainting_fill=1,
        resize_mode=0,
        denoising_strength=opt.strength,
        inpaint_full_res=False,
        inpaint_full_res_padding=32,
        inpainting_mask_invert=0,
    )

    processed = process_images_inner(p)

    p.close()

    shared.total_tqdm.clear()

    bytes_list = [from_img_to_base64(image) for image in processed.images]
    final_dict = {"images": bytes_list}
    return final_dict


def get_sampler_index(sampler_name):
    return all_samplers_map[sampler_name] if sampler_name in all_samplers_map else all_samplers_map["DDIM"]


def get_model_by_name(model_name: str):
    return checkpoint_alisases[model_name] if model_name in checkpoint_alisases else checkpoint_alisases["mdjrny-v4"]


def get_clip_stop_on_model(model_name: str) -> int:
    return 2 if "naiModel" in model_name else 1


def reload_model(model_name: str):
    new_model = get_model_by_name(model_name)
    if new_model.title == shared.opts.sd_model_checkpoint:
        return
    shared.opts.sd_model_checkpoint = new_model.title
    shared.opts.CLIP_stop_at_last_layers = get_clip_stop_on_model(shared.opts.sd_model_checkpoint)
    print(f"set CLIP_stop = {shared.opts.CLIP_stop_at_last_layers}")
    reload_model_weights()
    reload_vae_weights()


def use_hypernetwork(params: Txt2imgParams):
    if params.hypernetwork != 'none' and params.hypernetwork is not None:
        params.text = f'{params.text} <hypernet:{params.hypernetwork}:1>'


def from_img_to_base64(img):
    im_file = io.BytesIO()
    img.save(im_file, format="png")
    im_bytes = im_file.getvalue()
    return base64.b64encode(im_bytes).decode("UTF-8")


def txt2imgApiLogic(params: Txt2imgParams, response: Response):
    global errors_count
    try:
        use_hypernetwork(params)
        reload_model(params.model)
        result = txt2imgPatched(params)
        errors_count = 0
        return result
    except Exception as e:
        errors_count += 1
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return e.args[0]


def img2imgApiLogic(params: Img2imgParams, response: Response):
    global errors_count
    try:
        use_hypernetwork(params)
        reload_model(params.model)
        result = img2imgPatched(params)
        errors_count = 0
        return result
    except Exception as e:
        errors_count += 1
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return e.args[0]


def available_models() -> list:
    return list(model.model_name for model in checkpoints_list.values())


def available_items() -> dict:
    return {"models": list(model.model_name for model in checkpoints_list.values()),
            "loras": list(loras.keys()),
            "embeddings": list(sd_hijack.model_hijack.embedding_db.word_embeddings.keys()),
            "hypernetworks": list(shared.hypernetworks.keys())}


def errors_check() -> int:
    return errors_count


def alife_check():
    return "alife"


def txt2imgApi(params: Txt2imgParams, response: Response):
    return wrap_queued_call(txt2imgApiLogic)(params, response)


def img2imgApi(params: Img2imgParams, response: Response):
    result = wrap_queued_call(img2imgApiLogic)(params, response)
    return result


def AddCustomRoutes(api_router: FastAPI):
    api_router.add_api_route("/txt2img", txt2imgApi, methods=["POST"])
    api_router.add_api_route("/img2img", img2imgApi, methods=["POST"])
    api_router.add_api_route("/errors_check", errors_check, methods=["GET"])
    api_router.add_api_route("/available_models", available_models, methods=["GET"])
    api_router.add_api_route("/available_items", available_items, methods=["GET"])
    api_router.add_api_route("/alife", alife_check, methods=["GET"])
