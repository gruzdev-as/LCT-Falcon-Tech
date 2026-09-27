from common.src.configs.constants import GALLERY_PREFIX
from init.src.gallery import Gallery
from init.src.steps.images import object_key, upload_gallery


async def test_uploads_every_gallery_image(gallery: Gallery, objects: dict[str, bytes]) -> None:
    uploaded = await upload_gallery(gallery, concurrency=4)

    assert uploaded == gallery.count
    assert len(objects) == gallery.count
    assert all(key.startswith(f"{GALLERY_PREFIX}/") for key in objects)


async def test_a_second_upload_writes_nothing(gallery: Gallery, objects: dict[str, bytes]) -> None:
    await upload_gallery(gallery, concurrency=4)

    assert await upload_gallery(gallery, concurrency=4) == 0
    assert len(objects) == gallery.count


async def test_upload_fills_only_the_gaps(gallery: Gallery, objects: dict[str, bytes]) -> None:
    await upload_gallery(gallery, concurrency=4)
    del objects[object_key(gallery.entries[0])]

    assert await upload_gallery(gallery, concurrency=4) == 1


async def test_force_reuploads_everything(gallery: Gallery, objects: dict[str, bytes]) -> None:
    await upload_gallery(gallery, concurrency=4)

    assert await upload_gallery(gallery, concurrency=4, force=True) == gallery.count
