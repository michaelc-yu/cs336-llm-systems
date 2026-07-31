from pathlib import Path

import modal

MODAL_SHARED_PATH = Path("/shared-data")


def get_shared_assets_path() -> Path:
    if modal.is_local():
        (local_path := Path("local-shared-data").resolve()).mkdir(exist_ok=True, parents=True)
        return local_path
    else:
        return MODAL_SHARED_PATH
