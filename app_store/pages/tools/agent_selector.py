import dataclasses
import glob
import os
from pathlib import Path

import streamlit as st
import torch

from src.tools.model_util import ModelIO


MODEL_FOLDER = Path(os.getcwd()) / "model_objects"
DEFAULT_PATH = MODEL_FOLDER / "A/seed_0/_steps-30000.model"


@dataclasses.dataclass
class LoadedModel:
    model: torch.nn.Module
    path: str
    path_from_root: str
    name: str
    is_selected: bool = False


@st.cache_data
def load_model(model_name, device):
    model_dir = os.path.dirname(model_name)
    model_handler = ModelIO(directory=model_dir, tag="AppStore Analysis")
    model, start_num_steps = model_handler.load(device=device, path=model_name)
    if not hasattr(model, "set_device"):
        model.device = device
        model.pin = device == torch.device("cuda")
    else:
        model.set_device(device)
    return model


def is_already_loaded(model_name: str):
    loaded_models = st.session_state.loaded_models
    all_names = [model.name for model in loaded_models.values()]
    return True if model_name in all_names else False


def hydro_compatability(model):
    # if model object doesn't have attribute self.hydrogen_delay or self.no_hydrogen_focus set them both to True
    if not hasattr(model, "hydrogen_delay"):
        model.hydrogen_delay = True
    if not hasattr(model, "no_hydrogen_focus"):
        model.no_hydrogen_focus = True
    return model


def make_models_hydro_compatible():
    for name in st.session_state.loaded_models.keys():
        st.session_state.loaded_models[name].model = hydro_compatability(st.session_state.loaded_models[name].model)


def load_from_file():
    """
    By default, uploaded files are limited to 200MB.
    You can configure this using the server.maxUploadSize config option.
    For more info on how to set config options, see
    https://docs.streamlit.io/library/advanced-features/configuration#set-configuration-options
    """

    has_cuda = torch.cuda.is_available()
    device_list = ["cpu", "cuda"] if has_cuda else ["cpu"]
    with st.expander(r"$\textsf{\Large Agent Loader}$"):
        # Find file

        _ = st.file_uploader(
            "Upload file", label_visibility="hidden"
        )  # Find a file and copy its path (don't select it!):

        selected_model = st.text_input("or file path:", value=DEFAULT_PATH)
        if selected_model:
            path_from_root = os.path.relpath(selected_model, os.getcwd())
            # st.write(f"You selected: {path_from_root}")

        # Load model
        device = st.radio("Device:", device_list, horizontal=True)  # , index=1 if has_cuda else 0)
        new_name = st.text_input("Give this model a name:", value="validity_agent")
        if st.button("Load agent"):
            if not is_already_loaded(new_name):
                model = load_model(selected_model, device)
                model = hydro_compatability(model)
                st.session_state.loaded_models[new_name] = LoadedModel(
                    model=model, path=selected_model, path_from_root=path_from_root, name=new_name, is_selected=False
                )

                st.write(f"Loaded model {path_from_root} on device {device}.")
            else:
                st.write(f"Model {new_name} is already loaded.")


def load_default_agents():
    models = st.session_state.loaded_models

    models_is_empty = len(models) == 0
    if not models_is_empty:
        return

    # Get models from model_folder using glob
    model_files = glob.glob(str(MODEL_FOLDER / "**/*.model"), recursive=True)

    for model_file in model_files:
        agent_tag = model_file.split("model_objects/")[-1].split("/seed")[0]
        agent_seed = model_file.split("seed_")[-1].split("/")[0]
        if agent_seed != "0":
            continue
        new_name = agent_tag + "-seed" + agent_seed
        path_from_root = os.path.relpath(model_file, os.getcwd())
        model = load_model(model_file, "cpu")
        model = hydro_compatability(model)
        st.session_state.loaded_models[new_name] = LoadedModel(
            model=model, path=model_file, path_from_root=path_from_root, name=new_name, is_selected=False
        )


def display_loaded_agents():

    load_default_agents()

    if st.session_state.loaded_models:
        for model in st.session_state.loaded_models.values():
            st.write(model.name)
            agent_col0, agent_col1 = st.columns([2, 3])
            with agent_col0:
                if st.button("Remove", key=f"unload_model_{model.name}"):
                    st.session_state.loaded_models.pop(model.name)
                    st.rerun()

            with agent_col1:
                if st.button("To Playgrounds", key=f"to_playgrounds_{model.name}"):
                    st.session_state.pm.add_agent_to_pgs(model)
                    st.rerun()

            # if isinstance(model.model, PainnAC):
            #    st.write(f"std: {[np.round(np.exp(v), 3) for v in model.model.log_stds.tolist()]}")
            st.markdown("---")

    else:
        st.write("No models loaded yet.")


if __name__ == "__main__":
    load_from_file()
