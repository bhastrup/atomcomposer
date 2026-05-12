from typing import List, Optional, Dict
import pickle
import os
import time

import streamlit as st

from src.rl.env_container import SimpleEnvContainer
from src.agents.base import AbstractActorCritic
from app_store.pages.tools.agent_selector import LoadedModel

# from app_store.pages.tools.env_selector import LoadedEnv
from src.data.reference_dataloader import ReferenceData

# @dataclasses.dataclass
# class DoubleAgent:
#     """A DoubleAgent should contain a AbstractActorCritic agent and a SimpleEnvContainer"""
#     ac: AbstractActorCritic
#     envs: SimpleEnvContainer # can be multiple copies of the same environment

#     argmax_rollouts: List = None
#     stoch_rollouts: List = None


class DoubleAgent:
    """A DoubleAgent should contain a AbstractActorCritic agent and a SimpleEnvContainer"""

    def __init__(self, ac: AbstractActorCritic, envs: SimpleEnvContainer) -> None:
        self.ac = ac
        self.envs = envs  # can be multiple copies of the same environment

        # # Override envs' action and observation spaces with the ones from the agent
        # for env in self.envs.environments:
        #     env.action_space = ac.action_space
        #     env.observation_space = ac.observation_space

        self.argmax_rollouts: List = None
        self.stoch_rollouts: List = None

        self.stoch_dict: dict = {}  # 'df': {}, 'data': {}}
        self.argmax_dict: dict = {}  # 'df': {}, 'data': {}}
        self.ref_data_collection: Dict[str, ReferenceData] = {}


class Playground:
    """A Playground should contain of pairs of SimpleEnvContainers and Agents"""

    def __init__(self, id: Optional[int] = None):
        """TODO:
        - We should probably have a list of LoadedModels instead of acs, so we have more info about the agent.
        Then we can always get the acs when asking for the double agent.
        - We should just load the bag from the env side. Then the deployment button will instantiate the SimpleEnvContainer for these bags.

        """
        self.double_agents: List[DoubleAgent] = []
        self.id = id
        self.name = f"Playground {id}"

        self.editable = True
        self.deployable = False
        self.saving = False

        self.envs: SimpleEnvContainer = None
        self.agents: List[LoadedModel] = []

    def change_deploy_status(self):
        self.deployable = not self.deployable
        if self.deployable:
            for da in self.double_agents:
                for env in da.envs.environments:
                    env.action_space = da.ac.action_space
                    env.observation_space = da.ac.observation_space

    def change_edit_status(self):
        self.editable = not self.editable
        if self.editable:
            self.deployable = False

    def set_envs(self, envs):
        assert isinstance(envs, SimpleEnvContainer) and envs.get_size() == 1, f"Env size: {envs.get_size()}"
        self.envs = envs
        for da in self.double_agents:
            da.envs = envs

    # def set_envs(self, loaded_env: LoadedEnv):
    #     # TODO: Watchout: pg has loaded_env, but da has envs. Slightly confusing.
    #     assert isinstance(loaded_env, LoadedEnv) and loaded_env.env_container.get_size() == 1
    #     self.loaded_envs = loaded_env
    #     for da in self.double_agents:
    #         da.envs = loaded_env.env_container

    def add_agent(self, agent: LoadedModel):
        self.agents.append(agent)
        self.double_agents.append(DoubleAgent(ac=agent.model, envs=self.envs))

    def remove_agent(self, index):
        self.agents.pop(index)

    def remove_envs(self):
        self.envs = None

    def display_envs(self):
        if self.envs is None:
            return None  # st.write('No environment selected')
        if self.envs.get_size() > 1:
            raise NotImplementedError("Multiple environments not yet supported")

        assert self.envs.get_size() == 1

        formulas = self.envs.environments[0].formulas  # Wow, naming getting so ugly
        assert formulas is not None

        if len(formulas) == 1:
            # env_name = str(formulas) # TODO: make this a string instead of a formula
            env_name = self.envs.name
            if self.editable:
                if st.button(f"🌍 {env_name}", key=f"env_agent_{self.id}", help="Click to remove"):
                    self.remove_envs()
                    st.rerun()
            else:
                if st.button(
                    f"🌍 {env_name}", key=f"env_agent_{self.id}", help=f"{self.envs.environments}", disabled=True
                ):
                    pass
        else:
            env_name = f"{len(formulas)} formulas"
            with st.expander(env_name, expanded=False):
                for formula in formulas:
                    st.write(formula)

    def display_agents(self):
        n_agents = len(self.agents)
        if n_agents == 0:
            st.write("No agents selected")
            return None
        agent_cols = st.columns(n_agents)
        for i, ag in enumerate(self.agents):
            with agent_cols[i]:
                if self.editable:
                    if st.button(f"🤖 {ag.name}", key=f"agent_{self.id}_{i}", help="Click to remove"):
                        self.remove_agent(i)
                        st.rerun()
                else:
                    if st.button(
                        f"🤖 {ag.name}", key=f"agent_{self.id}_{i}", help=f"{ag.path_from_root}", disabled=True
                    ):
                        pass

    def show_playground(self):
        _, mid_col, _ = st.columns([1, 4, 1])
        with mid_col:
            with st.expander(f"{self.name} 🎢", expanded=True):
                # Environment
                if self.envs is not None:
                    env_agent_cols = st.columns([1, 3, 1])
                    with env_agent_cols[1]:
                        self.display_envs()

                # Agents
                st.write("")
                self.display_agents()

    def save_playground(self):
        directory = "saved_playgrounds/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        name = self.name
        use_default_name = st.radio(
            label=f"Use current name? ({name})",
            options=["Yes", "No"],
            index=None,
            key=f"use_default_name_save{self.id}",
        )

        if use_default_name == "Yes":
            self.check_and_write(directory, name)
        elif use_default_name == "No":
            new_name = st.text_input("Name", placeholder="my_playground", value=name)
            if new_name == name:
                st.write("Type new name")
            else:
                if st.button("Download!", key="save_playground"):
                    self.check_and_write(directory, new_name)

    def check_and_write(self, directory: str, name: str):
        path_name = f"{directory}/{name}.pickle"

        if os.path.exists(path_name):
            st.write("Name already exists!")
            decision = st.radio(
                label="Overwrite or rename?", options=["Rename", "Overwrite"], index=None
            )  # key=f'overwrite_{self.name}
            if decision == "Overwrite":
                self.download_pg_spinner(path_name)
            if decision == "Rename":
                new_name = st.text_input("Type a new name", value=name)
                if new_name != name:
                    self.check_and_write(directory, new_name)
        else:
            self.download_pg_spinner(path_name)

    def download_pg_spinner(self, path_name: str):
        with st.spinner("Downloading PlayGround..."):
            time.sleep(1)
            with open(path_name, "wb") as f:
                pickle.dump(self, f)


class PlaygroundManager:
    """A PlaygroundManager should contain a list of Playgrounds"""

    def __init__(self):
        self.playgrounds = []

    def new_id(self):
        if self.playgrounds:
            return self.playgrounds[-1].id + 1
        else:
            return 0

    def create_new_playground(self):
        new_id = self.new_id()
        self.playgrounds.append(Playground(id=new_id))

    def purge_playground(self, i):
        self.playgrounds.pop(i)

    def playgrounds_GUI(self):
        with st.expander(r"$\textsf{\Large Playgrounds 🎡}$", expanded=True):
            for i, pg in enumerate(self.playgrounds):
                id = pg.id
                cols = st.columns([3, 7, 4])
                with cols[0]:
                    # Use a radio button to indicate whether we're editing or not
                    index = 0 if pg.editable else 1
                    st.radio(
                        label="Edit/Lock",
                        options=["Edit ✏️", "🆗"],
                        index=index,
                        key=f"edit_pg_{id}",
                        on_change=pg.change_edit_status,
                        label_visibility="collapsed",
                    )

                    deployable = not pg.editable and pg.envs is not None and pg.agents
                    st.checkbox(
                        label="Deploy",
                        value=pg.deployable,
                        key=f"deploy_pg{id}",
                        on_change=pg.change_deploy_status,
                        disabled=not deployable,
                    )
                    if pg.deployable:
                        st.header("🟢")

                    # if st.button('Save 💾', key=f'save_playground_{id}'):
                    #     pg.saving = True
                    # if pg.saving == True:
                    #     pg.save_playground()

                with cols[1]:
                    pg.show_playground()

                with cols[2]:
                    if st.button("Delete ☠️", key=f"delete_playground_{id}"):
                        self.purge_playground(i)
                        st.rerun()

                    with st.expander("Download 💾"):
                        pg.save_playground()

                st.markdown("---")

            new_col, load_col = st.columns([1, 1])
            with new_col:
                if st.button("New Playground 💫"):
                    st.session_state.pm.create_new_playground()
                    st.rerun()
            with load_col:
                with st.expander(" Load from file 📤"):
                    pickle_file = st.file_uploader("Upload file", label_visibility="hidden", key="load_playground")
                    if pickle_file:
                        loaded_pg = pickle.load(pickle_file)
                        if st.button("Load", key=f"load_playground_{loaded_pg.id}"):
                            loaded_pg.id = self.new_id()
                            self.playgrounds.append(loaded_pg)
                            st.rerun()

    def add_agent_to_pgs(self, agent: LoadedModel):
        for pg in self.playgrounds:
            if pg.editable and agent not in pg.agents:
                pg.add_agent(agent)

    def add_env_to_pgs(self, envs: SimpleEnvContainer):
        # TODO: One could alse make a LoadedEnv that contains the envs and the name (str formulas)
        # TODO: Instead of letting LoadedEnv contain an env_container, we could simply inherit from SimpleEnvContainer
        for pg in self.playgrounds:
            if pg.editable and pg.envs is None:
                pg.set_envs(envs)
